
import scipy.special
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import delu # the new version of zero package
import torch.optim as optim

from .models import AMNet_model
from myutils import Utils


class AMNet():
    '''
    The class for running the baselines for AMNet, supervised graph methods.

    :param seed: int
        random seed
    :param model_name: str
        name of the model
    :param n_epochs: int, optional
        number of epochs
    :param in_channels: int, optional
        number of input features
    '''
    def __init__(self, seed:int, model_name:str, n_epochs=200,in_channels=17):
        

        self.seed = seed
        self.model_name = model_name
        self.utils = Utils()

        # parameters configuration
        self.params_config = {
            'K': 2,
            'M': 3,
            'hidden_channels': 32,
            'lr_f': 5e-2,
            'lr': 5e-4,
            'weight_decay': 5e-6,
            'beta': 1.,
            'patience': 10
        }

        # device
        self.device = self.utils.get_device(gpu_specific=True)
        print(self.device)
        print('\n')

        # hyper-parameter
        self.n_epochs = n_epochs # default is 1000
        
        # model init
        self.model = AMNet_model(in_channels = in_channels, hid_channels=self.params_config['hidden_channels'], num_class=2,
                   K=self.params_config['M'], filter_num=self.params_config['K'])
        
        self.beta = 0.5

    def apply_model(self, data,label):

        '''
        apply the model to the data

        :param data: torch_geometric.data.Data
            data object
        :param label: torch.tensor
            label of the data
        '''
        return self.model(data.x, data.edge_index, label=label)

    @torch.no_grad()
    def evaluate(self, data,label,val_idx,indexes):
        '''
        evaluate the model

        :param data:  torch_geometric.data.Data
            data object
        :param label: torch.tensor
            label of the data
        :param val_idx: torch.tensor
            index of the validation set
        :param indexes: torch.tensor
            index of the training set
        '''

        self.model.eval()
        score = []
        # get the output of the model
        out = self.apply_model(data,label)
        out = F.softmax(out[indexes], dim=1)[:, 1]
        
        score.append(out)

        score = torch.cat(score).cpu().numpy()
        score = scipy.special.expit(score)


        # calculate the metric
        if data.y is not None:
            target = data.y[indexes].cpu().numpy()
            metric = self.utils.metric(y_true=target, y_score=score)
        else:
            metric = {'aucroc': None, 'aucpr': None}

        return score, metric['aucroc']
    
    def save_model(self, path):
        '''
        save the model

        :param path: str
            path to save the model
        '''

        torch.save(self.model.state_dict(), path+'.pt')

    def load_model(self, path, data=None):
        '''
        load the model

        :param path: str
            path to load the model
        :param data: torch_geometric.data.Data, optional
            data object
        '''

        self.model.load_state_dict(torch.load(path+'.pt'))
        self.model.to(self.device)

    def fit(self, data, indexes):

        '''
        fit the model

        :param data: torch_geometric.data.Data
            data object
        :param indexes: torch.tensor
            index of the training set
        '''

        # set seed
        self.utils.set_seed(self.seed)

        # training set is used as the validation set in the anomaly detection task
        data = data.to(self.device)
        
        
        # move the model to the device
        self.model.to(self.device)

        # optimizer
        optimizer = optim.Adam([
          dict(params=self.model.filters.parameters(), lr=self.params_config['lr_f']),
          dict(params=self.model.lin, lr=self.params_config['lr'], weight_decay=self.params_config['weight_decay']),
          dict(params=self.model.attn, lr=self.params_config['lr'], weight_decay=self.params_config['weight_decay'])]

     )

        # Create a progress tracker for early stopping
        # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=self.params_config['patience'])

        weights = torch.Tensor([1., 1.])
        
        # loss function
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))

        # get the label of the training set and validation set
        anomaly = (data.y == 1).squeeze()
        normal = (data.y == 0).squeeze()
        train_idx = torch.zeros(normal.shape, dtype=torch.bool).to(self.device)
        train_idx[indexes] = 1

        idx_val = train_idx

        task_type = 'binclass'

        label_train = (train_idx & anomaly,train_idx & normal)
        label_val = (idx_val & anomaly,idx_val & normal)

        for epoch in range(1, self.n_epochs + 1):

            self.model.train()
            optimizer.zero_grad()
            
            # get the output of the model
            out, bias_loss = self.apply_model(data,label_train)

            loss = loss_fn(out[indexes], data.y.squeeze()[indexes])+ bias_loss * self.beta

            loss.backward()
            optimizer.step()
        
            # evaluate the model
            _, val_metric = self.evaluate(data,label_val,idx_val,indexes)
            print(f'Epoch {epoch:03d} | Validation metric: {val_metric:.4f}', end='')

          
            progress.update((-1 if task_type == 'regression' else 1) * val_metric)
            if progress.success:
                print(' <<< BEST VALIDATION EPOCH', end='')
            print()
            if progress.fail:
                break
        return self

    def predict_score(self, data, indexes):

        '''
        predict the score of the data
        
        :param data: torch_geometric.data.Data
            data object
        :param indexes: torch.tensor
            index of the training set
        '''
        
        data = data.to(self.device)
        anomaly = (data.y == 1).squeeze()
        normal = (data.y == 0).squeeze()
        idx_predict = torch.zeros(normal.shape, dtype=torch.bool).to(self.device)
        idx_predict[indexes] = 1
        label_predict = (idx_predict & anomaly,idx_predict & normal)
        score, test_metric= self.evaluate(data,label_predict,idx_predict,indexes)
        return score



