
import scipy.special
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import delu # the new version of zero package

from .models import GCN, SAGE
from myutils import Utils

class GNN():
    
    '''
    The class for running the baselines for GNN and SAGE, supervised graph methods.

    :param seed: int
        random seed
    :param model_name: str
        'GCN' or 'SAGE'
    :param loss_name: str, optional
        'nll' or 'bce'
    :param n_epochs: int, optional
        number of epochs
    :param in_channels: int, optional
        number of input features
    '''
     
    def __init__(self, seed:int, model_name:str, loss_name = 'nll' ,n_epochs=200,in_channels=17):


        self.seed = seed
        self.model_name = model_name
        self.utils = Utils()

        # device
        self.device = self.utils.get_device(gpu_specific=True)


        # hyper-parameter
        self.n_epochs = n_epochs # default is 1000
        self.loss_name = loss_name #nll or bce
        
        if self.loss_name == 'nll':
            d_out = 2
        else:
            d_out = 1

        # model init
        if self.model_name == 'GCN':
            gcn_parameters = {'num_layers': 2
                , 'hidden_channels': 128
                , 'dropout': 0.0
                , 'batchnorm': False}
            model_para = gcn_parameters.copy()
            self.model = GCN(in_channels=in_channels, out_channels=d_out, **model_para)
            self.lr = 0.01
            self.weight_decay =5e-7

        elif self.model_name == 'SAGE':
            sage_parameters = {'num_layers': 2
                , 'hidden_channels': 128
                , 'dropout': 0.0
                , 'batchnorm': False}
            model_para = sage_parameters.copy()
            self.model = SAGE(in_channels=in_channels, out_channels=d_out, **model_para)
            self.lr = 0.01
            self.weight_decay = 5e-7

        else:
            raise NotImplementedError

    def apply_model(self, data,train_idx):
        '''
        apply the model to the data

        :param data: torch_geometric.data.Data
            The graph data.
        :param train_idx: torch.Tensor
            training index
        '''

        return self.model(data.x, data.adj_t)[train_idx]

    @torch.no_grad()
    def evaluate(self, data,val_idx):
        '''
        evaluate the model

        :param data: torch_geometric.data.Data
            The graph data
        :param val_idx: torch.Tensor
            validation index
        '''

        self.model.eval()
        score = []

        # get the output of the model
        out = self.apply_model(data,val_idx)
        if self.loss_name == 'nll':
            out = out[:, 1]
        score.append(out)
        if self.loss_name == 'nll':
            score = torch.cat(score).cpu().numpy()
        else:
            score = torch.cat(score).squeeze(1).cpu().numpy()
        score = scipy.special.expit(score)

        # calculate the metric
        if data.y is not None:
            target = data.y[val_idx].cpu().numpy()
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

    def load_model(self, path,data=None):
        '''
        load the model

        :param path: str
            path to load the model
        :param data: torch_geometric.data.Data
            The graph data
        '''

        self.model.load_state_dict(torch.load(path+'.pt'))
        self.model.to(self.device)

        # add adjancy matrix to the data
        data = T.ToSparseTensor(remove_edge_index= False)(data)
        data.adj_t = data.adj_t.to_symmetric()

    def fit(self, data, indexes):
        '''
        fit the model

        :param data: torch_geometric.data.Data  
            The graph data.
        :param indexes: torch.Tensor
            training index
        '''

        # set seed
        self.utils.set_seed(self.seed)
        
        data = T.ToSparseTensor(remove_edge_index= False)(data)
        data.adj_t = data.adj_t.to_symmetric()

        # training set is used as the validation set in the anomaly detection task
        data = data.to(self.device)
        train_idx = indexes.to(self.device)
        val_idx = train_idx


        task_type = 'binclass'
        
        # move the model to the device
        self.model.to(self.device)

        # create an optimizer
        optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        )

        # Create a progress tracker for early stopping
        # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=100)

        if self.loss_name == 'nll':
            binloss = F.nll_loss
        else:
            binloss = F.binary_cross_entropy_with_logits

        # loss function
        loss_fn = (
            binloss
            if task_type == 'binclass'
            else F.cross_entropy
            if task_type == 'multiclass'
            else F.mse_loss
        )


        # training
        for epoch in range(1, self.n_epochs + 1):

            self.model.train()
            optimizer.zero_grad()
            out = self.apply_model(data,train_idx)
            if self.loss_name == 'bce':
                out = out.squeeze(1)
            loss = loss_fn(out, data.y[train_idx])

            loss.backward()
            optimizer.step()

            _, val_metric = self.evaluate(data,val_idx)
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
        predict the score
        
        :param data: torch_geometric.data.Data
            The graph data.
        :param indexes: torch.Tensor
            testing index
        '''

        data = data.to(self.device)
        predict_idx = indexes.to(self.device)
        score, _ = self.evaluate(data,predict_idx)
        return score



