
import rtdl
import scipy.special
import torch
import torch.nn.functional as F
# import zero
import delu # the new version of zero package


from myutils import Utils

class DeepCNN():
    '''
    The original code: https://yura52.github.io/rtdl/stable/index.html
    The original paper: "Revisiting Deep Learning Models for Tabular Data", NIPS 2019

    The class for running the baselines for DeepCNN, supervised tabular methods.

    :param seed: int
        random seed
    :param model_name: str
        'ResNet' or 'MLP' or 'FTTransformer'
    :param loss_name: str, optional
        'nll' or 'bce'
    :param n_epochs: int, optional
        number of epochs
    :param d_in: int, optional
        number of input features
    :param batch_size: int, optional
        batch size
    '''
    def __init__(self, seed:int, model_name:str, loss_name = 'nll' ,n_epochs=100, batch_size=64,d_in = 17):

        self.seed = seed
        self.model_name = model_name
        self.utils = Utils()

        
        self.device = self.utils.get_device(gpu_specific=True)
        print(self.device)
        print('\n')

        # Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
        # zero.improve_reproducibility(seed=self.seed)
        delu.improve_reproducibility(base_seed=int(self.seed))

        # hyper-parameter
        self.n_epochs = n_epochs # default is 1000
        self.batch_size = batch_size # default is 256
        self.loss_name = loss_name #nll or bce

        if self.loss_name == 'nll':
            d_out = 2
        else:
            d_out = 1
        
        if self.model_name == 'ResNet':
            self.model = rtdl.ResNet.make_baseline(
                d_in=d_in,
                d_main=128,
                d_hidden=256,
                dropout_first=0.2,
                dropout_second=0.0,
                n_blocks=2,
                d_out=d_out,
            )
            self.lr = 0.001
            self.weight_decay = 0.0

        elif self.model_name == 'MLP':
            self.model = rtdl.MLP.make_baseline(
                d_in=d_in,
                d_layers=[128],
                dropout=0.0,
                d_out=d_out,
            )
            self.lr = 0.01
            self.weight_decay = 5e-7

        elif self.model_name == 'FTTransformer':
            self.model = rtdl.FTTransformer.make_default(
                n_num_features=d_in,
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                d_out=d_out,
            )

        else:
            raise NotImplementedError
        

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
        :param data: torch.Tensor
            The tabular data
        '''
        self.model.load_state_dict(torch.load(path+'.pt'))
        self.model.to(self.device)
    
    def apply_model(self, x_num, x_cat=None):
        if isinstance(self.model, rtdl.FTTransformer):
            if self.loss_name == 'nll':
                return F.log_softmax(self.model(x_num, x_cat),dim=-1)
            else:
                return self.model(x_num, x_cat)
        elif isinstance(self.model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            if self.loss_name == 'nll':
                return F.log_softmax(self.model(x_num),dim=-1)
            else:
                return self.model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(self.model)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def evaluate(self, X, y=None):
        '''
        evaluate the model
        :param X: torch.Tensor
            The tabular data
        :param y: torch.Tensor
            The label
        '''
        self.model.eval()
        score = []
        # for batch in delu.iter_batches(X[part], 1024):

        for batch in delu.iter_batches(X, self.batch_size):
            out = self.apply_model(batch)
            if self.loss_name == 'nll':
                out = out[:, 1]
            score.append(out)
        if self.loss_name == 'nll':
            score = torch.cat(score).cpu().numpy()
        else:
            score = torch.cat(score).squeeze(1).cpu().numpy()
        score = scipy.special.expit(score)

        # calculate the metric
        if y is not None:
            target = y.cpu().numpy()
            metric = self.utils.metric(y_true=target, y_score=score)
        else:
            metric = {'aucroc': None, 'aucpr': None}

        return score, metric['aucroc']

    def fit(self, X_train, y_train, ratio=None):
        '''
        fit the model

        :param data: torch.Tensor
            The tabular data.
        :param indexes: torch.Tensor
            training index
        '''

        # set seed
        self.utils.set_seed(self.seed)

        # training set is used as the validation set in the anomaly detection task
        X = {'train': X_train.float().to(self.device),
             'val': X_train.float().to(self.device)}

        if self.loss_name == 'nll':
            y = {'train': y_train.to(torch.int64).to(self.device),
                 'val': y_train.to(torch.int64).to(self.device)}
        else:
            y = {'train': y_train.float().to(self.device),
                 'val': y_train.float().to(self.device)}

        task_type = 'binclass'

        

        self.model.to(self.device)
        optimizer = (
            self.model.make_default_optimizer()
            if isinstance(self.model, rtdl.FTTransformer)
            else torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        )
        
        if self.loss_name == 'nll':
            binloss = F.nll_loss
        else:
            binloss = F.binary_cross_entropy_with_logits

        loss_fn = (
            binloss
            if task_type == 'binclass'
            else F.cross_entropy
            if task_type == 'multiclass'
            else F.mse_loss
        )

        # Create a dataloader for batches of indices
        # Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
        train_loader = delu.data.IndexLoader(len(X['train']), self.batch_size, device=self.device)

        # Create a progress tracker for early stopping
        # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=100)

        # training

        for epoch in range(1, self.n_epochs + 1):
            for iteration, batch_idx in enumerate(train_loader):
                self.model.train()
                optimizer.zero_grad()
                x_batch = X['train'][batch_idx]
                y_batch = y['train'][batch_idx]
                out = self.apply_model(x_batch)
                if self.loss_name == 'bce':
                    out = out.squeeze(1)
                loss = loss_fn(out, y_batch)

                loss.backward()
                optimizer.step()
               
            _, val_metric = self.evaluate(X=X['val'], y=y['val'])
            print(f'Epoch {epoch:03d} | Validation metric: {val_metric:.4f}', end='')
            progress.update((-1 if task_type == 'regression' else 1) * val_metric)
            if progress.success:
                print(' <<< BEST VALIDATION EPOCH', end='')
            print()
            if progress.fail:
                break

        return self

    def predict_score(self, X):
        '''
        predict the score
        
        :param data: torch.Tensor
            The tabular data.
        :param indexes: torch.Tensor
            testing index
        '''

        X = X.float().to(self.device)
        score, _ = self.evaluate(X=X, y=None)
        return score



