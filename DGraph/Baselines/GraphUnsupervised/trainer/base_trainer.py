'''
Based on PyGOD repo: https://github.com/pygod-team/pygod/
'''

from inspect import signature
from abc import ABC, abstractmethod

import torch
import numpy as np

from torch_geometric.nn import GIN
from torch_geometric import compile
from torch_geometric.loader import NeighborLoader

from myutils import Utils



class DeepDetector(ABC):
    '''
    Abstract class for deep outlier detection algorithms.

    :param hid_dim : int, optional
      Hidden dimension of model. Default: ``64``.
    :param num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    :param dropout : float, optional
        Dropout rate. Default: ``0.``.
    :param weight_decay : float, optional
      Weight decay (L2 penalty). Default: ``0.``.
    :param act :  callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    :param backbone :  torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GIN``.
    :param contamination : float, optional
      The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    :param lr : float, optional
        Learning rate. Default: ``0.004``.
    :param epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    :param gpu : int, optional
        GPU Index, -1 for using CPU. Default: ``-1``.
    :param batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    :param num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    :param gan :  bool, optional
        Whether using adversarial training. Default: ``False``.
    :param verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    :param save_emb :  bool, optional
        Whether to save the embedding. Default: ``False``.
    :param compile_model :  bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs
        Other parameters for the backbone model.
    '''

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GIN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 gan=False,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):


        self.verbose = verbose

        # The outlier scores of the training data. Outliers tend to have
        # higher scores. This value is available once the detector is
        # fitted.
        self.decision_score_ = None

        # model param
        self.in_dim = None
        self.num_nodes = None
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.backbone = backbone
        self.kwargs = kwargs

        # training param
        self.lr = lr
        self.epoch = epoch
        self.utils = Utils()
        self.contamination = contamination

        # device
        self.device = self.utils.get_device(gpu_specific=True)
        self.batch_size = batch_size
        self.gan = gan
        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

        # other param
        
        self.model = None
        
        # The learned node hidden embeddings of shape
        # :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        # is ``True``. When the detector has not been fitted, ``emb`` is
        # ``None``. When the detector has multiple embeddings,
        # ``emb`` is a tuple of torch.Tensor.

        if save_emb:
            self.emb = None
        self.save_emb = save_emb
        self.compile_model = compile_model

    def load_model(self, path, data):
        '''
        load the model.

        :param path: str
            The path of the model.
        :param data: torch_geometric.data.Data
            The data used to load the model.
        '''

        self.num_nodes, self.in_dim = data.x.shape
        self.model = self.init_model(**self.kwargs)
        self.model.load_state_dict(torch.load(path+'.pt'))
        self.model.to(self.device)
        if self.batch_size == 0:
            self.batch_size = 16


    def save_model(self, path):
        '''
        save the model.

        :param path: str
            The path of the model.
        '''
        torch.save(self.model.state_dict(), path+'.pt')


    def fit(self, data, mask=None):
        '''
        Fit the detector with input data.

        :param data:  torch_geometric.data.Data
            The data used to fit the model.
        :param mask: list, optional
            The mask of the data used to fit the model.
        '''
        self.num_nodes, self.in_dim = data.x.shape
        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if mask is None:
            data_len = data.x.shape[0]
        else:
            
            data_len = data.x[mask].shape[0]
        if self.batch_size == 0:
            self.batch_size = 16

        if mask is None:
            loader = NeighborLoader(data,
                                    self.num_neigh,
                                    batch_size=self.batch_size)
        else:
            loader = NeighborLoader(data,
                                    self.num_neigh,
                                    batch_size=self.batch_size,
                                    input_nodes=mask)        
        if not self.gan:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_g = torch.optim.Adam(self.model.generator.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.model.discriminator.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.model.train()
        
        self.decision_score_ = torch.zeros(data.x.shape[0])
        
        for epoch in range(self.epoch):
            epoch_loss = 0
            i = 0
            if self.gan:
                self.epoch_loss_g = 0
            for sampled_data in loader:
                self.process_graph(sampled_data,data.edge_index.max())
                i += 1
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id
                loss, score = self.forward_model(sampled_data)
                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) == tuple:
                        self.emb[0][node_idx[:batch_size]] = \
                            self.model.emb[0][:batch_size].cpu()
                        self.emb[1][node_idx[:batch_size]] = \
                            self.model.emb[1][:batch_size].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = \
                            self.model.emb[:batch_size].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i == 2000:
                    break
                
            loss_value = epoch_loss / data_len
            if self.gan:
                loss_value = (self.epoch_loss_g / data_len, loss_value)

        self._process_decision_score()
        return self

    def decision_function(self, data, mask = None,label=None):
        '''
        compute the anomaly score of the input nodes.

        :param data:  torch_geometric.data.Data
          The data used to fit the model.
        :param mask: list, optional
            The mask of the data used to fit the model.
        :return: The anomaly score of the input nodes.
        '''
        
        if mask is None:
            loader = NeighborLoader(data,
                                    self.num_neigh,
                                    batch_size=self.batch_size)
        else:
            loader = NeighborLoader(data,
                                    self.num_neigh,
                                    batch_size=self.batch_size,
                                    input_nodes=mask)


        self.model.eval()
        if mask is None:
            data_len = data.x.shape[0]
        else:
            data_len = data.x[mask].shape[0]
        
        outlier_score = torch.zeros(data.x.shape[0])
        if self.save_emb:
            if type(self.hid_dim) is tuple:
                self.emb = (torch.zeros(data.x.shape[0], self.hid_dim[0]),
                            torch.zeros(data.x.shape[0], self.hid_dim[1]))
            else:
                self.emb = torch.zeros(data.x.shape[0], self.hid_dim)        
        i = 0
        for sampled_data in loader:
            i+=1
            self.process_graph(sampled_data,data.edge_index.max())
            loss, score = self.forward_model(sampled_data)
            
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id
            if self.save_emb:
                if type(self.hid_dim) is tuple:
                    self.emb[0][node_idx[:batch_size]] = \
                        self.model.emb[0][:batch_size].cpu()
                    self.emb[1][node_idx[:batch_size]] = \
                        self.model.emb[1][:batch_size].cpu()
                else:
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()

            outlier_score[node_idx[:batch_size]] = score

        if mask is None:
            return outlier_score
        else:
            return outlier_score[mask]

    def predict(self,
                data=None,
                mask=None,
                label=None,
                return_pred=True):
        '''
        Predict the anomaly score of nodes in testing data.
        
        :param data : torch_geometric.data.Data, optional
            The testing graph. If ``None``, the training data is used.
            Default: ``None``.
        :param label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.
        :param return_pred : bool, optional
            Whether to return the predicted binary labels. The labels
            are determined by the outlier contamination on the raw
            outlier scores. Default: ``True``.
        '''

        output = ()
        if data is None:
            score = self.decision_score_

        else:
            score = self.decision_function(data, mask ,label)

        if return_pred:
            pred = (score > self.threshold_).long()
            output += (pred,)
        
        if len(output) == 1:
            output = output[0]


        return output
    
    def _process_decision_score(self):
        '''
        Internal function to process the raw anomaly scores.
        '''

        self.threshold_ = np.percentile(self.decision_score_,
                                        100 * (1 - self.contamination))
        
        # The binary labels of the training data. 0 stands for inliers and 1 for outliers. 
        # It is generated by applying ``threshold_`` on ``decision_score_``.
        self.label_ = (self.decision_score_ > self.threshold_).long()

    @abstractmethod
    def init_model(self):
        """
        Initialize the neural network detector.

        Returns
        -------
        model : torch.nn.Module
            The initialized neural network detector.
        """

    @abstractmethod
    def forward_model(self, data):
        """
        Forward pass of the neural network detector.
        :param data : torch_geometric.data.Data
            The input graph.
        """

    @abstractmethod
    def process_graph(self, data):
        """
        Data preprocessing for the input graph.
        :param data : torch_geometric.data.Data
            The input graph.
        """

   