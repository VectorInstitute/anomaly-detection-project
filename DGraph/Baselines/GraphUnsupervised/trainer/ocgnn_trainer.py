'''
Based on PyGOD repo: https://github.com/pygod-team/pygod/
'''
import torch
from torch_geometric.nn import GCN

from . import DeepDetector
from ..models import OCGNNBase


class OCGNN(DeepDetector):
    """
    One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks

    Parameters
    ----------
    :param hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    :param num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    :param dropout : float, optional
        Dropout rate. Default: ``0.``.
    :param weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    :param act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    :param backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    :param contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    :param lr : float, optional
        Learning rate. Default: ``0.004``.
    :param epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    :param gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    :param batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    :param num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    :param beta : float, optional
        The weight between the reconstruction loss and radius.
        Default: ``0.5``.
    :param warmup : int, optional
        The number of epochs for warm-up training. Default: ``2``.
    :param eps : float, optional
        The slack variable. Default: ``0.001``.
    :param verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    :param save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    :param compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs
        Other parameters for the backbone model.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):
        self.beta = beta
        self.warmup = warmup
        self.eps = eps
        super(OCGNN, self).__init__(hid_dim=hid_dim,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    backbone=backbone,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    save_emb=save_emb,
                                    compile_model=compile_model,
                                    **kwargs)


    def process_graph(self, data,max_num_node):
        pass

    def init_model(self, **kwargs):
        '''
        Initialize the model.
        '''
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)

        return OCGNNBase(in_dim=self.in_dim,
                         hid_dim=self.hid_dim,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         beta=self.beta,
                         warmup=self.warmup,
                         eps=self.eps,
                         backbone=self.backbone,
                         device=self.device,
                         **kwargs).to(self.device)

    def forward_model(self, data):
        ''''
        Forward the model.
        
        :param data: torch_geometric.data.Data
            The input data.'''

        batch_size = data.batch_size

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        emb = self.model(x, edge_index)
        loss, score = self.model.loss_func(emb[:batch_size])

        return loss, score.detach().cpu()
