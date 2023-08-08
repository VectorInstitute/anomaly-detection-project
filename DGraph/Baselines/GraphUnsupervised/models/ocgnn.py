import torch
import torch.nn as nn
from torch_geometric.nn import GCN


class OCGNNBase(nn.Module):
    '''
    One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks (https://arxiv.org/abs/2011.14369)

    :param in_dim : int
        Input dimension of model.
    :param hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    :param num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    :param dropout : float, optional
        Dropout rate. Default: ``0.``.
    :param act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    :param backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    :param beta : float, optional
        The weight between the reconstruction loss and radius.
        Default: ``0.5``.
    :param warmup : int, optional
        The number of epochs for warm-up training. Default: ``2``.
    :param eps : float, optional
        The slack variable. Default: ``0.001``.
    **kwargs
        Other parameters for the backbone model.
    '''

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 device=None,
                 **kwargs):
        super(OCGNNBase, self).__init__()

        self.beta = beta
        self.warmup = warmup
        self.eps = eps

        self.gnn = backbone(in_channels=in_dim,
                            hidden_channels=hid_dim,
                            num_layers=num_layers,
                            out_channels=hid_dim,
                            dropout=dropout,
                            act=act,
                            **kwargs)

        self.r = 0
        self.c = torch.zeros(hid_dim).to(device)

        self.emb = None

    def forward(self, x, edge_index):
        '''
        Forward computation.

        :param x : torch.Tensor
            Input attribute embeddings.
        :param edge_index : torch.Tensor
            Edge index.

        '''

        self.emb = self.gnn(x, edge_index)
        return self.emb

    def loss_func(self, emb):
        '''
        Loss function for OCGNN

        :param emb : torch.Tensor
            Embeddings.
        '''

        dist = torch.sum(torch.pow(emb - self.c, 2), 1)
        score = dist - self.r ** 2
        loss = self.r ** 2 + 1 / self.beta * torch.mean(torch.relu(score))

        if self.warmup > 0:
            with torch.no_grad():
                self.warmup -= 1
                self.r = torch.quantile(torch.sqrt(dist), 1 - self.beta)
                self.c = torch.mean(emb, 0)
                self.c[(abs(self.c) < self.eps) & (self.c < 0)] = -self.eps
                self.c[(abs(self.c) < self.eps) & (self.c > 0)] = self.eps

        return loss, score
