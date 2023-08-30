import math
import torch
from torch import nn
from torch_geometric.nn import MLP
from .layers.conv import NeighDiff


class DONEBase(nn.Module):
    '''
    Deep Outlier Aware Attributed Network Embedding
    (https://arxiv.org/abs/2103.15054)

    :param x_dim : int
        Input dimension of attribute.
    :param s_dim : int
        Input dimension of structure.
    :param hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    :param num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    :param dropout : float, optional
        Dropout rate. Default: ``0.``.
    :param weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    :param act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    :param w1 : float, optional
        Weight of structure proximity loss. Default: ``0.2``.
    :param w2 : float, optional
        Weight of structure homophily loss. Default: ``0.2``.
    :param w3 : float, optional
        Weight of attribute proximity loss. Default: ``0.2``.
    :param w4 : float, optional
        Weight of attribute homophily loss. Default: ``0.2``.
    :param w5 : float, optional
        Weight of combination loss. Default: ``0.2``.
    **kwargs
        Other parameters for the backbone.
    '''

    def __init__(self,
                 x_dim,
                 s_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 **kwargs):
        super(DONEBase, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.attr_encoder = MLP(in_channels=x_dim,
                                hidden_channels=hid_dim,
                                out_channels=hid_dim,
                                num_layers=encoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.attr_decoder = MLP(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                out_channels=x_dim,
                                num_layers=decoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.struct_encoder = MLP(in_channels=s_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.struct_decoder = MLP(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=s_dim,
                                  num_layers=decoder_layers,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.neigh_diff = NeighDiff()
        self.emb = None

    def forward(self, x, s, edge_index):
        ''''
        Forward computation.

        :param x : torch.Tensor
            Input attribute embeddings.
        :param s : torch.Tensor
            Input structure embeddings.
        :param edge_index : torch.Tensor
            Edge index.
        '''
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()
        self.emb = (h_a, h_s)

        return x_, s_, h_a, h_s, dna, dns

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns):
        '''
        Loss function for DONE.

        :param x : torch.Tensor
            Input attribute embeddings.
        :param x_ : torch.Tensor
            Reconstructed attribute embeddings.
        :param s : torch.Tensor
            Input structure embeddings.
        :param s_ : torch.Tensor
            Reconstructed structure embeddings.
        :param h_a : torch.Tensor
            Attribute hidden embeddings.
        :param h_s : torch.Tensor
            Structure hidden embeddings.
        :param dna : torch.Tensor
            Attribute neighbor distance.
        :param dns : torch.Tensor
            Structure neighbor distance.
        '''

        # equation 9 is based on the official implementation, and it
        # is slightly different from the paper
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = self.w3 * dx + self.w4 * dna
        oa = tmp / torch.sum(tmp)

        # equation 8 is based on the official implementation, and it
        # is slightly different from the paper
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.w1 * ds + self.w2 * dns
        os = tmp / torch.sum(tmp)

        # equation 10
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)

        # equation 4
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)

        # equation 5
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)

        # equation 2
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)

        # equation 3
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)

        # equation 6
        loss_c = torch.mean(torch.log(torch.pow(oc, -1)) * dc)

        # equation 7
        loss = self.w3 * loss_prox_a + \
               self.w4 * loss_hom_a + \
               self.w1 * loss_prox_s + \
               self.w2 * loss_hom_s + \
               self.w5 * loss_c

        return loss, oa, os, oc

    @staticmethod
    def process_graph(data,max_num_node):
        '''
        Obtain the dense adjacency matrix of the graph.

        :param data : torch_geometric.data.Data
            Input graph.
        :param max_num_node : int  
            Maximum number of nodes in the graph.
        '''
        nodes = data.n_id
        neigh_batch = torch.zeros(len(nodes), max_num_node.item()+1).to(data.edge_index.device)

        column_idx = []
    
        for n in range(len(nodes)):
            mask = data.edge_index[1] == n
            column_idx = data.n_id[data.edge_index[0][mask]]
            neigh_batch[n][column_idx] = 1
        for n in range(len(nodes)):
            mask = data.edge_index[0] == n
            column_idx = data.n_id[data.edge_index[1][mask]]
            neigh_batch[n][column_idx] = 1
        data.s = neigh_batch
