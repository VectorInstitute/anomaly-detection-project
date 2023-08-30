import torch
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj

from .done import DONEBase


class AdONEBase(torch.nn.Module):
    '''
    Adversarial Outlier Aware Attributed Network Embedding (AdONE)

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
        Weight of alignment loss. Default: ``0.2``.
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
        super(AdONEBase, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        self.done = DONEBase(x_dim=x_dim,
                             s_dim=s_dim,
                             hid_dim=hid_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             act=act,
                             w1=self.w1,
                             w2=self.w2,
                             w3=self.w3,
                             w4=self.w4,
                             w5=self.w5,
                             **kwargs)

        self.discriminator = MLP(in_channels=hid_dim,
                                 hidden_channels=int(hid_dim / 2),
                                 out_channels=1,
                                 num_layers=2,
                                 dropout=dropout,
                                 act=torch.tanh)
        self.emb = None

    def forward(self, x, s, edge_index):
        '''
        Forward computation.

        :param x : torch.Tensor
            Input attribute embeddings.
        :param s : torch.Tensor
            Input structure embeddings.
        :param edge_index : torch.Tensor
            Edge index.
        '''
        x_, s_, h_a, h_s, dna, dns = self.done(x, s, edge_index)
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        self.emb = (h_a, h_s)

        return x_, s_, h_a, h_s, dna, dns, dis_a, dis_s

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns, dis_a, dis_s):
        '''
        Loss function for AdONE.

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
        :param  dns : torch.Tensor
            Structure neighbor distance.
        :param dis_a : torch.Tensor
            Attribute discriminator score.
        :param dis_s : torch.Tensor
            Structure discriminator score.

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

        # equation 12
        loss_alg = torch.mean(torch.log(torch.pow(oc, -1))
                              * (-torch.log(1 - dis_a) - torch.log(dis_s)))

        # equation 13
        loss = self.w3 * loss_prox_a + \
               self.w4 * loss_hom_a + \
               self.w1 * loss_prox_s + \
               self.w2 * loss_hom_s + \
               self.w5 * loss_alg

        return loss, oa, os, oc

    @staticmethod
    def process_graph(data,max_num_node):
        '''
        Obtain the dense adjacency matrix of the graph.

        :param data : torch_geometric.data.Data
            Input graph.
        :param max_num_node : int
            Maximum number of nodes in the dataset.
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
