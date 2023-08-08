'''
Based on PyGOD repo: https://github.com/pygod-team/pygod/
'''

import torch
import warnings

from . import DeepDetector
from ..models import AdONEBase
import os


class AdONE(DeepDetector):
    '''
    Adversarial Outlier Aware Attributed Network Embedding

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
    :param backbone : torch.nn.Module
        The backbone of AdONE is fixed to be MLP. Changing of this
        parameter will not affect the model. Default: ``None``.
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

    '''

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=None,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=1,
                 gpu=-1,
                 batch_size=8,
                 num_neigh=-1,
                 save_emb=False,
                 compile_model=False,
                 verbose=0,
                 **kwargs):

        if backbone is not None:
            warnings.warn("Backbone is not used in AdONE.")

        super(AdONE, self).__init__(hid_dim=hid_dim,
                                    num_layers=1,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
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

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.num_layers = num_layers

        self.attribute_score_ = None
        self.structural_score_ = None
        self.combined_score_ = None

    def process_graph(self, data,max_num_node):
        '''
        Process the graph data.

        :param data: torch_geometric.data.Data
            The graph data.
        :param max_num_node: int
            The maximum number of nodes in the graph.
        '''
        AdONEBase.process_graph(data,max_num_node)

    def init_model(self, **kwargs):

        self.attribute_score_ = torch.zeros(self.num_nodes)
        self.structural_score_ = torch.zeros(self.num_nodes)
        self.combined_score_ = torch.zeros(self.num_nodes)

        if self.save_emb:
            self.emb = (torch.zeros(self.num_nodes, self.hid_dim),
                        torch.zeros(self.num_nodes, self.hid_dim))

        return AdONEBase(x_dim=self.in_dim,
                         s_dim=self.num_nodes,
                         hid_dim=self.hid_dim,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act=self.act,
                         w1=self.w1,
                         w2=self.w2,
                         w3=self.w3,
                         w4=self.w4,
                         w5=self.w5,
                         **kwargs).to(self.device)

    def forward_model(self, data):
        '''
        Forward the model.

        :param data: torch_geometric.data.Data
            The graph data.
        '''
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        x_, s_, h_a, h_s, dna, dns, dis_a, dis_s = self.model(x, s, edge_index)
        loss, oa, os, oc = self.model.loss_func(x[:batch_size],
                                                x_[:batch_size],
                                                s[:batch_size],
                                                s_[:batch_size],
                                                h_a[:batch_size],
                                                h_s[:batch_size],
                                                dna[:batch_size],
                                                dns[:batch_size],
                                                dis_a[:batch_size],
                                                dis_s[:batch_size])
        


        self.attribute_score_[node_idx[:batch_size]] = oa.detach().cpu()
        self.structural_score_[node_idx[:batch_size]] = os.detach().cpu()
        self.combined_score_[node_idx[:batch_size]] = oc.detach().cpu()

        return loss, ((oa + os + oc) / 3).detach().cpu()
    
    def load_model(self, path, data):
        ''''
        Load the model from the given path.

        :param path: str
            The path to the model. 
        :param data: torch_geometric.data.Data
            The graph data.
        '''
        self.num_nodes, self.in_dim = data.x.shape
        self.model = self.init_model(**self.kwargs)
        self.model.done.attr_encoder.load_state_dict(torch.load(path+'/attr_encoder.pt'))
        self.model.done.attr_decoder.load_state_dict(torch.load(path+'/attr_decoder.pt'))
        self.model.done.struct_encoder.load_state_dict(torch.load(path+'/struct_encoder.pt'))
        self.model.done.struct_decoder.load_state_dict(torch.load(path+'/struct_decoder.pt'))
        self.model.to(self.device)
        if self.batch_size == 0:
            self.batch_size = 16


    def save_model(self, path):
        '''
        Save the model to the given path.
        
        :param path: str
            The path to save the model.
        '''
        os.mkdir(path)
        
        torch.save(self.model.done.attr_encoder.state_dict(), path+'/attr_encoder.pt')
        torch.save(self.model.done.attr_decoder.state_dict(), path+'/attr_decoder.pt')
        torch.save(self.model.done.struct_encoder.state_dict(), path+'/struct_encoder.pt')
        torch.save(self.model.done.struct_decoder.state_dict(), path+'/struct_decoder.pt')
