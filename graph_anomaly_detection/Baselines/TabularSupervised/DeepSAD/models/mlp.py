import torch.nn as nn
import torch.nn.functional as F

from .base_net import BaseNet


class MLP(BaseNet):
    '''
    Multilayer perceptron consisting of fully-connected layers with leaky ReLU activations.

    :param x_dim: int
        Dimensionality of the input.
    :param h_dims: list
        List of hidden dimensions.
    :param rep_dim: int
        Dimensionality of the representation.
    :param bias: bool
        Whether to use bias in the fully-connected layers.
    '''

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        '''
        Returns the representation of the input.

        :param x: torch.Tensor
            Input tensor.
        '''
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class MLP_Decoder(BaseNet):
    '''
    Decoder network for the MLP.

    :param x_dim: int
        Dimensionality of the input.
    :param h_dims: list
        List of hidden dimensions.
    :param rep_dim: int
        Dimensionality of the representation.
    :param bias: bool
        Whether to use bias in the fully-connected layers.
    '''

    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        '''
        Returns the reconstruction of the input.

        :param x: torch.Tensor 
            Input tensor.
        '''
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


class MLP_Autoencoder(BaseNet):
    '''
    Autoencoder consisting of an MLP encoder and decoder.

    :param x_dim: int
        Dimensionality of the input.
    :param h_dims: list    
        List of hidden dimensions.
    :param rep_dim: int
        Dimensionality of the representation.
    :param bias: bool
        Whether to use bias in the fully-connected layers.
    '''

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x):
        '''
        Returns the reconstruction of the input.
        
        :param x: torch.Tensor
            Input tensor.
        '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Linear_BN_leakyReLU(nn.Module):
    '''
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation

    :param in_features: int
        Size of each input sample
    :param out_features: int
        Size of each output sample
    :param bias: bool   
        If set to False, the layer will not learn an additive bias. Default: True
    :param eps: float
        A value added to the denominator for numerical stability. Default: 1e-05    
    '''


    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
