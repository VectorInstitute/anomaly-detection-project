
from .mlp import MLP, MLP_Autoencoder


def build_network(net_name, input_size ,ae_net=None):
    '''
    Builds the neural network.
    
    :param net_name: str
        Name of the network architecture.
    :param input_size: int
        Dimensionality of the input.
    :param ae_net: torch.nn.Module, optional
        Autoencoder network.
    '''
    net = None
    net = MLP(x_dim=input_size, h_dims=[100, 20], rep_dim=10, bias=False)

    return net

def build_autoencoder(net_name, input_size):
    '''
    Builds the corresponding autoencoder network.
    
    :param net_name: str
        Name of the network architecture.
    :param input_size: int
        Dimensionality of the input. 
    '''
    ae_net = None

    ae_net = MLP_Autoencoder(x_dim=input_size, h_dims=[100, 20], rep_dim=10, bias=False)

    return ae_net
