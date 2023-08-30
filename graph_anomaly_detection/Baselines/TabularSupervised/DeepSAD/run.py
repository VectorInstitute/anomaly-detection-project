import torch
import numpy as np

from .models.main import build_network, build_autoencoder
from .trainer.deepsad_trainer import DeepSADTrainer
from .trainer.ae_trainer import AETrainer
from .data_loader.dataset_wrapper import load_dataset

from myutils import Utils


class DeepSAD():
    '''
    A class for running the DeepSAD method. 

        :param seed: int
            Random seed.
        :param model_name: str
            Name of the method to use.
        :param eta: float
            Deep SAD hyperparameter eta (must be 0 < eta).
    '''

    def __init__(self, seed, model_name='DeepSAD',input_size=17,eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.utils = Utils()
        self.device = self.utils.get_device(True)  # get 
        print(self.device)
        print('\n')

        self.seed = seed

        
        self.xp_path = None
        self.load_config = None
        self.load_model_add = None
        

        self.eta = eta # eep SAD hyperparameter eta (must be 0 < eta).
        self.c = None  # hypersphere center c
        
        self.num_threads = 0
        self.n_jobs_dataloader = 0
        self.input_size = input_size

        self.net_name = 'dense' #  A string indicating the name of the neural network to use.
        self.trainer = None # DeepSADTrainer to train a Deep SAD model.
        self.optimizer_name = 'adam' # A string indicating the optimizer to use for training the Deep SAD network.
        self.lr = 0.001
        self.n_epochs = 50
        self.lr_milestone = [0]
        self.batch_size = 128
        self.weight_decay = 1e-6
        self.pretrain = True # whether to use auto-encoder for pretraining

        self.ae_net = None  # The autoencoder network corresponding to phi for network weights pretraining.
        self.ae_trainer = None # AETrainer to train an autoencoder in pretraining.
        self.ae_optimizer_name = 'adam' #  A string indicating the optimizer to use for pretraining the autoencoder.
        self.ae_lr = 0.001
        self.ae_n_epochs = 100
        self.ae_lr_milestone = [0]
        self.ae_batch_size = 128
        self.ae_weight_decay = 1e-6

        # Initialize DeepSAD model and set neural network phi
        self.set_network(self.net_name, self.input_size)


    def set_network(self, net_name, input_size):
        '''
        Builds the neural network phi.
        
        :param net_name: str
            A string indicating the name of the neural network to use.
        :param input_size: int
            The input size of the neural network.
        '''

        self.net_name = net_name
        self.input_size = input_size
        self.net = build_network(net_name, input_size) #The neural network phi.

    def fit(self, X_train, y_train):
        '''
        Trains the Deep SAD model on the training data.
        
        :param X_train: torch.Tensor
            The training instances.
        :param y_train: torch.Tensor
            The training labels.
        '''

        # Set seed (using myutils)
        self.utils.set_seed(self.seed)
        # Set the number of threads used for parallelizing CPU operations
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        
        # Load data
        data = {'X_train': X_train, 'y_train': y_train}
        dataset = load_dataset(data=data, train=True)
        input_size = dataset.train_set.data.size(1) #input size


        # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
        if self.load_model_add:
            self.load_model(model_path=self.load_model_add, load_ae=True, map_location=self.device)
            
        if self.pretrain:
            # Pretrain model on dataset (via autoencoder)
            self.pretrain(dataset,
                             input_size,
                             optimizer_name=self.ae_optimizer_name,
                             lr=self.ae_lr,
                             n_epochs=self.ae_n_epochs,
                             lr_milestones=self.ae_lr_milestone,
                             batch_size=self.ae_batch_size,
                             weight_decay=self.ae_weight_decay,
                             device=self.device,
                             n_jobs_dataloader=self.n_jobs_dataloader)

        # Train model on dataset
        self.train(dataset,
                   optimizer_name=self.optimizer_name,
                   lr=self.lr,
                   n_epochs=self.n_epochs,
                   lr_milestones=self.lr_milestone,
                   batch_size=self.batch_size,
                          weight_decay=self.weight_decay,
                          device=self.device,
                          n_jobs_dataloader=self.n_jobs_dataloader)

        return self
    
    def predict_score(self, X):
        '''
        Predicts a score for the samples X.

        :param X: torch.Tensor
            The samples to predict the score for.
        '''

        # input randomly generated y label for consistence
        dataset = load_dataset(data={'X_test': X, 'y_test': np.random.choice([0, 1], X.shape[0])}, train=False)
        score = self.test(dataset, device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)

        return score

    def train(self, dataset , optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        
        '''
        Trains the Deep SAD model on the training data.

        :param dataset: torch.utils.data.Dataset
            The training data.
        :param optimizer_name: str
            A string indicating the optimizer to use for training the Deep SAD network.
        :param lr: float
            The learning rate for training the Deep SAD network.
        :param n_epochs: int
            The number of epochs to train the Deep SAD network.
        :param lr_milestones: tuple
            The epochs at which to reduce the learning rate.
        :param batch_size: int
            The batch size to use for training.
        :param weight_decay: float
            The weight decay (L2 penalty) to use for training the Deep SAD network (Adam optimizer only).
        :param device: str
            The device on which to run the training.
        :param n_jobs_dataloader: int
            The number of workers for the PyTorch dataloader.
        '''

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset , device: str = 'cuda', n_jobs_dataloader: int = 0):
        '''Tests the Deep SAD model on the test data.
        
        :param dataset: torch.utils.data.Dataset
            The test data.
        :param device: str
            The device on which to run the training.
        :param n_jobs_dataloader: int
            The number of workers for the PyTorch dataloader.

        '''

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        score = self.trainer.test(dataset, self.net)

        return score

    def pretrain(self, dataset , input_size ,optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        '''
        Pretrains the weights for the Deep SAD network phi via autoencoder.

        :param dataset: torch.utils.data.Dataset
            The training data.
        :param input_size: int
            The input size of the autoencoder.
        :param optimizer_name: str
            A string indicating the optimizer to use for training the autoencoder network.
        :param lr: float
            The learning rate for training the autoencoder network.
        :param n_epochs: int
            The number of epochs to train the autoencoder network.
        :param lr_milestones: tuple
            The epochs at which to reduce the learning rate.
        :param batch_size: int
            The batch size to use for training.
        :param weight_decay: float
            The weight decay (L2 penalty) to use for training the autoencoder network (Adam optimizer only).
        :param device: str
            The device on which to run the training.
        :param n_jobs_dataloader: int
            The number of workers for the PyTorch dataloader.
        '''

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name, input_size)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)


        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        '''
        Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder.
        '''

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, path, save_ae=True):
        '''
        Save Deep SAD model to export_model.
        
        :param path: str
            The export_model path. 
        :param save_ae: bool, optional
            Whether to save the Deep SAD model (parameters of the encoder and the center c) as well as the parameters
        '''

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, path + '.tar')

    def load_model(self, path,data):
        '''
        Load Deep SAD model from model_path.
        
        :param path: str
            The export_model path.
        :param data: torch.tensor
            The training or test data to determine the input size of the autoencoder.
        '''

        model_dict = torch.load(path + '.tar', map_location=self.device)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if self.ae_net is None:
            self.ae_net = build_autoencoder(self.net_name,self.input_size)
        self.ae_net.load_state_dict(model_dict['ae_net_dict'])
