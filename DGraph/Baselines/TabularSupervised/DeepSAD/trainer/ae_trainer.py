from sklearn.metrics import roc_auc_score, average_precision_score


import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class AETrainer(ABC):

    '''
    Abstract base class for autoencoder trainer.

    :param optimizer_name: str
        Name of the optimizer for training the autoencoder.
    :param lr: float
        Initial learning rate during pretraining.
    :param n_epochs: int
        Number of epochs for training.
    :param lr_milestones: tuple
        Epochs at which to reduce the learning rate.
    :param batch_size: int
        Batch size for mini-batch training.
    :param weight_decay: float
        Weight decay (L2 penalty) for training.
    :param device: str
        Device on which to perform training.
    :param n_jobs_dataloader: int
        Number of workers for PyTorch's data loaders.
    '''

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

       
    def train(self, dataset , ae_net):

        '''
        Trains the autoencoder on the training set.

        :param dataset: torch.utils.data.Dataset
            Training dataset.
        :param ae_net: torch.nn.Module
            Autoencoder network.
        '''
        

        # Get train data loader
        train_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        ae_net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                n_batches += 1


        return ae_net

    def test(self, dataset, ae_net):

        '''
        Tests the autoencoder on the test set.
        
        :param dataset: torch.utils.data.Dataset
            Training dataset.
        :param ae_net: torch.nn.Module
            Autoencoder network.
        '''

        # Get test data loader
        test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1
