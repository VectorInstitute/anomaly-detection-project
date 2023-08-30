from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from abc import ABC, abstractmethod

import torch
import torch.optim as optim
import numpy as np


class DeepSADTrainer(ABC):
    '''
    Abstract base class for Deep SAD trainer.

    :param c: float
        Hypersphere center c.
    :param eta: float
        Hypersphere radius eta.
    :param optimizer_name: str
        Name of the optimizer for training the Deep SAD network.
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

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader


        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

    def train(self, dataset , net):
        '''
        Trains the Deep SAD network on the training set.

        :param dataset: torch.utils.data.Dataset
            Training dataset.
        :param ae_net: torch.nn.Module
            DeepSAD network.
        '''
        

        # Get train data loader
        train_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            self.c = self.init_center_c(train_loader, net)

        # Training
        
        net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # transfer the label "1" to "-1" for the inverse loss
                semi_targets[semi_targets==1] = -1

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                scheduler.step()
               
                epoch_loss += loss.item()
                n_batches += 1


        return net

    def test(self, dataset , net):

        '''
        Tests the Deep SAD network on the testing set.

        :param dataset: torch.utils.data.Dataset
            Training dataset.
        :param ae_net: torch.nn.Module
            DeepSAD network.
        '''

        # Get test data loader
        test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        scores = np.array(scores)
        # self.test_aucroc = roc_auc_score(labels, scores)
        # self.test_aucpr = average_precision_score(labels, scores, pos_label = 1)

        return scores

    def init_center_c(self, train_loader: DataLoader, net, eps=0.1):
        '''
        Initialize hypersphere center c as the mean from an initial forward pass on the data.

        :param train_loader: torch.utils.data.DataLoader
            Training data loader.
        :param ae_net: torch.nn.Module
            Autoencoder network.
        :param eps: float
            Small value to avoid division by zero when no training samples are given.
        '''
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
