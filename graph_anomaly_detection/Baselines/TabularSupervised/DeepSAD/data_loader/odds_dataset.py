from torch.utils.data import Dataset

import torch
import os


class ODDSDataset(Dataset):
    '''
    ODDSDataset class for datasets_cc from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/
    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.

    :param data: dict
        Dictionary with keys 'X_train', 'y_train', 'X_test', 'y_test' and values being the corresponding datasets.
    :param train: bool
        If True, return training set, else return test set.
   '''

    def __init__(self, data, train=True):
        super(Dataset, self).__init__()
        self.train = train

        if self.train:
            self.data = torch.tensor(data['X_train'], dtype=torch.float32)
            self.targets = torch.tensor(data['y_train'], dtype=torch.int64)
        else:
            self.data = torch.tensor(data['X_test'], dtype=torch.float32)
            self.targets = torch.tensor(data['y_test'], dtype=torch.int64)

        # self.semi_targets = torch.zeros_like(self.targets)
        self.semi_targets = self.targets

    def __getitem__(self, index):
        '''
        Returns a data sample and its target.
        '''
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)
