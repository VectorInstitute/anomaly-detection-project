from torch.utils.data import DataLoader
from .odds_dataset import ODDSDataset

from abc import ABC


class Dataset_Wrapper(ABC):

    '''
    Abstract base class loading dataloaders.

    :param data: torch.utils.data.Dataset
        Dataset to load.
    :param train: bool
        If True, load training set, else load test set.

    '''

    def __init__(self, data, train):
        super().__init__()


        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # training or testing dataset
        self.train = train

        if self.train:
            # Get training set
            self.train_set = ODDSDataset(data=data, train=True)
        else:
            # Get testing set
            self.test_set = ODDSDataset(data=data, train=False)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        '''
        Returns torch.utils.data.DataLoader for training and test data.

        :param batch_size: int
            Batch size.
        :param shuffle_train: bool
            If True, training-set dataloader will have shuffle=True.    
        :param shuffle_test: bool
            If True, test-set dataloader will have shuffle=True.
        :param num_workers: int
            Number of workers for data loading. 
        '''

        if self.train:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers, drop_last=True)
            return train_loader
        else:
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers, drop_last=False)
            return test_loader






def load_dataset(data, train=True):
    '''Loads the dataset.'''

    # for tabular data
    dataset = Dataset_Wrapper(data=data, train=train)

    return dataset