import numpy as np
from math import ceil


from myutils import Utils
from . import DGraphFin


class DataGenerator():
    '''
    The class to generate the graph data

    :param seed: int
        seed for reproducible results
    :param dataset: str
        specific the dataset name
    '''
    def __init__(self, seed:int=42, dataset_name:str=None, root: str='/ssd003/projects/aieng/public/anomaly_detection_datasets'):
        

        self.seed = seed
        self.dataset = dataset_name
        self.root = root
        # myutils function
        self.utils = Utils()

    def graph_generator(self, X=None, y=None, minmax=True,
                  rla=None, at_least_one_labeled=False):
        '''
        Generate the graph data

        rla: list
            ratio of labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: bool
            whether to guarantee at least one labeled anomalies in the training set
        '''
        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is 'DgraphFin':
            data = DGraphFin(root=self.root, name='DGraphFin')[0]
            x = data.x
            y = data.y
            # spliting the current data to the training set and testing set
            split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}


        # show the statistic
        self.utils.data_description(data,rla)


        # minmax scaling
        if minmax:
            x = (x - x.mean(0)) / x.std(0)
            data.x = x


        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y[split_idx['train']] == 0)[0]
        idx_anomaly = np.where(y[split_idx['train']] == 1)[0]

        if type(rla) == float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(rla * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(rla * len(idx_anomaly)), replace=False)
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)

        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        d = data.y[split_idx['train']]
        d[idx_unlabeled] = 0
        data.y[split_idx['train']] = d
        d = data.y[split_idx['train']]
        d[idx_labeled_anomaly] = 1
        data.y[split_idx['train']] = d
        if data.y.dim() == 2:
            data.y = data.y.squeeze(1)

        return data