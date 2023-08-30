import numpy as np
import random
import torch

# metric
from sklearn.metrics import roc_auc_score, average_precision_score



class Utils():
    def __init__(self):
        pass


    def set_seed(self, seed):
        '''
        remove randomness by setting seed

        :param seed: random seed
        '''

        # basic seed
        np.random.seed(seed)
        random.seed(seed)

        # pytorch seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_device(self, gpu_specific=False):
        '''
        get device for training and testing

        :param gpu_specific: if True, use gpu if available
        '''

        if gpu_specific:
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                print(f'number of gpu: {n_gpu}')
                print(f'cuda name: {torch.cuda.get_device_name(0)}')
                print('GPU is on')
            else:
                print('GPU is off')

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    def data_description(self,data,rla):
        '''
        show the statistic of the dataset
        
        :param data: torch_geometric.data.Data
        :param rla: labeled anomalies ratio
        '''
        X = data.x
        y = data.y
        anomalies = (sum(y[data.train_mask])+sum(y[data.valid_mask])+sum(y[data.valid_mask])).item()
        len_total = len(y[data.train_mask])+ len(y[data.valid_mask]) + len(y[data.test_mask])
        des_dict = {}
        des_dict['Train Nodes'] = X[data.train_mask].shape[0]
        des_dict['Validation Nodes'] = X[data.valid_mask].shape[0]
        des_dict['Test Nodes'] = X[data.test_mask].shape[0]
        des_dict['Anomalies'] = str(anomalies)+'/'+str(len_total)
        des_dict['Background Nodes'] = X.shape[0] - (des_dict['Train Nodes']+des_dict['Validation Nodes']+des_dict['Test Nodes'])
        des_dict['Features'] = X.shape[1]
        des_dict['Labeled Anomalies Ratio(%)'] = rla

        print(des_dict)
        print('\n')

    # metric
    def metric(self, y_true, y_score, pos_label=1):
        '''
        :param y_true: true label
        :param y_score: predicted score
        '''
        aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
        aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

        return {'aucroc':aucroc, 'aucpr':aucpr}