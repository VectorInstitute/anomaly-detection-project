from myutils import Utils
import numpy as np

#add the baselines from the pyod package
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from joblib import dump, load


class PYOD():
    def __init__(self, seed, model_name, tune=False):
        '''
        The class to run PYOD models

        :param seed: int
            seed for reproducible results
        :param model_name: str
            model name
        :param tune: bool, optional
            if necessary, tune the hyper-parameter based on the validation set constructed by the labeled anomalies
        '''

        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {'IForest':IForest,'CBLOF':CBLOF}

        self.tune = tune

    def grid_hp(self, model_name):
        
        '''
        define the hyper-parameter search grid for different unsupervised mdoel
        '''

        param_grid_dict = {'IForest': [10, 50, 100, 500], # n_estimators, default=100
                           'CBLOF': [4, 6, 8, 10], # n_clusters, default=8
                           }

        return param_grid_dict[model_name]

    def grid_search(self, X_train, y_train, ratio=None):
        '''
        implement the grid search for unsupervised models and return the best hyper-parameters
        the ratio could be the ground truth anomaly ratio of input dataset

        :param X_train: torch.Tensor
            the training data
        :param y_train: torch.Tensor
            the training label
        :param ratio: float, optional
            the ground truth anomaly ratio of input dataset
        '''

        # set seed
        self.utils.set_seed(self.seed)
        # get the hyper-parameter grid
        param_grid = self.grid_hp(self.model_name)

        if param_grid is not None:
            # index of normal ana abnormal samples
            idx_a = np.where(y_train==1)[0]
            idx_n = np.where(y_train==0)[0]
            idx_n = np.random.choice(idx_n, int((len(idx_a) * (1-ratio)) / ratio), replace=True)

            idx = np.append(idx_n, idx_a) #combine
            np.random.shuffle(idx) #shuffle

            # valiation set (and the same anomaly ratio as in the original dataset)
            X_val = X_train[idx]
            y_val = y_train[idx]

            # fitting
            metric_list = []
            for param in param_grid:
                try:
                    if self.model_name == 'IForest':
                        model = self.model_dict[self.model_name](n_estimators=param).fit(X_train)

                    elif self.model_name == 'CBLOF':
                        model = self.model_dict[self.model_name](n_clusters=param).fit(X_train)
                    else:
                        raise NotImplementedError

                except:
                    metric_list.append(0.0)
                    continue

                try:
                    # model performance on the validation set
                    score_val = model.decision_function(X_val)
                    metric = self.utils.metric(y_true=y_val, y_score=score_val, pos_label=1)
                    metric_list.append(metric['aucpr'])

                except:
                    metric_list.append(0.0)
                    continue

            best_param = param_grid[np.argmax(metric_list)]

        else:
            metric_list = None
            best_param = None

        print(f'The candidate hyper-parameter of {self.model_name}: {param_grid},',
              f' corresponding metric: {metric_list}',
              f' the best candidate: {best_param}')

        return best_param
    
    def save_model(self, path):
        '''
        save the model

        :param path: str
            the path to save the model
        '''
        dump(self.model, path+'.joblib')

    def load_model(self, path,data=None):
        '''
        load the model

        :param path: str
            the path to load the model
        '''
        self.model = load(path+'.joblib')

    def fit(self, X_train, y_train, ratio=None):
        '''
        fit the model

        :param X_train: torch.Tensor
            the training data
        :param y_train: torch.Tensor
            the training label
        '''
        # selecting the best hyper-parameters of unsupervised model for fair comparison (if labeled anomalies is available)
        if sum(y_train) > 0 and self.tune:
            assert ratio is not None
            best_param = self.grid_search(X_train, y_train, ratio)
        else:
            best_param = None

        print(f'best param: {best_param}')

        # set seed
        self.utils.set_seed(self.seed)

        # fit best on the best param
        if best_param is not None:
            if self.model_name == 'IForest':
                self.model = self.model_dict[self.model_name](n_estimators=best_param).fit(X_train)

            elif self.model_name == 'CBLOF':
                self.model = self.model_dict[self.model_name](n_clusters=best_param).fit(X_train)
                
            else:
                raise NotImplementedError

        else:
            # unsupervised method would ignore the y labels
            self.model = self.model_dict[self.model_name]().fit(X_train, y_train)

        return self

    # from pyod: for consistency, outliers are assigned with larger anomaly scores
    def predict_score(self, X):
        '''
        predict the anomaly score

        :param X: torch.Tensor
            the input data
        '''
        score = self.model.decision_function(X)
        return score