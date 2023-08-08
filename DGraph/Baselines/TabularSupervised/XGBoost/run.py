import xgboost as xgb
from myutils import Utils

class XGBoost():
    """
    The original code: https://xgboost.readthedocs.io/en/latest/python/python_api.html
    The class to run XGBoost model
    
    :param seed: int
        random seed
    :param model_name: str
        the name of the model
    """
    def __init__(self, seed:int, model_name:str=None):
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {'XGB':xgb.XGBClassifier}

        self.model = self.model_dict[self.model_name](random_state=self.seed)

    def fit(self, X_train, y_train, ratio=None):
        '''
        Fit the model with training data

        :param X_train: torch.Tensor
            the training data
        :param y_train: torch.Tensor
            the training label
        '''

        # fitting
        self.model.fit(X_train.numpy(), y_train.numpy())

        return self
    def save_model(self, path):
        '''
        Save the model

        ::param path: str
            the path of the model
        '''
        self.model.save_model(path)

    def load_model(self, path,data=None):
        '''
        load the model 

        :param path: str
            the path of the model
        :param data: torch.Tensor
            the training data
        '''
        self.model.load_model(path)

    def predict_score(self, X):
        '''
        predict the score of the data

        :param X_train: torch.Tensor
            the testing data
        '''
        score = self.model.predict_proba(X.numpy())[:, 1]
        return score