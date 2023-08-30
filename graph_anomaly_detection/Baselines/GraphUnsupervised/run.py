from myutils import Utils


from .trainer import DONE
from .trainer import AdONE
from .trainer import OCGNN

class PYGOD():
    '''
    The class for running the baselines for unsupervised graph methods.

    :param seed: int
        seed for reproducible results
    :param model_name: str
        model name
    :param tune: bool
        if necessary, tune the hyper-parameter based on the validation set constructed by the labeled anomalies
    '''
    def __init__(self, seed, model_name, tune=False):

        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {'DONE':DONE,'AdONE':AdONE,  'OCGNN':OCGNN}

        self.tune = tune

        self.model = self.model_dict[self.model_name]()

    
    def save_model(self, path):
        '''
        save the model

        :param path: str
            path to save the model
        '''
        self.model.save_model(path)

    def load_model(self, path, data):
        '''
        load the model

        :param path: str
            path to load the model
        :param data: torch_geometric.data.Data
            The graph data.
        '''

        self.model.load_model(path,data)

    def fit(self, data, indexes):
        '''
        fit the model

        :param data: torch_geometric.data.Data
            data object
        :param indexes: torch.Tensor
            index of the training set
        '''

        # set seed
        self.utils.set_seed(self.seed)

        # fit best on the best param
        self.model = self.model.fit(data,indexes)
        return self

    def predict_score(self, data,indexes):
        '''
        predict the score
        
        :param data: torch_geometric.data.Data
            data object
        :param indexes: torch.Tensor
            index of the training set
        '''
        score = self.model.decision_function(data,indexes)
        return score