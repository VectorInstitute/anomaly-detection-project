import os
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
import argparse

from Dataset.prepare_data import DataGenerator
from myutils import Utils

class RunPipeline():
    def __init__(self, dir:str,
                 suffix:str,
                 model_type:str,
                supervision:str, 
                seed_list:list, 
                rla_list:list):
        '''
        The class to run the pipeline
        
        :param dir: the directory of saved results
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param model_type: Tabular or Graph
        :param supervision: Unsupervised or Supervised
        :param seed_list: seed list for random number generator
        :param rla_list: ratio of labeled anomalies or number of labeled anomalies
        '''

        # utils function
        self.utils = Utils()

        if dir is None:
            self.dir = os.getcwd() 
        else:
            self.dir = dir

        # the suffix of all saved files 
        self.suffix = suffix 

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if not os.path.exists(os.path.join(self.dir, 'checkpoints',self.suffix)):
            os.makedirs(os.path.join(self.dir, 'checkpoints',self.suffix))
        if not os.path.exists(os.path.join(self.dir, 'results',self.suffix)):
            os.makedirs(os.path.join(self.dir, 'results',self.suffix))
        
        # seed list
        self.seed_list = seed_list
        
        self.model_type = model_type
        self.supervision = supervision

        # data generator instantiation
        self.data_generator = DataGenerator()

        # ratio of labeled anomalies
        self.rla_list = rla_list

    

        # model name
        self.model_dict = {}

        if self.model_type == 'Tabular':
            # unsupervised algorithms
            if self.supervision == 'Unsupervised':
                from Baselines.TabularUnsupervised.run import PYOD

                # from pyod
                for _ in ['IForest', 'CBLOF', 'ECOD']:
                    self.model_dict[_] = PYOD


            # supervised algorithms
            elif self.supervision == 'Supervised':
                from Baselines.TabularSupervised.DeepSAD.run import DeepSAD
                from Baselines.TabularSupervised.XGBoost.run import XGBoost
                from Baselines.TabularSupervised.DeepCNN.run import FTTransformer

                self.model_dict['DeepSAD'] = DeepSAD

                # from sklearn
                self.model_dict['XGB'] = XGBoost

                # ResNet and FTTransformer for tabular data
                for _ in ['FTTransformer','MLP']:
                    self.model_dict[_] = FTTransformer

        elif self.model_type == 'Graph':
                # unsupervised algorithms
            if self.supervision == 'Unsupervised':
                from Baselines.GraphUnsupervised.run import PYGOD

                # from pygod
                for _ in ['OCGNN', 'DONE','AdONE']:
                    self.model_dict[_] = PYGOD

            elif self.supervision == 'Supervised':
                from Baselines.GraphSupervised.GNN.run import GNN
                from Baselines.GraphSupervised.AMNet.run import AMNet

                for _ in ['GCN', 'SAGE']:
                    self.model_dict[_] = GNN
                self.model_dict['AMNet'] = AMNet
        else:
            raise NotImplementedError
        
        self.save_checkpoint = True



    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the checkpoint_path should be specified
            print('self.model_name',self.model_name)
        
            if self.model_name == 'FTTransformer':
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, batch_size=512, n_epochs=200)
            elif self.model_name == 'MLP':
                print(len(self.data.train_mask))
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, batch_size=len(self.data.train_mask), n_epochs=200)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # fitting
            start_time = time.time()
            if self.model_type == 'Graph':
                self.clf = self.clf.fit(data=self.data, indexes=self.data.train_mask)
            else:
                self.clf = self.clf.fit(X_train=self.data.x[self.data.train_mask], y_train=self.data.y[self.data.train_mask])
            end_time = time.time(); time_fit = end_time - start_time

            # if self.save_checkpoint and self.model_name not in ['DONE','AdONE']:
            self.clf.save_model(self.checkpoint_path)


            # predicting score (inference)
            start_time = time.time()
            if self.model_type == 'Graph':
                score_test = self.clf.predict_score(self.data, self.data.test_mask)
            else:
                score_test = self.clf.predict_score(self.data.x[self.data.test_mask])
            end_time = time.time(); time_inference = end_time - start_time

            # performance
            if self.model_type == 'Graph':
                result = self.utils.metric(y_true=self.data.y[self.data.test_mask].cpu(), y_score=score_test)
            else:
                result = self.utils.metric(y_true= self.data.y[self.data.test_mask], y_score=score_test)

            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

    # run the experiment
    def run(self,model_name:str=None):
        #  filteting dataset that does not meet the experimental requirements
        self.data_generator.dataset =  'DgraphFin'
       

        print(f'DgraphFin datasets, {len(self.model_dict.keys())} models')
        print(self.model_dict.keys())

        # save the results
        df_AUCROC = pd.DataFrame(data=None, index=self.rla_list, columns=list(self.model_dict.keys()))
        df_AUCPR = pd.DataFrame(data=None, index=self.rla_list, columns=list(self.model_dict.keys()))
        df_time_fit = pd.DataFrame(data=None, index=self.rla_list, columns=list(self.model_dict.keys()))
        df_time_inference = pd.DataFrame(data=None, index=self.rla_list, columns=list(self.model_dict.keys()))

        # model name 
        if model_name is not None:
            self.model_dict = {model_name: self.model_dict[model_name]}

        for seed in self.seed_list:
            self.seed = seed
            self.data_generator.seed = self.seed
            self.path_base = self.model_type+'_'+self.supervision+'_seed_'+str(self.seed)

            path_AUCROC = os.path.join(self.dir, 'results',self.suffix, 'AUCROC_' +self.path_base+ '.csv')
            if not os.path.exists(path_AUCROC):
                df_AUCROC.to_csv(path_AUCROC)
            path_AUCPR = os.path.join(self.dir, 'results',self.suffix, 'AUCPR_' + self.path_base+ '.csv')
            if not os.path.exists(path_AUCPR):
                df_AUCPR.to_csv(path_AUCPR)
            path_time_fit = os.path.join(self.dir, 'results',self.suffix, 'time_fit_' + self.path_base+ '.csv')
            if not os.path.exists(path_time_fit):
                df_time_fit.to_csv(path_time_fit)
            path_time_inference = os.path.join(self.dir, 'results',self.suffix, 'time_inference_' + self.path_base + '.csv')
            if not os.path.exists(path_time_inference):
                df_time_inference.to_csv(path_time_inference)

            for la in tqdm(rla_list):

                if self.supervision == 'Unsupervised' and la != 0.0:
                    continue
                if self.supervision != 'Unsupervised' and la == 0.0:
                    continue

                print(f'Current experiment la: {la}')

                # generate data

                try:
                    self.data = self.data_generator.graph_generator(rla=la, at_least_one_labeled=True)
                except Exception as error:
                    print(f'Error when generating data: {error}')
                    pass
                    continue

                for model_name in tqdm(self.model_dict.keys()):
                    self.model_name = model_name
                    self.clf = self.model_dict[self.model_name]
                    self.checkpoint_path = os.path.join(self.dir, 'checkpoints', self.suffix, self.path_base+'_la_' + str(la)+'_model_'+self.model_name)


                    # fit model
                    time_fit, time_inference, result = self.model_fit()

                    # load the previous results
                    df_AUCROC = pd.read_csv(path_AUCROC, index_col=0)
                    df_AUCPR = pd.read_csv(path_AUCPR, index_col=0)
                    df_time_fit = pd.read_csv(path_time_fit, index_col=0)
                    df_time_inference = pd.read_csv(path_time_inference, index_col=0)


                    # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                    df_AUCROC[model_name].loc[la] = result['aucroc']
                    df_AUCPR[model_name].loc[la] = result['aucpr']
                    df_time_fit[model_name].loc[la] = time_fit
                    df_time_inference[model_name].loc[la] = time_inference

                    df_AUCROC.to_csv(path_AUCROC, index=True)
                    df_AUCPR.to_csv(path_AUCPR, index=True)
                    df_time_fit.to_csv(path_time_fit, index=True)
                    df_time_inference.to_csv(path_time_inference, index=True)
                    

# run the above pipeline for reproducing the results in the paper
parser = argparse.ArgumentParser(description='AD_BENCH')
parser.add_argument('--dir', type=str, default='/ssd003/projects/aieng/public/anomaly_detection_models/DGraphFin',help='Directory of result file')
parser.add_argument('--suffix', type=str, default='Test_results',help='Suffix of result file')
parser.add_argument('--model_type', type=str, default='Tabular',help='Type of model')
parser.add_argument('--supervision', type=str, default='Unsupervised',help='Supervision of model')
parser.add_argument('--model_name', type=str, default=None,help='Name of model')
parser.add_argument('--seed_list', type=str, default='20',help='Seed list')
parser.add_argument('--rla_list', type=str, default='0,0.01,0.05,0.1,0.25,0.5,0.75,1',help='rla list')
args = parser.parse_args()
seed_list = [int(item) for item in args.seed_list.split(',')]
rla_list = [float(item) for item in args.rla_list.split(',')]

print('rla_list',rla_list)

pipeline = RunPipeline(dir = args.dir ,suffix=args.suffix,model_type=args.model_type,
                    supervision=args.supervision, 
                    seed_list=seed_list, 
                    rla_list=rla_list)

pipeline.run(model_name=args.model_name)