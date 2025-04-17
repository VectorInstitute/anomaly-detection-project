import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from Dataset.UCSD_dataset import load_test_data
from DMAD.model import *
from utils import *
from sklearn.metrics import roc_auc_score



class DMAD():
    '''
    The class for running the baselines for DMAD video anomaly detection.

    :param config: dict
        Configuration of the model.
    :param train_dataset: torch.data.utils.Dataset
        The training dataset.
    :param test_datasets: list of torch.data.utils.Dataset
        The testing datasets.
    :param test_ground_truth: list of np.array
        The ground truth of the testing datasets.
    :param seed: int
        seed for reproducible results
    '''

    def __init__(self, config, train_dataset, test_datasets, test_ground_truth, seed = 2021, tune=False):

        self.seed = seed
        self.config = config
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

        # Model setting
        self.model = convAE(self.config['background_path'],self.config['c'], 5, self.config['msize'], self.config['dim'])

        # Report the training process
        self.log_dir = os.path.join('./exp', self.config['dataset_type'], self.config['exp_dir'])
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.orig_stdout = sys.stdout
        f = open(os.path.join(self.log_dir, 'log.txt'),'w')
        if self.config['log_type'] == 'txt':
            sys.stdout = f
        
        if os.path.exists(self.config['results_dir']) == False: 
            os.mkdir(self.config['results_dir'])

        self.train_dataset = train_dataset
        self.test_datasets = test_datasets
        self.test_ground_truth = test_ground_truth


    def fit(self):
        ''' 
        Train the model.
        '''
        
        train_batch = data.DataLoader(self.train_dataset, batch_size = self.config['batch_size'], shuffle=True,
                                    num_workers=self.config['num_workers'], drop_last=True, pin_memory=self.config['pin_memory'])


        optimizer = torch.optim.AdamW([{'params': self.model.encoder.parameters()},
            {'params': self.model.decoder.parameters()},
            {'params': self.model.offset_net.parameters()},
            {'params': self.model.vq_layer.parameters()},
            {'params': self.model.bkg, "lr": 50*self.config['lr']},]
            ,lr=self.config['lr'])
        
            # {'params': self.bkg, "lr": 50*self.config['lr']}
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        self.model .cuda()


        # Training
        early_stop = {'idx' : 0,
                    'best_eval_auc' : 0}
        log_interval = 100
        loss_dict = self.model.latest_losses()
        losses = {k + '_train': 0 for k, v in loss_dict.items()}
        epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            self.model.train()
            for j,(imgs, _) in enumerate(train_batch):
                imgs = Variable(imgs).cuda()
                outputs = self.model.forward(imgs[:,0:12])

                optimizer.zero_grad()
                loss = self.model.loss_function(imgs[:,-3:], *outputs)
                loss.backward()
                optimizer.step()
                ########################################
                latest_losses = self.model.latest_losses()
                for key in latest_losses:
                    losses[key + '_train'] += float(latest_losses[key])
                    epoch_losses[key + '_train'] += float(latest_losses[key])

                if j % log_interval == 0:
                    for key in latest_losses:
                        losses[key + '_train'] /= log_interval
                    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
                    print('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                                ' {time:3.2f}   {loss}'
                                .format(epoch=epoch, batch=j * len(imgs),
                                        total_batch=len(train_batch) * len(imgs),
                                        percent=int(100. * j / len(train_batch)),
                                        time=time.time() - start_time,
                                        loss=loss_string))
                    start_time = time.time()
                    for key in latest_losses:
                        losses[key + '_train'] = 0

            scheduler.step()
            if epoch>4:optimizer.param_groups[-1]['lr'] = self.config['lr']*20
            print('----------------------------------------')
            print('Epoch:', epoch+1, '; Time:', time.time()-start_time)
            print('----------------------------------------')

            time_start = time.time()
            
            # Evaluate the model
            score = self.predict_score()

            # Compute the AUC
            aucroc = roc_auc_score(y_true=1 - self.test_ground_truth, y_score=score)

            if aucroc > early_stop['best_eval_auc']:
                early_stop['best_eval_auc'] = aucroc
                early_stop['idx'] = 0
                self.save_model(os.path.join(self.log_dir, 'model.pth'))
            else:
                early_stop['idx'] += 1
                print('AUCROC drop! Model not saved')

            print('With {} epochs, auc score is: {}, best score is: {}, used time: {}'.format(epoch+1, aucroc, early_stop['best_eval_auc'], time.time()-time_start))
            print('--------------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------------')


        print('Training is finished')

        sys.stdout = self.orig_stdout

        return self
    

    def predict_score(self, inference = False, inference_dataset = None):
        '''
        Predict the anomaly score for the testing data.

        :param inference: bool  
            Whether predict the score for inference over inference dataset.
        :param inference_dataset: torch.data.utils.Dataset
            The testing dataset for inference.
        '''

        self.model .cuda()

        
        print('Test of', self.config['dataset_type'])

        if inference:
            test_datasets = inference_dataset
        else:
            test_datasets = self.test_datasets

        list1 = {}
        list2 = {}
        list3 = {}
        comb = {}

        score = []
        reconstructed_images_level_1 = []
        reconstructed_images_level_2 = []
        reconstructed_images_level_3 = []

        self.model.eval()
        with torch.no_grad():
            for test_dataset in test_datasets:
                video_name = test_dataset.dir.split('/')[-1]
                list1[video_name] = []
                list2[video_name] = []
                list3[video_name] = []
                comb[video_name] = []

                # Loading dataset
                test_batch = data.DataLoader(test_dataset, batch_size = self.config['test_batch_size'],
                                            shuffle=False, num_workers=self.config['num_workers_test'], drop_last=False)

                for k,(imgs, _) in enumerate(test_batch):

                    imgs = Variable(imgs).cuda()

                    outputs = self.model.forward(imgs[:, 0:12],True)

                    if inference:
                        reconstructed_images_level_1.append(denormalize(outputs[2].cpu().numpy()[0], 1.0, 127.5))
                        reconstructed_images_level_2.append(denormalize(outputs[1].cpu().numpy()[0], 1.0, 127.5))
                        reconstructed_images_level_3.append(denormalize(outputs[0].cpu().numpy()[0], 1.0, 127.5))

                    self.model.loss_function(imgs[:, -3:], *outputs, True)
                    latest_losses = self.model.latest_losses()

                    list1[video_name].append(float(latest_losses['err1']))
                    list2[video_name].append(float(latest_losses['mse2']))
                    list3[video_name].append(float(latest_losses['grad']))


                list1[video_name] = anomaly_score_list_inv(list1[video_name])
                list2[video_name] = anomaly_score_list_inv(list2[video_name])
                list3[video_name] = anomaly_score_list_inv(list3[video_name])

            keys = list1.keys()
            merged_list1 = []
            merged_list2 = []
            merged_list3 = []
            for video_name in keys:
                merged_list1.extend(list1[video_name])
                merged_list2.extend(list2[video_name])
                merged_list3.extend(list3[video_name])
                
            merged_list1 = conf_avg(np.array(merged_list1))
            merged_list2 = conf_avg(np.array(merged_list2))
            merged_list3 = conf_avg(np.array(merged_list3))
            hyp_alpha = [0.2, 0.4, 0.6]
            score = np.array(merged_list1) * hyp_alpha[0] + np.array(merged_list2) * hyp_alpha[1] + np.array(merged_list3) * hyp_alpha[2]

            comb = {}
            start_idx = 0
            for video_name in keys:
                length = len(list1[video_name])
                list1[video_name] = merged_list1[start_idx:start_idx+length]
                list2[video_name] = merged_list2[start_idx:start_idx+length]
                list3[video_name] = merged_list3[start_idx:start_idx+length]
                comb[video_name] = score[start_idx:start_idx+length]
                np.savez(self.config['results_dir']+'/res_list_video_'+str(video_name)+'.npz', list1[video_name], list2[video_name], list3[video_name], comb[video_name])
                start_idx += length

        if inference:
            return score, [reconstructed_images_level_1,reconstructed_images_level_2,reconstructed_images_level_3]
        else:
            return score
    
    def inference(self, test_folder, data_config ,inference_video_name=None):
        '''
        Inference the anomaly score for the testing data.

        :param test_folder: str
            The path to the testing folder.
        :param data_config: dict
            The configuration of the data.
        :param inference_video_name: str
            The name of the video for inference.
        '''

        inference_dataset, ground_truth_labels = load_test_data(test_folder = test_folder,data_config=data_config,inference_video_name=inference_video_name)
        score, reconstructed = self.predict_score(inference = True, inference_dataset = inference_dataset)
                                          
        return visualize(test_folder+'/'+inference_video_name, ground_truth_labels, score, reconstructed)

    
    
    def save_model(self, path):
        '''
        save the model

        :param path: str
            path to save the model
        '''

        torch.save(self.model, os.path.join(path))

    def load_model(self, path):
        '''
        load the model

        :param path: str
            path to load the model
        '''

        try:
            self.model.load_state_dict(torch.load(path).state_dict(),strict=False)
        except:
            self.model.load_state_dict(torch.load(path),strict=False)
        self.model.cuda()
        