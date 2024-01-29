import os
import numpy as np
from torch.utils.data import Dataset
import random

# Define a dataset class for loading normal data
class Normal_dataset(Dataset):
    def __init__(self, is_train=1, data_path='/ssd003/projects/aieng/public/anomaly_detection_datasets/UCF_crime/UCF-Crime-Features/', modality='TWO'):
        super().__init__()
        self.is_train = is_train
        self.data_path = data_path
        self.modality = modality
        # Determine the data source based on whether it's for training or testing
        if self.is_train == 1:
            self.data_src = os.path.join(self.data_path, 'train_normal.txt')
            self.datalist = open(self.data_src, 'r').readlines()
            self.datalist = [idx.strip() for idx in self.datalist] 
        else:
            self.data_src = os.path.join(self.data_path, 'test_normal.txt')
            self.datalist = open(self.data_src, 'r').readlines()
            random.shuffle(self.datalist)  # Shuffle the list for testing
            self.datalist = [idx.strip() for idx in self.datalist] 


    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.datalist)


    def __getitem__(self, idx):
        if self.is_train == 1:
            # Load RGB and flow features for training data
            rgb_features_path = os.path.join(self.data_path, 'all_rgbs', self.datalist[idx] + '.npy')
            flow_features_path = os.path.join(self.data_path, 'all_flows', self.datalist[idx] + '.npy')
            rgb_features = np.load(rgb_features_path)  
            flow_features = np.load(flow_features_path)  
            # Concatenate RGB and Flow features to have a more comprehensive representation of video contents
            concatinated_features = np.concatenate([rgb_features, flow_features], axis=1)  # 32,2048
            # Return features based on the selected modality for training
            if self.modality == 'RGB':
                return rgb_features
            elif self.modality == 'FLOW':
                return flow_features
            else:        
                return concatinated_features
        else:
            # Load data for testing, including name, frames, and ground truth
            name = self.datalist[idx].split(' ')[0]   # Extract information from the given text file
            frames = int(self.datalist[idx].split(' ')[1])  # Convert to int
            gts = int(self.datalist[idx].split(' ')[2])
            rgb_features_path = os.path.join(self.data_path, 'all_rgbs', name + '.npy')
            flow_features_path = os.path.join(self.data_path, 'all_flows', name + '.npy')
            rgb_features = np.load(rgb_features_path)  
            flow_features = np.load(flow_features_path)  
            # Concatenate RGB and Flow features for testing
            concatinated_features = np.concatenate([rgb_features, flow_features], axis=1)  # 32,2048
            # Return features along with ground truth and frame information based on modality
            if self.modality == 'RGB':
                return rgb_features, gts, frames
            elif self.modality == 'FLOW':
                return flow_features, gts, frames
            else:        
                return concatinated_features, gts, frames


# Define a dataset class for loading abnormal data
class Abnormal_dataset(Dataset):
    def __init__(self, is_train=1, data_path='/ssd003/projects/aieng/public/anomaly_detection_datasets/UCF_crime/UCF-Crime-Features/', modality='TWO'):
        self.is_train = is_train
        self.data_path = data_path
        self.modality = modality    
        # Determine the data source based on whether it's for training or testing
        if self.is_train == 1:
            self.data_src = os.path.join(self.data_path, 'train_anomaly.txt')
            self.datalist = open(self.data_src, 'r').readlines()
            self.datalist = [idx.strip() for idx in self.datalist]
        else:
            self.data_src = os.path.join(self.data_path, 'test_anomaly.txt')
            self.datalist = open(self.data_src, 'r').readlines()
            self.datalist = [idx.strip() for idx in self.datalist]
    

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.datalist)


    def __getitem__(self, idx):
        if self.is_train == 1:
            # Load RGB and flow features for training data
            rgb_features_path = os.path.join(self.data_path + 'all_rgbs', self.datalist[idx] + '.npy')
            flow_features_path = os.path.join(self.data_path + 'all_flows', self.datalist[idx] + '.npy')
            rgb_features = np.load(rgb_features_path)  
            flow_features = np.load(flow_features_path)
            # Concatenate RGB and Flow features for training
            concatinated_features = np.concatenate([rgb_features, flow_features], axis=1)  # 32,2048
            # Return features based on the selected modality for training
            if self.modality == 'RGB':
                return rgb_features
            elif self.modality == 'FLOW':
                return flow_features
            else:        
                return concatinated_features
        else:
            # Load data for testing, including name, frames, and ground truth
            name = self.datalist[idx].split('|')[0]   # Extract information from the given text file
            frames = int(self.datalist[idx].split('|')[1])  # Convert str to int
            gts = self.datalist[idx].split('|')[2][1:-1].split(',')  # Extract ground truth values as a list
            gts = [int(i) for i in list(gts)]  # Convert str to int
            rgb_features_path = os.path.join(self.data_path + 'all_rgbs', name + '.npy')
            flow_features_path = os.path.join(self.data_path + 'all_flows', name + '.npy')
            rgb_features = np.load(rgb_features_path)  
            flow_features = np.load(flow_features_path) 
            # Concatenate RGB and Flow features for testing
            concatinated_features = np.concatenate([rgb_features, flow_features], axis=1)  # 32,2048
            # Return features along with ground truth and frame information based on modality
            if self.modality == 'RGB':
                return rgb_features, gts, frames
            elif self.modality == 'FLOW':
                return flow_features, gts, frames
            else:        
                return concatinated_features, gts, frames