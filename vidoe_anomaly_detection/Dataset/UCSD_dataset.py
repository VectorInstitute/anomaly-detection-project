import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import sys
OS_NAME = sys.platform
SEP = '\\' if OS_NAME == "win32" else '/'
import torchvision.transforms as transforms

rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, c):
    '''
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: str
        the full path of image
    :param resize_height: int
        resized height
    :param resize_width: int
        resized width
    '''
    
    if c == 1:
        image_decoded = np.repeat(cv2.imread(filename, 0)[:, :, None], 3, axis=-1)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized




class UCSD_Dataset(data.Dataset):
    '''
    UCSD Pedestrian Dataset

    :param video_folder: str
        The path of video folder.
    :param transform: torch.transform
        The data transform.
    :param resize_height: int
        The height of resized input image.
    :param resize_width: int    
        The width of resized input image.
    :param train: bool
        Train or test.
    :param time_step: int
        The number of input frames.
    :param num_pred: int
        The number of predicted frames.
    :param c: int
        The number of channels of input image.
    '''

    def __init__(self, video_folder, transform, resize_height, resize_width, train ,time_step=4, num_pred=1, c=1):
        self.c = c
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        if train:
            self.setup()
            self.samples = self.get_all_samples()
        else:
            self.samples = self.get_one_sample(video_folder)
        
        
    def setup(self):
        '''
        Setting up dataset.
        '''

        videos = glob.glob(os.path.join(self.dir, '*'))
        for idx, video in enumerate(sorted(videos)):
            video_name = video.split(SEP)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['idx'] = idx
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            if self.videos[video_name]['length']==0:
                del self.videos[video_name]
            
            
    def get_all_samples(self):
        '''
        Get all videos in dataset.
        '''

        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split(SEP)[-1]
            if video_name[-1].isdigit():
                for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                    frames.append(self.videos[video_name]['frame'][i])
                            
        return frames     

    def get_one_sample(self,video_path):
        '''
        Get one video in dataset.

        :param video_path: str
            The path of video folder.
        '''

        frames = []
        video_name = video_path.split(SEP)[-1]
        self.videos[video_name] = {}
        self.videos[video_name]['path'] = video_path
        self.videos[video_name]['frame'] = glob.glob(os.path.join(video_path, '*.tif'))
        self.videos[video_name]['frame'].sort()
        self.videos[video_name]['idx'] = 0
        self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
        for i in range(len(self.videos[video_name]['frame'])-self._time_step):
            frames.append(self.videos[video_name]['frame'][i])
                            
        return frames          
            
        
    def __getitem__(self, index):
        '''
        Get item from dataset.

        :param index: int
            The index of item.
        '''

        video_name = self.samples[index].split(SEP)[-2]
        frame_name = int(self.samples[index].split(SEP)[-1].split('.')[-2])
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i-1], self._resize_height, self._resize_width, self.c)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0),self.transform(np.array([[self.videos[video_name]['idx']]]))
        
        
    def __len__(self):
        return len(self.samples)
    

def load_train_data(train_folder, data_config):
    '''
    Load training data.

    :param train_folder: str
        The path of training folder.
    :param data_config: dict
        The config of training data.
    '''

    train_dataset = UCSD_Dataset(train_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), train = True ,resize_height=data_config['h'], resize_width=data_config['w'], time_step=data_config['time_step'], c=data_config['c'])
    return train_dataset

def load_test_data(test_folder,data_config, inference_video_name=None):
    '''
    Load testing data.
    
    :param test_folder: str 
        The path of testing folder.
    :param data_config: dict
        The config of testing data.
    :param inference_video_name: str
        The name of inference video.
    '''

    if inference_video_name is None:
        videos_path = sorted(glob.glob(os.path.join(test_folder, '*')))
        videos_list = []

        for video in videos_path:
            video_name = video.split('/')[-1]
            if video_name[-1].isdigit():
                videos_list.append(video_name)
    else:
        videos_list = [inference_video_name]

    test_datasets = []
    labels_list = []


    for video_name in videos_list:

        test_dataset = UCSD_Dataset(test_folder+'/'+video_name, transforms.Compose([
                            transforms.ToTensor(),
                            ]), train = False ,resize_height=data_config['h'], resize_width=data_config['w'], time_step=data_config['time_step'], c=data_config['c'])
        labels = np.load(test_folder + '/Labals/frame_labels_' + video_name + '.npy')
        
        test_datasets.append(test_dataset)
        labels_list += list(labels)

    return test_datasets, np.array(labels_list)
