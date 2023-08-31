import numpy as np
import os
import sys
OS_NAME = sys.platform
SEP = '\\' if OS_NAME == "win32" else '/'

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    '''
    The anomaly score inversion function for each frame.

    :param psnr: float
        The anomaly score for a frame.
    :param max_psnr: float
        The maximum anomaly score among all frames in the video.
    :param min_psnr: float
        The minimum anomaly score among all frames in the video.
    '''

    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr+1e-8)))

def anomaly_score_list_inv(psnr_list):
    '''
    The anomaly score inversion function.

    :param psnr_list: list
        The list of anomaly score for frames in a video.
    '''
        
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))
        
    return anomaly_score_list

def denormalize(x, mean, std):
    '''
    The denormalization function.
    
    :param x: np.array
        The input array.
    :param mean: float
        The mean of the input array.
    :param std: float       
        The standard deviation of the input array.
    '''

    return (x + mean) * std

def conf_avg(x, size=11, n_conf=5):
        '''
        The confidence kernel function.

        :param x: np.array
            The input array.
        :param size: int
            The size of the kernel.
        '''
        a = x.copy()
        b = []
        weight = np.array([1, 1, 1, 1, 1.2, 1.6, 1.2, 1, 1, 1, 1])

        for i in range(x.shape[0] - size + 1):
            a_ = a[i:i + size].copy()
            u = a_.mean()
            dif = abs(a_ - u)
            sot = np.argsort(dif)[:n_conf]
            mask = np.zeros_like(dif)
            mask[sot] = 1
            weight_ = weight * mask
            b.append(np.sum(a_ * weight_) / weight_.sum())
        for _ in range(size // 2):
            b.append(b[-1])
            b.insert(0, 1)
        return b
    
def visualize(path_to_folder,anomaly_label,total, reconstructed = None):
    '''
    The visualization function.

    :param path_to_folder: str
        The path to the folder containing the .tif images.
    :param anomaly_label: list
        The list of anomaly labels for frames in a video.
    :param total: list
        The list of anomaly scores for frames in a video.
    :param reconstructed: list
        The list of reconstructed frames in a video.
    '''

    total = (1.2-np.array(total))

    # Replace 'num_images' with the number of .tiff images you have in the folder.
    num_images = len(total)

    # Function to generate frames from the .tif images.
    def generate_frames():
        for i in range(1, num_images + 1):
            image_path = os.path.join(path_to_folder, f"{str(i).zfill(3)}.tif")
            with Image.open(image_path) as img:
                # Convert to RGB if needed (remove the conversion line if the images are already in RGB format).
                img = img.convert("RGB")
                yield img

    larger_embed_limit_in_mb = 150
    plt.rcParams['animation.embed_limit'] = larger_embed_limit_in_mb

    if reconstructed is None:
        # Create the animation using matplotlib.
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create two subplots side by side.
        axs[0][0].axis('off')
        axs[1][0].axis('off')
        axs[1][1].axis('off')
    else:
        # Create the animation using matplotlib.
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create two subplots side by side.

        axs[0][0].axis('off')
        axs[1][1].axis('off')
        axs[1][0].axis('off')

    ims = []
    for i,frame in enumerate(generate_frames()):
        
        im1 = axs[0][0].imshow(frame)
        if i >= 4:

            im2 = axs[0][1].fill_between(range(i-4), 0 , 1 , where=anomaly_label[0:i-4]==1 ,color='red',alpha=0.2)
            im3,= axs[0][1].plot(range(len(total[0:i-4])), total[0:i-4],label='Anomaly Score',color='green') # You can customize the second subplot here.

            if reconstructed is not None:
                im5 = axs[1][0].imshow((reconstructed[0][i-4].astype(np.float32) / 255.0).transpose(1, 2, 0))
                im6 = axs[1][1].imshow((reconstructed[1][i-4].astype(np.float32) / 255.0).transpose(1, 2, 0))

        else:
            im2,= axs[1][0].plot([] ,[],label='Label',color='green') # You can customize the second subplot here.
            im3,= axs[1][0].plot([], [],label='Anomaly Score',color='red') # You can customize the second subplot here.

        axs[0][1].legend(handles=[im3])

        if reconstructed is not None and i >= 4:
            ims.append([im1, im2, im3, im5, im6])
            
        else:
            ims.append([im1, im2, im3])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)

    return ani