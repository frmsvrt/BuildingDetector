import warnings
warnings.simplefilter("ignore", UserWarning)

import cv2
import numpy as np

import matplotlib.pyplot as plt

from skimage import transform
import torch
from torchvision import transforms

def show_tensorboard_image(sat_img, 
                           map_img, 
                           out_img, 
                           save_file_path=None, 
                           as_numpy=False):
    """
    Show 3 images side by side for verification on tensorboard. 
    Takes in torch tensors.
    """
    # show different image from the batch
    batch_size = sat_img.size(0)
    img_num = np.random.randint(batch_size)

    f, ax = plt.subplots(1, 3, figsize=(12, 5))
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    ax = ax.ravel()

    ax[0].imshow(sat_img[img_num,:,:,:].cpu().numpy().transpose((1,2,0)))
    ax[0].axis('off')
    ax[1].imshow(map_img[img_num,0,:,:].cpu().numpy())
    ax[1].axis('off')
    ax[2].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
    ax[2].axis('off')

    if save_file_path is not None:
        f.savefig(save_file_path)

    if as_numpy:
        f.canvas.draw()
        width, height = f.get_size_inches() * f.get_dpi()
        mplimage = np.frombuffer(f.canvas.tostring_rgb(), 
                                 dtype='uint8').reshape(int(height), 
                                                       int(width), 3)
        plt.cla()
        plt.close(f)

        return mplimage
    
class ToTensorX(object):
    def __call__(self, sample):
        print('as')
        return transforms.functional.to_tensor(sample)
    
class ToTensorY(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).unsqueeze(0).float().div(255)
    
class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img = sample['sat'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {'sat': transforms.functional.to_tensor(sat_img),
                'mask': torch.from_numpy(map_img).unsqueeze(0).float().div(255)} # unsqueeze for the channel dimension