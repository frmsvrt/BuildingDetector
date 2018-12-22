import warnings
warnings.simplefilter("ignore", UserWarning)

import cv2
import numpy as np

import matplotlib.pyplot as plt

from skimage import transform
import torch
from torchvision import transforms

def sharping(im):
    gK = cv2.getGaussianKernel(21, 5)
    low_pass = cv2.filter2D(im, -1, gK)
    res = im - low_pass
    ret = im + res
    return ret

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

class RandomHFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        p = np.random.randn()
        if p > 0.5:
            return {'sat' : sample['sat'][:, ::-1, :],
                    'mask': sample['mask'][:, ::-1]}
        else:
            return sample

class RandomVFlip(transforms.RandomVerticalFlip):
    def __call__(self, sample):
        p = np.random.randn()
        if p > 0.5:
            return {'sat' : sample['sat'][::-1, :],
                    'mask': sample['mask'][::-1, :]}
        else:
            return sample

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img = sample['sat'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # unsqueeze for the channel dimension

        return {'sat': transforms.functional.to_tensor(sat_img.copy()),
                'mask': torch.from_numpy(map_img.copy()).unsqueeze(0).float().div(255)}


class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {'sat': transforms.functional.normalize(sample['sat'], self.mean, self.std),
                'mask': sample['mask']}

class PaddedInput(object):
    def __call__(self, sample):
        padded = np.pad(sample['sat'], ((62, 62), (62, 62), (0, 0)), 'constant')
        return {'sat' : padded,
                'mask': sample['mask']}

class Sharpnes(object):
    def __call__(self, sample):
        p = np.random.randn()
        if p > 0.5:
            return {'sat' : sharping(sample['sat']),
                    'mask' : sample['mask']}
        else:
            return sample

class Rotate90(object):
    def __call__(self, sample):
        p = np.random.rand()
        if p > 0.5:
            return {'sat' : np.rot90(sample['sat']),
                    'mask': np.rot90(sample['mask'])}
        else:
            return sample

class InvertChannel(object):
    def __call__(self, sample):
        p = np.random.rand()
        if p > 0.5:
            return {'sat' : sample['sat'][:,:,::-1],
                    'mask': sample['mask']}
        else:
            return sample
