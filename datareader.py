import os
import pathlib
import pandas as pd

import numpy as np
import cv2
import skimage.io as io

from torch.utils.data import Dataset
from utils import ToTensorX, ToTensorY

PATH = pathlib.Path('/home/antares/ssd_data/topcoder/spacenet/rgbs/')
TRAIN_RGBS = 'train_rgb'
TEST_RGBS = 'test_rgb'
MASKS_DIR = 'masks'
new_masks = 'new_masks'
TRAIN_DIR = PATH/TRAIN_RGBS
TEST = PATH/TEST_RGBS
CSVS_DIR = pathlib.Path('/home/antares/ssd_data/topcoder/spacenet/SpaceNet-Off-Nadir_Train/summaryData/')

def load_train_csv():
    x_names = []
    y_names = []
    for fn in os.listdir(CSVS_DIR):
        dat = pd.read_csv(CSVS_DIR/fn)
        names = dat['ImageId'][:].tolist()
        names = set(names)
        for idx, fn in enumerate(names):
            y_n = 'mask_' + fn[-14:] + '.tif'
            if fn[0] == 'P':
                fn = fn + '.png'
            else:
                fn = 'Pan-Sharpen_' + fn + '.png'
                x_names.append(PATH/TRAIN_RGBS/fn)
                y_names.append(PATH/MASKS_DIR/y_n)
                
    return x_names, y_names

def load_test_data():
    test_names = [PATH/TEST_RGBS/fn for fn in os.listdir(PATH/TEST_RGBS)]
    return test_names

class DataStream(Dataset):
    def __init__(self, xnames, ynames, sz=256, transform=None):
        self.xnames = xnames
        self.ynames = ynames
        self.sz = sz
        self.transform = transform
        
    def __len__(self):
        return len(self.xnames)
    
    def __getitem__(self, idx):
        x = io.imread(self.xnames[idx])
        y = io.imread(self.ynames[idx])
        if self.sz is not None:
            x, y = list(map(lambda x: cv2.resize(x, (self.sz, self.sz)), [x, y]))
        sample = {'sat' : x, 'mask' : y}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample