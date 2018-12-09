import os
import pathlib
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.optim as optim
import torch.functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import UNetSmall
from metrics import BCEDiceLoss, dice_coeff
from datareader import load_train_csv, load_test_data, DataStream
from utils import *

def main(xnames, ynames, num_epochs, lr, sz=256, ckpt=None):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train, x_test, y_train, y_test = train_test_split(xnames,
                                                        ynames,
                                                        random_state=123)
    tfms = transforms.Compose([ToTensorTarget()])
    ds = DataStream(x_train, y_train, sz=256, transform=tfms)
    vds = DataStream(x_test, y_test, sz=256, transform=tfms)
    dm = DataLoader(ds, batch_size=32, num_workers=23)
    vdm = DataLoader(vds, batch_size=32, num_workers=23)

    model = UNetSmall()
    model = model.cuda()

    if ckpt is not None:
        pass

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=18,
                                             gamma=0.01)

    dice_metric = None

    for epoch in range(num_epochs):
        dice = do_epoch(dm=dm,
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        lr_sched=lr_scheduler,
                        mode='train',
        )
        print('Dice: %.3f' % dice )
        if dice_metric is None or dice > dice_metric:
            valid_dice = do_epoch(dm=vdm,
                                  model=model,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  lr_sched=lr_scheduler,
                                  mode='valid',
            )
            if dice_metric is None:
                dice_metric = valid_dice
                print('Dice: %.3f' % valid_dice)
                torch.save(model, './small_unet_%.3f.pt' % dice_metric)
                print('Saved.. ')
            elif valid_dice > dice_metric:
                dice_metric = valid_dice
                print('Val Dice: %.3f' % valid_dice)
                torch.save(model, './small_unet_%.3f.pt' % dice_metric)
                print('Saved.. ')
            else:
                print('Val Dice lower: %.3f' % valid_dice, 'Current dice: %.3f' % dice_metric)

def do_epoch(dm, model, optimizer, criterion, lr_sched, mode='train'):
    dc = []
    if mode == 'train':
        lr_sched.step()
        for idx, sample in enumerate(tqdm(dm)):
            X = Variable(sample['sat'].cuda())
            Y = Variable(sample['mask'].cuda())

            optimizer.zero_grad()
            y_pred = model(X)
            y_pred = torch.nn.functional.sigmoid(y_pred)
            loss = criterion(y_pred, Y)
            loss.backward()
            optimizer.step()

            dc.append(dice_coeff(y_pred, Y))
    elif mode == 'valid':
        model.eval()
        for idx, sample in enumerate(tqdm(dm)):
            X = Variable(sample['sat'].cuda(), volatile=True)
            Y = Variable(sample['mask'].cuda(), volatile=True)

            y_pred = model(X)
            y_pred = torch.nn.functional.sigmoid(y_pred)

            dc.append(dice_coeff(y_pred, Y))

    return np.mean(dc)

if __name__ == '__main__':
    xnames, ynames = load_train_csv()
    main(xnames, ynames, 50, 1e-3)
