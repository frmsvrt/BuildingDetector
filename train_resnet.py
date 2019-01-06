import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

from model import UNetSmall, UResNet, UNet
from metrics import BCEDiceLoss, dice_coeff
from datareader import load_train_csv, load_test_data, DataStream
from utils import *

from tqdm import tqdm

from sklearn.model_selection import train_test_split

# torch.cuda.set_device(0)

def main(xnames,
         ynames,
         num_epochs,
         lr,
         sz=256,
         bs=30,
         ckpt=False,
         mname='./unet_256_aug.pt',
         ):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train, x_test, y_train, y_test = train_test_split(xnames,
                                                        ynames,
                                                        random_state=123)
    tfms = transforms.Compose([RandomVFlip(),
                               RandomHFlip(),
                               Sharpnes(),
                               Rotate90(),
                               InvertChannel(),
                               ToTensorTarget(),
                               NormalizeTarget(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    ds = DataStream(x_train, y_train, sz=sz, transform=tfms)
    vds = DataStream(x_test, y_test, sz=sz, transform=tfms)
    dm = DataLoader(ds, batch_size=bs, num_workers=23)
    vdm = DataLoader(vds, batch_size=bs, num_workers=23)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt is not False:
        model = torch.load(ckpt).to(device)
        print('Loading model.. Finished.')
    else:
        model = UResNet(num_classes=1).to(device)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=1e-7,
                           )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    dice_metric = None

    for epoch in range(num_epochs):
        dice = do_epoch(dm=dm,
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        lr_sched=lr_scheduler,
                        mode='train',
                        epoch=epoch,
        )
        print('Dice: %.3f' % dice )
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
            torch.save(model, mname)
            print('Saved.. ')
        elif valid_dice > dice_metric:
            dice_metric = valid_dice
            print('Val Dice: %.3f' % valid_dice)
            torch.save(model, mname)
            print('Saved.. ')
        else:
            print('Val Dice lower: %.3f' % valid_dice, 'Current dice: %.3f' % dice_metric)

        with open('./log.log', 'a+') as f:
            print(dice, valid_dice, file=f)

def do_epoch(dm, model, optimizer, criterion, lr_sched, mode='train'):
    dc = []
    L = []
    # lr_sched.step()
    if mode == 'train':
        model.train()
        for idx, sample in enumerate(dm):
            X = Variable(sample['sat'].to(device))
            Y = Variable(sample['mask'].to(device))

            optimizer.zero_grad()
            y_pred = model(X)
            y_pred = torch.nn.functional.sigmoid(y_pred)
            loss = criterion(y_pred, Y)
            loss.backward()
            optimizer.step()

            l = loss.detach().cpu().numpy()

            dc.append(dice_coeff(y_pred, Y))
            L.append(l)

            print('\r', 'Epoch: ', epoch,
                  'step: ,' idx, '|', len(dm),
                  'Loss: %.4f' % np.mean(L),
                  'Dice: %.4f' % np.mean(dc),
                  end=' ')

    elif mode == 'valid':
        model.eval()
        with torch.no_grad():
            for idx, sample in enumerate(dm):
                X = Variable(sample['sat'].cuda())
                Y = Variable(sample['mask'].cuda())

                y_pred = model(X)
                y_pred = torch.nn.functional.sigmoid(y_pred)

                dc.append(dice_coeff(y_pred, Y))

                print('\r', idx, '|', len(dm),
                      'Dice: %.4f' % np.mean(dc),
                      end=' ')

    if model == 'valid':
       lr_sched.step(np.mean(dc))
    return np.mean(dc)

if __name__ == '__main__':
    xnames, ynames = load_train_csv()
    ckpt = False
    # train on 256x256 crops
    main(xnames=xnames,
         ynames=ynames,
         num_epochs=100,
         lr=1e-4,
         sz=256,
         bs=32,
         ckpt=ckpt)
    # train on 512x512 crops
    if os.path.exists('./unet_256_aug.pt'):
        ckpt = './unet_256_aug.pt'
    main(xnames=xnames,
         ynames=ynames,
         num_epochs=50,
         lr=1e-4,
         sz=512,
         bs=14,
         ckpt=ckpt,
         mname='./unet_512_aug.pt')

    # train on 1024x1024 crops
    if os.path.exists('./unet_512_aug.pt'):
        ckpt = './unet_512_aug.pt'
    main(xnames=xnames,
         ynames=ynames,
         num_epochs=30,
         lr=1e-5,
         sz=1024,
         bs=4,
         ckpt=ckpt,
         mname='./unet_1024_aug_2.0.pt')
