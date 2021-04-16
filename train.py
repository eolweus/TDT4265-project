import logging
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import sklearn.metrics
from utils.checkpoint import CheckPointer
from utils.logger import setup_logger
from trainer import do_train

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from DatasetLoader import DatasetLoader, TTELoader, ResizedLoader
from Unet2D import Unet2D

from decouple import config

TTE_BASE_PATH=config('TTE_BASE_PATH')
TTE_FULL_BASE_PATH=config('TTE_FULL_BASE_PATH')
TTE_TEST_BASE_PATH=config('TTE_FULL_TEST_BASE_PATH')
TEE_BASE_PATH=config('TEE_BASE_PATH')


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    model.cuda()
    
    ### Setup for checkpointing
    save_folder = "outputs"
    logger = logging.getLogger('U.trainer')
    arguments = {"epoch": 0, "step": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, save_folder, save_to_disk, logger,
        )
    extra_checkpoint_data = checkpointer.load() # Load last checkpoint
    arguments.update(extra_checkpoint_data) 
    ####
    
    # The trainer has been moved to trainer.py
    train_loss, valid_loss = do_train(model,train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs, checkpointer, arguments)
    return train_loss, valid_loss


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_score(predb, yb):

    predflat = predb.argmax(dim=1).view(-1)
    yflat = yb.cuda().view(-1)
    intersection = (predflat * yflat).sum()
    
    return (2 * intersection) / (predflat.sum() + yflat.sum())


def dice_score(predb, yb):
    predflat = predb.argmax(dim=1).view(-1)
    yflat = yb.view(-1)
    intersection = (predflat * yflat).sum()
    
    return (2 * intersection) / (predflat.sum() + yflat.sum())

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def main ():
    #enable if you want to see some plotting
    visual_debug = True

    #batch size
    bs = 12

    #epochs
    epochs_val = 50

    #learning rate
    learn_rate = 0.01

    # sets the matplotlib display backend (most likely not needed)
    # mp.use('TkAgg', force=True)

    # base_path = Path(TTE_BASE_PATH)
    # data = ResizedLoader(base_path)

    base_path = Path(TTE_FULL_BASE_PATH)
    data = TTELoader(base_path)

    print(len(data))

    train_partition = 2*len(data)//3
    val_partition = len(data)-train_partition

    #split the training dataset and initialize the data loaders
    train_dataset, valid_dataset = torch.utils.data.random_split(data, (train_partition, val_partition))
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=bs, shuffle=True)

    if visual_debug:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(150))
        ax[1].imshow(data.open_mask(150))
        plt.show()

    xb, yb = next(iter(train_data))
    print (xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1,2)

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    #do some training
    train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, dice_score, epochs=epochs_val)

    #plot training and validation losses
    if visual_debug:
        plt.figure(figsize=(10,8))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.legend()
        plt.show()

    #predict on the next train batch (is this fair?)
    xb, yb = next(iter(train_data))
    with torch.no_grad():
        predb = unet(xb.cuda())

    #show the predicted segmentations
    if visual_debug:
        fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
        for i in range(bs):
            ax[i,0].imshow(batch_to_img(xb,i))
            ax[i,1].imshow(yb[i])
            ax[i,2].imshow(predb_to_mask(predb, i))

        plt.show()

if __name__ == "__main__":
    main()
