import logging
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import sklearn.metrics
import pathlib
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
from configs import cfg

from DatasetLoader import DatasetLoader, TTELoader, ResizedLoader
from utils.checkpoint import CheckPointer
from utils.logger import setup_logger
from utils.dice import dice_multiclass as dice_score
from trainer import do_train

from Unet2D import Unet2D



def do_evaluation(data, model, dice_fn, dataloader):
    running_dice = 0
    model.train(False)
    for x, y in data:
        x = x.cuda()
        y = y.cuda()
        outputs = model(x)
        Dice = dice_fn(outputs, y)
        running_dice  += Dice*data.batch_size
    avg_dice = running_dice / len(dataloader.dataset)
    print("Run results")
    print('-' * 10)
    print('Test Dice: {}'.format(avg_dice))
    print('-' * 10)
    return avg_dice


def start_train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
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


def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def main ():

    #create logger
    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger("U", "logs")
    logger.info("Loaded configuration file {}".format(cfg))
    
    # sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)

    #load the training data
    """cluster_path = '../../../../../work/datasets/medical_project/CAMUS_resized'
    #Path(DATA_BASE_PATH)
    base_path = Path(cluster_path)
    data = DatasetLoader(base_path/'train_gray', 
                        base_path/'train_gt')"""
    
    TTE_train_path = './Data/training'
    TTE_test_path =  './Data/testing'
    #base_path = Path(TTE_FULL_BASE_PATH)
    data = TTELoader(TTE_train_path)
    test_dataset = TTELoader(TTE_test_path)

    #split the training dataset and initialize the data loaders
    print("Train Data length: {}".format(len(data)))
    print("Test Data length: {}".format(len(test_dataset)))
    train_dataset, validation_dataset = torch.utils.data.random_split(data, (450 - cfg.VALIDATION_SIZE, cfg.VALIDATION_SIZE))
    train_data = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_data = DataLoader(validation_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    #test_data = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    if cfg.VISUAL_DEBUG:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(150))
        ax[1].imshow(data.open_mask(150))
        plt.show()
    #print(train_data[1].shape)
    xb, yb = next(iter(train_data))
    print (xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 2 channels as output
    
    unet = Unet2D(1,2)
    logger.info(unet)
    
    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=cfg.LEARN_RATE)

    #do some training 
    train_loss, valid_loss = start_train(unet, train_data, valid_data, loss_fn, opt, dice_score, epochs=cfg.EPOCHS)
    
    # Evaluate network
    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    result = do_evaluation(test_data, unet, dice_score, test_data)
    logger.info("Evaluation result: {}".format(result))

    #plot training and validation losses
    if cfg.VISUAL_DEBUG:
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
    if cfg.VISUAL_DEBUG:
        fig, ax = plt.subplots(cfg.BATCH_SIZE,3, figsize=(15,cfg.BATCH_SIZE*5))
        for i in range(cfg.BATCH_SIZE):
            ax[i,0].imshow(batch_to_img(xb,i))
            ax[i,1].imshow(yb[i])
            ax[i,2].imshow(predb_to_mask(predb, i))

        plt.show()

if __name__ == "__main__":
    main()
