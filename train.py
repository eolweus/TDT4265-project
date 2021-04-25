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
from decouple import config
import numpy as np

from DatasetLoader import DatasetLoader, TTELoader, ResizedLoader, TEELoader
from utils.checkpoint import CheckPointer
from utils.logger import setup_logger
from utils.dice import dice_multiclass, acc_metric
import utils.plotter as plotter
from trainer import do_train
from Unet2D import Unet2D


TTE_BASE_PATH=config('TTE_BASE_PATH')
TTE_FULL_BASE_PATH=config('TTE_FULL_BASE_PATH')
TTE_FULL_TEST_BASE_PATH=config('TTE_FULL_TEST_BASE_PATH')
TEE_BASE_PATH=config('TEE_BASE_PATH')

def do_evaluation(data, model, dice_fn):
    running_dice = 0
    model.train(False)
    for x, y in data:
        x = x.cuda()
        y = y.cuda()
        outputs = model(x)
        Dice = dice_fn(outputs, y)
        running_dice  += Dice*data.batch_size
    avg_dice = running_dice / len(data.dataset)
    print("Run results")
    print('-' * 10)
    print('Test Dice: {}'.format(avg_dice))
    print('-' * 10)
    return avg_dice


def start_train(model, train_dl, valid_dl, loss_fn, optimizer, dice_fn, acc_fn, epochs=1):
    model.cuda()
    
    ### Setup for checkpointing
    logger = logging.getLogger('U.trainer')
    arguments = {"epoch": 0, "step": 0, "train_loss": [], "valid_loss": [], 'valid_dice': [], 'pixel_acc': []}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, save_to_disk, logger
    )
    if cfg.TRAINING.USE_CHECKPOINT:
        extra_checkpoint_data = checkpointer.load() # Load last checkpoint
        arguments.update(extra_checkpoint_data) 
    
    # The trainer has been moved to trainer.py
    train_loss, valid_loss, valid_dice, pixel_acc = do_train(model,train_dl, valid_dl, loss_fn, optimizer, dice_fn, acc_fn, epochs, checkpointer, arguments)
    return train_loss, valid_loss, valid_dice, pixel_acc

def evauate_and_log_results(logger, unet, tte_test_data, tee_test_data):
    # image = reader.Execute();
    logger.info('Start evaluating on TTE data...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    result = do_evaluation(tte_test_data, unet, dice_multiclass)
    logger.info("Evaluation result: {}".format(result))
    
    logger.info('Start evaluating on TEE data...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    result = do_evaluation(tee_test_data, unet, dice_multiclass)
    logger.info("Evaluation result: {}".format(result))

def get_dataset_path(): 
    paths = {
        "TTE": TTE_FULL_BASE_PATH,
        "RESIZED": TTE_BASE_PATH,
    }
    assert cfg.DATASET in paths.keys()\
        , print("ASSERTION ERROR: config dataset value is not a dataset")
    return paths[cfg.DATASET.upper()]

def create_dataset(use_transforms=False):
    loaders = {
        "TTE": TTELoader,
        "RESIZED": ResizedLoader,
    }
    assert cfg.DATASET in loaders.keys()\
        , print("ASSERTION ERROR: config dataset value is not a dataset")
    return loaders[cfg.DATASET](get_dataset_path(), use_transforms)

def load_test_data():
    tte_test_data = TTELoader(TTE_FULL_TEST_BASE_PATH)
    tee_test_data = TEELoader(TEE_BASE_PATH)
    return tte_test_data, tee_test_data

def main ():
    #create logger
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger("U", "logs")
    logger.info("Loaded configuration file {}".format(cfg))

    lr = cfg.SOLVER.LEARN_RATE
    bs = cfg.SOLVER.BATCH_SIZE
    test_bs = cfg.SOLVER.TEST_BATCH_SIZE
    epochs = cfg.TRAINING.EPOCHS
    
    # sets the matplotlib display backend (most likely not needed)
    # mp.use('TkAgg', force=True)

    data = create_dataset()
    tte_test_dataset, tee_test_dataset = load_test_data()

    #split the training dataset and initialize the data loaders
    print("Train Data length: {}".format(len(data)))
    train_partition = int(cfg.TRAINING.TRAIN_PARTITION*len(data))
    val_partition = len(data)-train_partition

    # split the training dataset and initialize the data loaders
    train_dataset, valid_dataset = torch.utils.data.random_split(data, (train_partition, val_partition))
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    tte_test_data = DataLoader(tte_test_dataset, batch_size=test_bs, shuffle=True)
    tee_test_data = DataLoader(tee_test_dataset, batch_size=test_bs, shuffle=True)

    # TODO: set up test set

    # TODO: rotation of tee is implemented in getitem, not in open_as_array
    # TODO: check if you can visualize a rotated TEE mask
    if cfg.TRAINING.VISUAL_DEBUG:
        plotter.plot_image_and_mask(data, 1)
    xb, yb = next(iter(train_data))
    print (xb.shape, yb.shape)

    # build the Unet2D with default input and output channels as given by config
    
    unet = Unet2D()
    # logger.info(unet)
    
    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=lr)

    #do some training 
    train_loss, valid_loss, valid_dice, pixel_acc = start_train(unet, train_data, valid_data, loss_fn, opt, dice_multiclass, acc_metric, epochs=epochs)
    
    # Evaluate networke(f)
    evauate_and_log_results(logger, unet, tte_test_data, tee_test_data)

    # plot training and validation losses
    if cfg.TRAINING.VISUAL_DEBUG:
        plotter.plot_train_and_val_loss(train_loss, valid_loss)

    # show the predicted segmentations
    if cfg.TRAINING.VISUAL_DEBUG:
        plotter.predict_on_batch_and_plot(tte_test_data, unet, test_bs)

    # TODO: add test parameter to config
    if cfg.TRAINING.VISUAL_DEBUG:
        plotter.predict_on_batch_and_plot(tee_test_data, unet, test_bs)


if __name__ == "__main__": 
    main()
