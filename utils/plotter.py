import matplotlib.pyplot as plt
import numpy as np
import torch

from augmentation import Augmenter


def plot_dice_history(dice_per_class):
    plt.figure(figsize=(10,8))
    print(dice_per_class[-12:])
    plt.plot(dice_per_class[1::4], label='LV endo dice history')
    plt.plot(dice_per_class[2::4], label='LV evi dice history')
    plt.plot(dice_per_class[3::4], label='LA dice history')
    plt.legend()
    plt.show()
    
def plot_avg_dice_history(valid_dice):
    plt.figure(figsize=(10,8))
    plt.plot(valid_dice, label='Average validation Dice History')
    plt.legend()
    plt.show()

def plot_train_and_val_loss(train_loss, valid_loss):
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.show()

def plot_predicted_segmentations(bs, predicted_batch, image_batch, mask_batch):
    fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
    for i in range(bs):
        ax[i,0].imshow(batch_to_img(image_batch,i))
        ax[i,1].imshow(mask_batch[i])
        ax[i,2].imshow(predb_to_mask(predicted_batch, i))
    plt.show()

def predict_on_batch_and_plot(dataset, unet, bs):
    #predict on the next train batch
    image_batch, mask_batch = next(iter(dataset))
    with torch.no_grad():
        predicted_batch = unet(image_batch.cuda())
    
    plot_predicted_segmentations(bs, predicted_batch, image_batch, mask_batch)

def plot_image_and_mask(data, idx):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(data.open_as_array(idx))
    ax[1].imshow(data.open_mask(idx))
    plt.show()

def plot_with_augmentations(data, idx):
    aug = Augmenter()

    fig, ax = plt.subplots(1,2)
    fig1, fig2 = data.remove_image_borders(idx)
    ax[0].imshow(fig1)
    ax[1].imshow(fig2)
    plt.show()

# Support functions
def batch_to_img(image_batch, idx):
    img = np.array(image_batch[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(prediction_batch, idx):
    p = torch.functional.F.softmax(prediction_batch[idx], 0)
    return p.argmax(0).cpu()
