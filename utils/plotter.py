import matplotlib.pyplot as plt
import torch
from train import batch_to_img, predb_to_mask


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

def predict_on_batch_and_plot(dataset, unet):
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