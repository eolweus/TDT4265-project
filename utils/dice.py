import numpy as np
import torch

def dice_score(predb, yb):

    predflat = predb.argmax(dim=1).view(-1)
    yflat = yb.cuda().view(-1)
    intersection = (predflat * yflat).sum()
    
    return (2 * intersection) / (predflat.sum() + yflat.sum())

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == to_cuda(yb)).float().mean()

def dice_multiclass(predb, yb, smooth=1e-7):
    num_classes = predb.shape[1]
    
    predb = torch.eye(num_classes)[predb.argmax(1)].permute(0, 3, 1, 2)
    predb = predb[:,-3:,:,:]
    
    yb_encoded = torch.eye(num_classes)[yb.squeeze(1)].permute(0, 3, 1, 2)
    yb_encoded = yb_encoded[:,-3:,:,:]

    dims = (2,3)
    
    intersection = torch.sum(predb * yb_encoded, dims)
    union_and_intersection = torch.sum(predb + yb_encoded, dims)

    mean_dice_loss = (2. * intersection / (union_and_intersection + smooth)).mean(0)
    return mean_dice_loss
    
