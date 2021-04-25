import numpy as np
import torch

def dice_score(predb, yb):

    predflat = predb.argmax(dim=1).view(-1)
    yflat = yb.cuda().view(-1)
    intersection = (predflat * yflat).sum()
    
    return (2 * intersection) / (predflat.sum() + yflat.sum())

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def dice_multiclass(predb, yb, smooth=1e-7):
    num_classes = predb.shape[1]
    
    predb = torch.eye(num_classes)[predb.argmax(1)].permute(0, 3, 1, 2)
    
    yb_encoded = torch.eye(num_classes)[yb.squeeze(1)].permute(0, 3, 1, 2).float()
    yb_encoded = yb_encoded.type(predb.type())
    
    dims = (2,3)
    
    intersection = torch.sum(predb * yb_encoded, dims)
    union_and_intersection = torch.sum(predb + yb_encoded, dims)

    dice_per_class = (2. * intersection / (union_and_intersection + smooth))
    dice_per_class = dice_per_class.mean(0)
    dice_per_class_reduced = dice_per_class[-3:]
    mean_dice = dice_per_class_reduced.mean()
    return mean_dice, dice_per_class



def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements