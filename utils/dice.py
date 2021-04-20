import numpy as np
import torch

def dice_score(predb, yb):

    predflat = predb.argmax(dim=1).view(-1)
    yflat = yb.cuda().view(-1)
    intersection = (predflat * yflat).sum()
    
    return (2 * intersection) / (predflat.sum() + yflat.sum())


def dice_multiclass(predb, yb, smooth=1e5):
    batch_size = predb.shape[0]
    n_classes = predb.shape[1]
    dice_scores = np.zeros((n_classes, batch_size))
    for batch in range(batch_size):
        pred = predb[batch, :, :, :]
        target_flat = yb.cuda()[batch, :, :].view(-1)
        # Ignore IoU for background class ("0")
        for cls in range(1,n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_class_flat = pred[cls, :, :].view(-1)
            intersection = (pred_class_flat * target_flat).sum()
            dice_scores[cls, batch] = dice_scores[cls,batch] + ((2 * intersection + smooth) / (pred_class_flat.sum() + target_flat.sum() + smooth)).item()

    dice_scores = np.mean(dice_scores, axis=1)
    return np.mean(dice_scores) #list(dice_scores)

#### Fosskokt, må endres, men brukes intill videre
def dice_metric(logits, true, eps=1e-7):
    num_classes = logits.shape[1]

    logits = torch.eye(num_classes)[logits.argmax(1)].permute(0, 3, 1, 2)
    
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

    true_1_hot = true_1_hot.type(logits.type())

    dims = (2,3)
    
    # calculate TP
    intersection = torch.sum(logits * true_1_hot, dims)

    # calculate 2TP+FN+FP
    cardinality = torch.sum(logits + true_1_hot, dims)

    dice_loss_class = (2. * intersection / (cardinality + eps)).mean(0)
    dice_loss_mean = dice_loss_class.mean()
    
    return dice_loss_mean #, dice_loss_class
    
