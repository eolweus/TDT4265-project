import numpy as np

def dice_score(predb, yb):

    predflat = predb.argmax(dim=1).view(-1)
    yflat = yb.cuda().view(-1)
    intersection = (predflat * yflat).sum()
    
    return (2 * intersection) / (predflat.sum() + yflat.sum())


#### fosskokt, skal se om alt er reiktig
def dice_multiclass(predb, yb, smooth=1e5):
    batch_size = predb.shape[0]
    n_classes = predb.shape[1]
    dice_scores = np.zeros((n_classes, batch_size))
    for batch in range(batch_size):
        pred = predb[batch, :, :, :]
        target_flat = to_cuda(yb)[batch, :, :].view(-1)
        # Ignore IoU for background class ("0")
        for cls in range(1,n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_class_flat = pred[cls, :, :].view(-1)
            intersection = (pred_cls * target).sum()
            dice[cls, batch] = dice[cls,batch] + ((2 * intersection + smooth) / (pred_cls.sum() + target.sum() + smooth)) #.item()

    dice_scores = np.mean(dice, axis=1)
    return np.mean(dice_scores), list(dice_scores)