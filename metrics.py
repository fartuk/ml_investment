import numpy as np



def median_absolute_relative_error(gt, pred):
    mask = gt != 0
    pred = pred[mask]
    gt = gt[mask]
    vals = np.abs((gt - pred) / gt)
    vals = vals[~np.isnan(vals)]
    return np.median(vals)







