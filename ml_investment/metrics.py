import numpy as np
from sklearn.metrics import log_loss


def median_absolute_relative_error(gt, pred):
    mask = gt != 0
    pred = pred[mask]
    gt = gt[mask]
    vals = np.abs((gt - pred) / gt)
    vals = vals[~np.isnan(vals)]
    return np.median(vals)

def mean_absolute_relative_error(gt, pred):
    mask = gt != 0
    pred = pred[mask]
    gt = gt[mask]
    vals = np.abs((gt - pred) / gt)
    vals = vals[~np.isnan(vals)]
    return np.mean(vals)

def nan_log_loss(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    mask = np.isnan(gt) | np.isnan(pred)
    gt = gt[~mask]
    pred = pred[~mask]
    return log_loss(gt, pred)
    
def median_abs_diff(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    result = np.median(np.abs(gt - pred))
    return result
    
def max_rise_norm(series):
    if len(series) == 0:
        return np.nan
    result = (series.max() - series[0]) / series[0]
    return result

def max_fall_norm(series):
    if len(series) == 0:
        return np.nan
    result = (series[0] - series.min()) / series[0]
    return result

def up_std_norm(series):
    if len(series) == 0:
        return np.nan
    up_vals = series[series >= series.mean()]
    result = ((up_vals - series.mean()) ** 2).mean() ** (1/2) / series[0]
    return result

def down_std_norm(series):
    if len(series) == 0:
        return np.nan
    down_vals = series[series < series.mean()]
    result = ((down_vals - series.mean()) ** 2).mean() ** (1/2) / series[0]
    return result 

def std_norm(series):
    if len(series) == 0:
        return np.nan
    result = ((series - series.mean()) ** 2).mean() ** (1/2) / series[0]
    return result   



