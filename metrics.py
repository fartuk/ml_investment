import numpy as np



def median_absolute_relative_error(gt, pred):
    mask = gt != 0
    pred = pred[mask]
    gt = gt[mask]
    vals = np.abs((gt - pred) / gt)
    vals = vals[~np.isnan(vals)]
    return np.median(vals)



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



