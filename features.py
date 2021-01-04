import numpy as np
from data import load_quarterly_data_cf1
from utils import load_json
config = load_json("config.json")


def calc_series_stats(series, str_prefix=''):
    stats = {
            '{}_mean'.format(str_prefix):np.mean(series),
            '{}_max'.format(str_prefix):np.max(series),
            '{}_min'.format(str_prefix):np.min(series),
            '{}_std'.format(str_prefix):np.std(series)
            }
    
    return stats
    
    
def calc_series_feats(data, columns, str_prefix=''):
    feats = {}
    for col in columns:
        series = [x[col] for x in data]
        curr_feats = calc_series_stats(series, str_prefix='{}_{}'.format(str_prefix, col))
        feats.update(curr_feats)
        
    return feats            
                
    
def calc_feats_single_ticker(ticker, max_back_quarter, columns):
    result = []
    
    quarterly_data = load_quarterly_data_cf1(ticker, config)
    
    for back_quarter in range(max_back_quarter):
        try:
            curr_data = quarterly_data[back_quarter:]
            if len(curr_data) == 0:
                break

            feats = {
                'ticker':ticker, 
                'date':curr_data[0]['date'],
                'marketcap':curr_data[0]['marketcap'],
#                 'f1':curr_data[1]['marketcap'],
            }

            for quarter_cnt in [2, 4, 10]:
                series_feats = calc_series_feats(curr_data[:quarter_cnt][::-1], columns, 'quarter{}'.format(quarter_cnt))

                feats.update(series_feats)

            result.append(feats)
        except:
            None
            
    return result









