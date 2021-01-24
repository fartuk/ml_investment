import numpy as np
import pandas as pd

from multiprocessing import Pool
from itertools import repeat
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data import load_quarterly_data_cf1, translate_currency_cf1
from utils import load_json
config = load_json("config.json")


def calc_series_stats(series, str_prefix=''):
    series = np.array(series)
    series = list(series[series != np.array(None)])
    diff_arr = list(np.diff(series))
        
    stats = {
            '{}_mean'.format(str_prefix):np.mean(series),
            '{}_median'.format(str_prefix):np.median(series),
            '{}_max'.format(str_prefix):np.max(series, initial=np.nan),
            '{}_min'.format(str_prefix):np.min(series, initial=np.nan),
            '{}_std'.format(str_prefix):np.std(series),

            '{}_diff_mean'.format(str_prefix):np.mean(diff_arr),
            '{}_diff_median'.format(str_prefix):np.median(diff_arr),
            '{}_diff_max'.format(str_prefix):np.max(diff_arr, initial=np.nan),
            '{}_diff_min'.format(str_prefix):np.min(diff_arr, initial=np.nan),
            '{}_diff_std'.format(str_prefix):np.std(diff_arr),
        
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
    if len(quarterly_data) == 0:
        return result
    
#     quarterly_data = translate_currency_cf1(quarterly_data, columns)

    for back_quarter in range(max_back_quarter):
        curr_data = quarterly_data[back_quarter:]
        if len(curr_data) == 0:
            break

        feats = {
            'ticker':ticker, 
            'date':curr_data[0]['date'],
            'marketcap':curr_data[0]['marketcap'],
        }

        for quarter_cnt in [2, 4, 10]:
            series_feats = calc_series_feats(curr_data[:quarter_cnt][::-1], 
                                             columns, 'quarter{}'.format(quarter_cnt))

            feats.update(series_feats)

        result.append(feats)

            
    return result



class QuarterlyFeatureCalculator:
    def __init__(self, columns, max_back_quarter=10):
        self.columns = columns
        self.max_back_quarter=max_back_quarter
        
        
    def calc_feats(self, ticker_list, n_jobs=10):
        p = Pool(n_jobs)
        params_gen = zip(ticker_list, repeat(self.max_back_quarter), repeat(self.columns))
        X = []
        for ticker_feats_arr in tqdm(p.starmap(calc_feats_single_ticker, params_gen)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X)
        X = X[X['marketcap'].isnull()==False]
        
        X = self.calc_global_ticker_feats(X)
        
        return X
    
    
    def calc_global_ticker_feats(self, X):
        tickers_df = pd.read_csv('{}/cf1/tickers.csv'.format(config['data_path']))

        X = pd.merge(X, tickers_df[['ticker', 'sector', 'sicindustry']], on='ticker', how='left')
        le = LabelEncoder()
        X['sector'] = le.fit_transform(X['sector'].fillna('None'))
        X['sicindustry'] = le.fit_transform(X['sicindustry'].fillna('None'))
        
        return X










