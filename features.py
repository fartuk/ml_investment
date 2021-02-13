import numpy as np
import pandas as pd

from multiprocessing import Pool
from itertools import repeat
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data import load_quarterly_data_cf1, translate_currency_cf1
from utils import load_json


def calc_series_stats(series, name_prefix=''):
    series = np.array(series)
    series = list(series[series != np.array(None)])
    diff_arr = list(np.diff(series))
        
    stats = {
            '{}_mean'.format(name_prefix):np.mean(series),
            '{}_median'.format(name_prefix):np.median(series),
            '{}_max'.format(name_prefix):np.max(series, initial=np.nan),
            '{}_min'.format(name_prefix):np.min(series, initial=np.nan),
            '{}_std'.format(name_prefix):np.std(series),

            '{}_diff_mean'.format(name_prefix):np.mean(diff_arr),
            '{}_diff_median'.format(name_prefix):np.median(diff_arr),
            '{}_diff_max'.format(name_prefix):np.max(diff_arr, initial=np.nan),
            '{}_diff_min'.format(name_prefix):np.min(diff_arr, initial=np.nan),
            '{}_diff_std'.format(name_prefix):np.std(diff_arr),
        
            }
    
    return stats
    
              
                



class QuarterlyFeatures:
    def __init__(self, config, columns, quarter_counts=[2, 4, 10], max_back_quarter=10):
        self.config = config
        self.columns = columns
        self.quarter_counts = quarter_counts
        self.max_back_quarter=max_back_quarter


    def _calc_series_feats(self, data, str_prefix=''):
        result = {}
        for quarter_cnt in self.quarter_counts:
            for col in self.columns:
                series = [x[col] for x in data[:quarter_cnt][::-1]]
                feats = calc_series_stats(series, name_prefix='quarter{}_{}'.format(quarter_cnt, col))
                result.update(feats)
            
        return result  


    def _single_ticker(self, ticker):
        result = []
        quarterly_data = load_quarterly_data_cf1(ticker, self.config)
        #quarterly_data = translate_currency_cf1(quarterly_data, self.columns)
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        for back_quarter in range(max_back_quarter):
            curr_data = quarterly_data[back_quarter:]

            feats = {
                'ticker':ticker, 
                'date':curr_data[0]['date'],
                #'marketcap':curr_data[0]['marketcap'],
            }

            series_feats = self._calc_series_feats(curr_data)
            feats.update(series_feats)

            result.append(feats)
           
        return result
        
        
    def calculate(self, tickers, n_jobs=10):
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X)
        #X = X[X['marketcap'].isnull()==False]
        
        
        return X
    
    

class BaseCompanyFeatures:
    def __init__(self, config, cat_columns):
        self.config = config
        self.cat_columns = cat_columns


    def calculate(self, tickers):
        result = pd.DataFrame()
        result['ticker'] = tickers
        tickers_df = pd.read_csv('{}/cf1/tickers.csv'.format(self.config['data_path']))
        result = pd.merge(result, tickers_df[['ticker'] + self.cat_columns], on='ticker', how='left')

        le = LabelEncoder()
        for col in self.cat_columns:
            result[col] = le.fit_transform(result[col].fillna('None'))
        
        return result














































