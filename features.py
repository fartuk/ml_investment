import json
import numpy as np
import pandas as pd

from multiprocessing import Pool
from itertools import repeat
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data import SF1Data
from utils import load_json


def calc_series_stats(series, name_prefix=''):
    series = np.array(series)
    series = list(series[series != np.array(None)])
    diff_arr = list(np.diff(series))
    if len(series) == 0:
        series = np.array([np.nan])

    if len(diff_arr) == 0:
        diff_arr = np.array([np.nan])   
         
    stats = {
            '{}_mean'.format(name_prefix): np.mean(series),
            '{}_median'.format(name_prefix): np.median(series),
            '{}_max'.format(name_prefix): np.max(series),
            '{}_min'.format(name_prefix): np.min(series),
            '{}_std'.format(name_prefix): np.std(series),

            '{}_diff_mean'.format(name_prefix):np.mean(diff_arr),
            '{}_diff_median'.format(name_prefix):np.median(diff_arr),
            '{}_diff_max'.format(name_prefix):np.max(diff_arr),
            '{}_diff_min'.format(name_prefix):np.min(diff_arr),
            '{}_diff_std'.format(name_prefix):np.std(diff_arr),     
            }
    
    return stats
    
              
                
class FeatureMerger:
    def __init__(self, fc1, fc2, on):
        self.fc1 = fc1
        self.fc2 = fc2
        self.on = on
        
        
    def calculate(self, config, tickers):
        X1 = self.fc1.calculate(config, tickers)
        X2 = self.fc2.calculate(config, tickers)
        X = pd.merge(X1, X2, on=self.on, how='left')        
        X.index = X1.index
        return X
        


class QuarterlyFeatures:
    def __init__(self, 
                 columns,
                 quarter_counts=[2, 4, 10],
                 max_back_quarter=10):
        self.columns = columns
        self.quarter_counts = quarter_counts
        self.max_back_quarter = max_back_quarter
        self._data_loader = None
        

    def _calc_series_feats(self, data, str_prefix=''):
        result = {}
        for quarter_cnt in self.quarter_counts:
            for col in self.columns:
                series = [x[col] for x in data[:quarter_cnt][::-1]]
                name_prefix = 'quarter{}_{}'.format(quarter_cnt, col)
                feats = calc_series_stats(series, name_prefix=name_prefix)
                result.update(feats)

        return result  
        
        
    def _single_ticker(self, ticker):
        result = []
        quarterly_data = self._data_loader.load_quarterly_data([ticker])        
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        for back_quarter in range(max_back_quarter):
            curr_data = quarterly_data[back_quarter:]

            feats = {
                'ticker': ticker, 
                'date': curr_data[0]['date'],
            }
            
            series_feats = self._calc_series_feats(curr_data)
            feats.update(series_feats)
            
            result.append(feats)
           
        return result
        
        
    def calculate(self, data_path, tickers, n_jobs=10):
        self._data_loader = SF1Data(data_path)
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X
    

class QuarterlyDiffFeatures:
    def __init__(self, 
                 columns,
                 compare_idxs=[1, 4],
                 max_back_quarter=10):
        self.columns = columns
        self.compare_idxs = compare_idxs
        self.max_back_quarter = max_back_quarter
        self._data_loader = None
        

    def _calc_diff_feats(self, data):
        result = {}   
        curr_quarter = np.array([data[0][col] for col in self.columns], 
                                                            dtype='float')              
        for quarter_idx in self.compare_idxs:
            if len(data) >= quarter_idx + 1:
                compare_quarter = np.array([data[quarter_idx][col] 
                                  for col in self.columns], dtype='float')
            else:
                compare_quarter = np.array([np.nan for col in self.columns], 
                                                 dtype='float')     
            curr_feats = (curr_quarter - compare_quarter) / compare_quarter
            curr_feats = {'compare{}_{}'.format(quarter_idx, col):val 
                            for col, val in zip(self.columns, curr_feats)}      
            result.update(curr_feats)      
               
        return result
        
        
    def _single_ticker(self, ticker):
        result = []
        quarterly_data = self._data_loader.load_quarterly_data([ticker])        
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        for back_quarter in range(max_back_quarter):
            curr_data = quarterly_data[back_quarter:]

            feats = {
                'ticker': ticker, 
                'date': curr_data[0]['date'],
            }
               
            diff_feats = self._calc_diff_feats(curr_data)
            feats.update(diff_feats)         
            result.append(feats)
           
        return result
        
        
    def calculate(self, data_path, tickers, n_jobs=10):
        self._data_loader = SF1Data(data_path)
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X



class BaseCompanyFeatures:
    def __init__(self, cat_columns):
        self.cat_columns = cat_columns


    def calculate(self, data_path, tickers):
        data_loader = SF1Data(data_path)
        result = pd.DataFrame()
        result['ticker'] = tickers
        tickers_df = data_loader.load_tickers()
        result = pd.merge(result, tickers_df[['ticker'] + self.cat_columns],
                          on='ticker', how='left')

        le = LabelEncoder()
        for col in self.cat_columns:
            result[col] = le.fit_transform(result[col].fillna('None'))
        
        result = result.set_index(['ticker'])
        
        return result


class PriceFeatures:
    None









































