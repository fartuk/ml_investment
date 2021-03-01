import json
import numpy as np
import pandas as pd

from multiprocessing import Pool
from itertools import repeat
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from copy import deepcopy

from data import SF1Data
from utils import load_json


def calc_series_stats(series, name_prefix=''):
    series = np.array(series).astype('float')
    series = series[~np.isnan(series)] 
    series = list(series)
    if len(series) == 0:
        series = np.array([np.nan])
        
    stats = {
            '{}_mean'.format(name_prefix): np.mean(series),
            '{}_median'.format(name_prefix): np.median(series),
            '{}_max'.format(name_prefix): np.max(series),
            '{}_min'.format(name_prefix): np.min(series),
            '{}_std'.format(name_prefix): np.std(series),
            }
    
    return stats
    
              
                
class FeatureMerger:
    def __init__(self, fc1, fc2, on):
        self.fc1 = fc1
        self.fc2 = fc2
        self.on = on
        
        
    def calculate(self, data_loader, tickers):
        X1 = self.fc1.calculate(data_loader, tickers)
        X2 = self.fc2.calculate(data_loader, tickers)
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
                series = data[col].values[:quarter_cnt][::-1].astype('float')
                name_prefix = 'quarter{}_{}'.format(quarter_cnt, col)

                feats = calc_series_stats(series, name_prefix=name_prefix)
                diff_feats = calc_series_stats(np.diff(series), 
                                name_prefix='{}_diff'.format(name_prefix))

                result.update(feats)
                result.update(diff_feats)
                                
        return result  
        
        
    def _single_ticker(self, ticker):
        result = []
        quarterly_data = self._data_loader.load_quarterly_data([ticker])        
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        for back_quarter in range(max_back_quarter):
            curr_data = quarterly_data[back_quarter:]

            feats = {
                'ticker': ticker, 
                'date': curr_data['date'].values[0],
            }

            series_feats = self._calc_series_feats(curr_data)
            feats.update(series_feats)
            
            result.append(feats)
           
        return result
        
        
    def calculate(self, data_loader, tickers, n_jobs=10):
        self._data_loader = data_loader
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X
    

class QuarterlyDiffFeatures:
    def __init__(self, 
                 columns,
                 compare_quarter_idxs=[1, 4],
                 max_back_quarter=10):
        self.columns = columns
        self.compare_quarter_idxs = compare_quarter_idxs
        self.max_back_quarter = max_back_quarter
        self._data_loader = None
        

    def _calc_diff_feats(self, data):
        result = {}   
        curr_quarter = np.array([data[col].values[0] 
                                    for col in self.columns], dtype='float')              
        for quarter_idx in self.compare_quarter_idxs:
            if len(data) >= quarter_idx + 1:
                compare_quarter = np.array([data[col].values[quarter_idx] 
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
                'date': curr_data['date'].values[0],
            }
               
            diff_feats = self._calc_diff_feats(curr_data)
            feats.update(diff_feats)         
            result.append(feats)
           
        return result
        
        
    def calculate(self, data_loader, tickers, n_jobs=10):
        self._data_loader = data_loader
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X



class BaseCompanyFeatures:
    def __init__(self, cat_columns):
        self.cat_columns = cat_columns
        self.col_to_encoder = {}

    def calculate(self, data_loader, tickers):
        base_df = data_loader.load_base_data()
        is_fitted = True if len(self.col_to_encoder) > 0 else False
        for col in self.cat_columns:
            base_df[col] = base_df[col].fillna('None')
            if is_fitted:
                base_df[col] = self.col_to_encoder[col].transform(base_df[col])                    
            else:      
                le = LabelEncoder()      
                base_df[col] = le.fit_transform(base_df[col])        
                self.col_to_encoder[col] = le
          
           
        result = pd.DataFrame()
        result['ticker'] = tickers
        
        result = pd.merge(result, base_df[['ticker'] + self.cat_columns],
                          on='ticker', how='left')
        
        result = result.set_index(['ticker'])
        
        return result


class PriceFeatures:
    None









































