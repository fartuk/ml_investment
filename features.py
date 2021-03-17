import numpy as np
import pandas as pd

from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Union, List, Dict


def calc_series_stats(series: Union[List[float], np.array],
                      name_prefix: str='',
                      norm: bool=False) -> Dict[str, np.float]:
    '''
    Calculate base statistics on series
            
    Parameters
    ----------
    series:
        series by which statistics are calculated
    name_prefix:
        string prefix of returned features
    norm:
        normilize resulted statistics to first element or not
        
    Returns
    -------
        Dict with calculated features 
    '''
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
    
    if norm:
        stats = {key: stats[key] / np.abs(series[0]) for key in stats}
    
    return stats
    
              
                
class FeatureMerger:
    '''
    Feature calculator that combined two other feature calculators.
    Both should implements calculate(data_loader, tickers) interface
    Merge is executed by left. 
    '''
    def __init__(self, fc1, fc2, on=Union[str, List[str]]):
        '''     
        Parameters
        ----------
        fc1:
            first feature calculator 
            implements calculate(data_loader, tickers: List[str]) ->
                                 pd.DataFrame interface
        fc2:
            second feature calculator 
            implements calculate(data_loader, tickers: List[str]) ->
                                 pd.DataFrame interface
        on:
            columns on which merge the results of executed calculate methods
        '''
        self.fc1 = fc1
        self.fc2 = fc2
        self.on = on
        
        
    def calculate(self, data_loader, tickers: List[str]) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements necessary for feature calculators
            interfaces of data loading
        tickers:
            tickers of companies to calculate features for 
                        
        Returns
        -------
            pd.DataFrame with resulted combined features
        '''
        X1 = self.fc1.calculate(data_loader, tickers)
        X2 = self.fc2.calculate(data_loader, tickers)
        X = pd.merge(X1, X2, on=self.on, how='left')        
        X.index = X1.index
        return X
        


class QuarterlyFeatures:
    '''
    Feature calculator for qaurtrly-based statistics. 
    Return features for company quarter slices.
    '''
    def __init__(self, 
                 columns: List[str],
                 quarter_counts: List[int]=[2, 4, 10],
                 max_back_quarter: int=10):
        '''     
        Parameters
        ----------
        columns:
            column names for feature calculation(like revenue, debt etc)
        quarter_counts:
            list of number of quarters for statistics calculation. 
            e.g. if quarter_counts = [2] than statistics will be calculated
            on current and previous quarter
        max_back_quarter:
            max number of company slices in time. 
            If max_back_quarter = 1 than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters 
        '''
        self.columns = columns
        self.quarter_counts = quarter_counts
        self.max_back_quarter = max_back_quarter
        self._data_loader = None
        

    def _calc_series_feats(self, data: pd.DataFrame,
                           str_prefix: str='') -> Dict[str, float]:
        result = {}
        for quarter_cnt in self.quarter_counts:
            for col in self.columns:
                series = data[col].values[:quarter_cnt][::-1].astype('float')
                name_prefix = 'quarter{}_{}'.format(quarter_cnt, col)

                feats = calc_series_stats(series, name_prefix=name_prefix)
                diff_feats = calc_series_stats(np.diff(series), 
                                               name_prefix='{}_diff'.format(
                                                    name_prefix))

                result.update(feats)
                result.update(diff_feats)
                                
        return result  
        
        
    def _single_ticker(self, ticker:str) -> List[Dict[str, float]]:
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
        
        
    def calculate(self, data_loader, tickers: List[str],
                  n_jobs: int=10) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements load_quarterly_data(tickers: List[str]) -> 
                                                 pd.DataFrame interface
        tickers:
            tickers of companies to calculate features for 
        n_jobs:
            number of threads                
                      
        Returns
        -------
            pd.DataFrame with result features and
            having index ['ticker', 'date']
        '''
        self._data_loader = data_loader
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X
    

class QuarterlyDiffFeatures:
    '''
    Feature calculator for qaurtr-to-another-quarter company
    indicators(revenue, debt etc) progress evaluation.
    Return features for company quarter slices.
    '''
    def __init__(self, 
                 columns: List[str],
                 compare_quarter_idxs: List[int]=[1, 4],
                 max_back_quarter: int=10):
        '''     
        Parameters
        ----------
        columns:
            column names for feature calculation(like revenue, debt etc)
        compare_quarter_idxs:
            list of back quarter idxs for progress calculation. 
            e.g. if compare_quarter_idxs = [1] than current quarter 
            will be compared with previous quarter. 
            If compare_quarter_idxs = [4] than current quarter 
            will be compared with previous year quarter.
        max_back_quarter:
            max number of company slices in time. 
            If max_back_quarter = 1 than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters
        '''
        self.columns = columns
        self.compare_quarter_idxs = compare_quarter_idxs
        self.max_back_quarter = max_back_quarter
        self._data_loader = None
        

    def _calc_diff_feats(self, data: pd.DataFrame) -> Dict[str, float]:
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
        
        
    def _single_ticker(self, ticker: str) -> List[Dict[str, float]]:
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
        
        
    def calculate(self, data_loader, tickers: List[str],
                  n_jobs: int=10) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements load_quarterly_data(tickers: List[str]) -> 
                                                 pd.DataFrame interface
        tickers:
            tickers of companies to calculate features for 
        n_jobs:
            number of threads                
                      
        Returns
        -------
            pd.DataFrame with result features and
            having index ['ticker', 'date']
        '''
        self._data_loader = data_loader
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X



class BaseCompanyFeatures:
    '''
    Feature calculator for base calculating/processing base
    company information(sector, industry etc). 
    Encode categorical columns via label encoding. 
    Return features for current company state.
    '''
    def __init__(self, cat_columns:List[str]):
        '''     
        Parameters
        ----------
        cat_columns:
            column names of categorical features for encoding
        '''
        self.cat_columns = cat_columns
        self.col_to_encoder = {}

    def calculate(self, data_loader, tickers: List[str]) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements load_base_data(tickers: List[str]) -> 
                                            pd.DataFrame interface
        tickers:
            tickers of companies to calculate features for             
                      
        Returns
        -------
            pd.DataFrame with result features and
            having index ['ticker']
        '''        
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



class DailyAggQuarterFeatures:
    '''
    Feature calculator for daily-based statistics for quarter slices.
    Return features for company quarter slices.
    '''
    def __init__(self, 
                 columns: List[str], 
                 agg_day_counts: List[int] = [100, 200], 
                 max_back_quarter: int=10):
        self.columns = columns
        self.agg_day_counts = agg_day_counts
        self.max_back_quarter = max_back_quarter
        
        
    def _calc_series_feats(self, data: pd.DataFrame,
                           str_prefix: str='') -> Dict[str, float]:
        result = {}
        for day_cnt in self.agg_day_counts:
            for col in self.columns:
                series = data[col].values[:day_cnt][::-1].astype('float')
                name_prefix = 'days{}_{}'.format(day_cnt, col)
                feats = calc_series_stats(series, name_prefix=name_prefix,
                                          norm=True)

                result.update(feats)
                               
        return result   
           
        
    def _single_ticker(self, ticker: str) -> List[Dict[str, float]]:
        result = []
        quarterly_data = self._data_loader.load_quarterly_data([ticker])
        daily_data = self._data_loader.load_daily_data([ticker])     
        daily_dates = daily_data['date'].astype(np.datetime64).values      
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        for back_quarter in range(max_back_quarter):
            curr_data = quarterly_data[back_quarter:]
            curr_date = curr_data['date'].values[0]
            
            feats = {
                'ticker': ticker, 
                'date': curr_date,
            }
            
            curr_daily_data = daily_data[daily_dates < np.datetime64(curr_date)]
            daily_feats = self._calc_series_feats(curr_daily_data)
            feats.update(daily_feats)

            result.append(feats)
           
        return result
        
                
    def calculate(self, data_loader, tickers: List[str],
                  n_jobs: int=10) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements 
            load_quarterly_data(tickers: List[str]) -> pd.DataFrame 
            load_daily_data(tickers: List[str]) -> pd.DataFrame             
            interfaces
        tickers:
            tickers of companies to calculate features for 
        n_jobs:
            number of threads                
                      
        Returns
        -------
            pd.DataFrame with result features and
            having index ['ticker', 'date']
        '''
        self._data_loader = data_loader
        p = Pool(n_jobs)
        X = []
        for ticker_feats_arr in tqdm(p.imap(self._single_ticker, tickers)):
            X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])

        return X











    























