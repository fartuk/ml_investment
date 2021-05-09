import copy
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Union, List, Dict
from .utils import int_hash_of_str

def calc_series_stats(series: Union[List[float], np.array],
                      name_prefix: str='',
                      norm: bool=False) -> Dict[str, float]:
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
    
              
                
       


class QuarterlyFeatures:
    '''
    Feature calculator for qaurtrly-based statistics. 
    Return features for company quarter slices.
    '''
    def __init__(self,
                 data_key: str,
                 columns: List[str],
                 quarter_counts: List[int]=[2, 4, 10],
                 max_back_quarter: int=10,
                 n_jobs: int=cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.QuarterlyFeatures.calculate`
        columns:
            column names for feature calculation(like revenue, debt etc)
        quarter_counts:
            list of number of quarters for statistics calculation. 
            e.g. if ``quarter_counts = [2]`` than statistics will be calculated
            on current and previous quarter
        max_back_quarter:
            max number of company slices in time. 
            If ``max_back_quarter = 1`` than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters 
        n_jobs:
            number of threads for calculation         
        '''
        self.data_key = data_key
        self.columns = columns
        self.quarter_counts = quarter_counts
        self.max_back_quarter = max_back_quarter
        self.n_jobs= n_jobs
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
        quarterly_data = self._data_loader.load([ticker])
        if quarterly_data is None:
            return result
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
        
        
    def calculate(self, data: Dict, index: List[str]) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.features.QuarterlyFeatures.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            list of tickers to calculate features for, i.e. ``['AAPL', 'TSLA']``
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted features with index ``['ticker', 'date']``.
            Each row contains features for ``ticker`` company 
            at ``date`` quarter
        '''
        self._data_loader = data[self.data_key]
        with Pool(self.n_jobs) as p:
            X = []
            for ticker_feats_arr in tqdm(p.imap(self._single_ticker, index)):
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
                 data_key:str,
                 columns: List[str],
                 compare_quarter_idxs: List[int]=[1, 4],
                 max_back_quarter: int=10,
                 n_jobs: int=cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.QuarterlyDiffFeatures.calculate`
        columns:
            column names for feature calculation(like revenue, debt etc)
        compare_quarter_idxs:
            list of back quarter idxs for progress calculation. 
            e.g. if ``compare_quarter_idxs = [1]`` than current quarter 
            will be compared with previous quarter. 
            If ``compare_quarter_idxs = [4]`` than current quarter 
            will be compared with previous year quarter.
        max_back_quarter:
            max number of company slices in time. 
            If ``max_back_quarter = 1`` than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters
        n_jobs:
            number of threads for calculation         
        '''
        self.data_key = data_key
        self.columns = columns
        self.compare_quarter_idxs = compare_quarter_idxs
        self.max_back_quarter = max_back_quarter
        self.n_jobs = n_jobs
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
        quarterly_data = self._data_loader.load([ticker])
        if quarterly_data is None:
            return result
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
        
        
    def calculate(self, data: Dict, index: List[str]) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.features.QuarterlyDiffFeatures.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            list of tickers to calculate features for, i.e. ``['AAPL', 'TSLA']``
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted features with index ``['ticker', 'date']``.
            Each row contains features for ``ticker`` company 
            at ``date`` quarter
        '''
        self._data_loader = data[self.data_key]
        with Pool(self.n_jobs) as p:
            X = []
            for ticker_feats_arr in tqdm(p.imap(self._single_ticker, index)):
                X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])
        
        return X



class HashingEncoder:
    def transform(self, vals):
        result = [int_hash_of_str(str(x)) for x in vals]
        return result
        

class BaseCompanyFeatures:
    '''
    Feature calculator for getting base
    company information(sector, industry etc). 
    Encode categorical columns via hashing label encoding. 
    Return features for current company state.
    '''
    def __init__(self, data_key:str, cat_columns:List[str]):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.BaseCompanyFeatures.calculate`
        cat_columns:
            column names of categorical features for encoding
        '''
        self.data_key = data_key
        self.cat_columns = cat_columns
        self.he = HashingEncoder()

    def calculate(self, data: Dict, index: List[str]) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.features.BaseCompanyFeatures.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            list of tickers to calculate features for, i.e. ``['AAPL', 'TSLA']``
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted features with index ``['ticker']``.
            Each row contains features for ``ticker`` company
        '''
        base_df = data[self.data_key].load(index)
        for col in self.cat_columns:
            base_df[col] = base_df[col].fillna('None')
            base_df[col] = self.he.transform(base_df[col])        
          
           
        result = pd.DataFrame()
        result['ticker'] = index
        
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
                 daily_data_key: str,
                 quarterly_data_key: str,
                 columns: List[str], 
                 agg_day_counts: List[int] = [100, 200], 
                 max_back_quarter: int=10,
                 daily_index=None,
                 n_jobs: int=cpu_count()):
        '''     
        Parameters
        ----------
        daily_data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.DailyAggQuarterFeatures.calculate` 
            for daily data loading
        quarterly_data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.DailyAggQuarterFeatures.calculate`
            for quarterly data loading
        columns:
            column names for feature calculation(like marketcap, pe)
        agg_day_counts:
            list of days counts to calculate statistics on. 
            e.g. if ``agg_day_counts = [100, 200]`` statistics will be 
            calculated based on last 100 and 200 days(separetly). 
        max_back_quarter:
            max number of company slices in time. 
            If ``max_back_quarter = 1`` than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters
        daily_index:
            indexes for ``data[daily_data_key]`` dataloader. 
            If ``None`` than index will be the same as for ``data[quarterly]``.
            I.e. if you want to use this class for calculating 
            commodities features, ``daily_index`` may be 
            list of interesting commodities codes.
            If you want want to use it i.e. for calculating daily price 
            features, ``daily_index`` should be ``None``

        n_jobs:
            number of threads for calculation         
        '''
        self.daily_data_key = daily_data_key
        self.quarterly_data_key = quarterly_data_key
        self.columns = columns
        self.agg_day_counts = agg_day_counts
        self.max_back_quarter = max_back_quarter
        self.daily_index = daily_index
        self.n_jobs = n_jobs
        self._daily_data_loader = None
        self._quarterly_data_loader = None

        
    def _calc_series_feats(self, data: pd.DataFrame,
                           str_prefix: str='') -> Dict[str, float]:
        result = {}
        for day_cnt in self.agg_day_counts:
            for col in self.columns:
                series = data[col].values[:day_cnt][::-1].astype('float')
                name_prefix = '{}_days{}_{}'.format(str_prefix, day_cnt, col)
                feats = calc_series_stats(series, name_prefix=name_prefix,
                                          norm=True)

                result.update(feats)
                               
        return result   
           
        
    def _single_ticker(self, ticker: str) -> List[Dict[str, float]]:
        result = []
        quarterly_data = self._quarterly_data_loader.load([ticker])
        if quarterly_data is None:
            return result
       

        daily_data = copy.deepcopy(self.daily_data) 
        if self.daily_index is None:
            daily_data[''] = self._daily_data_loader.load([ticker])     
        
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        for back_quarter in range(max_back_quarter):
            curr_data = quarterly_data[back_quarter:]
            curr_date = np.datetime64(curr_data['date'].values[0])
            
            feats = {}
            feats['ticker'] = ticker
            feats['date'] = curr_date
            for idx in daily_data.keys():
                daily_dates = daily_data[idx]['date'].values      
                curr_daily_data = daily_data[idx][daily_dates < curr_date]
                daily_feats = self._calc_series_feats(curr_daily_data, idx)
                feats.update(daily_feats)

            result.append(feats)
           
        return result
        
                
    def calculate(self, data: Dict, index: List[str]) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data
        
        Parameters
        ----------
        data:
            dict having fields named as values in ``daily_data_key`` and 
            ``quarterly_data_key`` params of 
            :func:`~ml_investment.features.DailyAggQuarterFeatures.__init__`
            This fields should contain classes implementing
            ``load(index) -> pd.DataFrame`` interfaces
        index:
            list of tickers to calculate features for, i.e. ``['AAPL', 'TSLA']``
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted features with index ``['ticker', 'date']``.
            Each row contains features for ``ticker`` company 
            at ``date`` quarter
        '''
        self._daily_data_loader = data[self.daily_data_key]
        self._quarterly_data_loader = data[self.quarterly_data_key]
        self.daily_data = {}
        if self.daily_index is not None:
            for idx in self.daily_index:    
                self.daily_data[idx] = self._daily_data_loader.load([idx])     

        with Pool(self.n_jobs) as p:
            X = []
            for ticker_feats_arr in tqdm(p.imap(self._single_ticker, index)):
                X.extend(ticker_feats_arr)

        X = pd.DataFrame(X).set_index(['ticker', 'date'])

        return X





class FeatureMerger:
    '''
    Feature calculator that combined two other feature calculators.
    Merge is executed by left. 
    '''
    def __init__(self, fc1, fc2, on=Union[str, List[str]]):
        '''     
        Parameters
        ----------
        fc1:
            first feature calculator 
            implements ``calculate(data: Dict, index) -> pd.DataFrame`` 
            interface
        fc2:
            second feature calculator 
            implements ``calculate(data: Dict, index) -> pd.DataFrame``
            interface
        on:
            columns on which merge the results of executed calculate methods
        '''
        self.fc1 = fc1
        self.fc2 = fc2
        self.on = on
        
        
    def calculate(self, data: Dict, index) -> pd.DataFrame:
        '''     
        Interface to calculate features for tickers 
        based on data
        
        Parameters
        ----------
        data:
            dict having field names needed for ``fc1`` and ``fc2``
            This fields should contain classes implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            indexes dor feature calculators. I.e. if features about companies
            than index may be list of tickers, like ``['AAPL', 'TSLA']``
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted merged features
        '''
        X1 = self.fc1.calculate(data, index)
        X2 = self.fc2.calculate(data, index)
        X = pd.merge(X1, X2, on=self.on, how='left')        
        X.index = X1.index
        return X
 




    























