import copy
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Union, List, Dict, Callable
from .utils import int_hash_of_str, get_quarter_idx

def calc_series_stats(series: Union[List[float], np.array],
                      stats: Dict[str, Callable]={'mean': np.mean,
                                                  'median': np.median,
                                                  'max': np.max,
                                                  'min': np.min,
                                                  'std': np.std},
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
        
    result = {'{}_{}'.format(name_prefix, key): stats[key](series) 
              for key in stats}
    
    if norm:
        result = {key: result[key] / np.abs(series[0]) for key in result}
    
    return result
    
              
                
       


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
                 min_back_quarter: int=0,
                 stats: Dict[str, Callable]={'mean': np.mean,
                                             'median': np.median,
                                             'max': np.max,
                                             'min': np.min,
                                             'std': np.std},
                 calc_stats_on_diffs: bool=True,
                 data_preprocessing: Callable=None,
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
            max bound of company slices in time. 
            If ``max_back_quarter = 1`` than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters 
        min_back_quarter:
            min bound of company slices in time. 
            If ``min_back_quarter = 0`` (default) than features will be calculated
            for all quarters. 
            If ``min_back_quarter = 2`` than current and previous quarter slices 
            will not be used for feature calculation 
        stats:
            aggregation functions for features calculation.
            Should be as ``Dict[str, Callable]``.
            Keys of this dict will be used as features names prefixes.
            Values of this dict should implement 
            ``foo(x:List) -> float`` interface
        calc_stats_on_diffs:
            calculate statistics on series diffs( ``np.diff(series)`` ) or not
        data_preprocessing:
            function implemening ``foo(x) -> x_`` interface. 
            It will be used before feature calculation.
        n_jobs:
            number of threads for calculation         
        '''
        self.data_key = data_key
        self.columns = columns
        self.quarter_counts = quarter_counts
        self.max_back_quarter = max_back_quarter
        self.min_back_quarter = min_back_quarter
        self.stats = stats
        self.calc_stats_on_diffs = calc_stats_on_diffs
        self.data_preprocessing = data_preprocessing
        self.n_jobs = n_jobs
        self._data_loader = None
        

    def _calc_series_feats(self, data: pd.DataFrame,
                           str_prefix: str='') -> Dict[str, float]:
        result = {}
        for quarter_cnt in self.quarter_counts:
            for col in self.columns:
                series = data[col].values[:quarter_cnt][::-1].astype('float')
                name_prefix = 'quarter{}_{}'.format(quarter_cnt, col)

                feats = calc_series_stats(series=series,
                                          stats=self.stats,
                                          name_prefix=name_prefix)
                result.update(feats)

                if self.calc_stats_on_diffs:
                    diff_feats = calc_series_stats(series=np.diff(series),
                                                   stats=self.stats,
                                                   name_prefix='{}_diff'\
                                                    .format(name_prefix))
                    result.update(diff_feats)
                                
        return result  
        
        
    def _single_ticker(self, ticker:str) -> List[Dict[str, float]]:
        result = []
        quarterly_data = self._data_loader.load([ticker])
        
        if quarterly_data is None:
            return result

        if self.data_preprocessing is not None:
            quarterly_data = self.data_preprocessing(quarterly_data)
        
        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        min_back_quarter = min(self.min_back_quarter, len(quarterly_data) - 1)
        assert min_back_quarter <= max_back_quarter
        
        for back_quarter in range(min_back_quarter, max_back_quarter):
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
                 min_back_quarter: int=0,
                 norm: bool=True,
                 data_preprocessing: Callable=None,
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
            max bound of company slices in time. 
            If ``max_back_quarter = 1`` than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters 
        min_back_quarter:
            min bound of company slices in time. 
            If ``min_back_quarter = 0`` (default) than features will be calculated
            for all quarters. 
            If ``min_back_quarter = 2`` than current and previous quarter slices 
            will not be used for feature calculation
        norm:
            normalize to compare quarter or not
        data_preprocessing:
            function implemening ``foo(x) -> x_`` interface. 
            It will be used before feature calculation.
        n_jobs:
            number of threads for calculation         
        '''
        self.data_key = data_key
        self.columns = columns
        self.compare_quarter_idxs = compare_quarter_idxs
        self.max_back_quarter = max_back_quarter
        self.min_back_quarter = min_back_quarter
        self.norm = norm
        self.data_preprocessing=data_preprocessing
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
            curr_feats = curr_quarter - compare_quarter
            
            if self.norm:
                curr_feats = curr_feats / compare_quarter

            curr_feats = {'compare{}_{}'.format(quarter_idx, col):val 
                            for col, val in zip(self.columns, curr_feats)}      
            result.update(curr_feats)      
               
        return result
        
        
    def _single_ticker(self, ticker: str) -> List[Dict[str, float]]:
        result = []
        quarterly_data = self._data_loader.load([ticker])
        
        if quarterly_data is None:
            return result
        
        if self.data_preprocessing is not None:
            quarterly_data = self.data_preprocessing(quarterly_data)

        max_back_quarter = min(self.max_back_quarter, len(quarterly_data) - 1)
        min_back_quarter = min(self.min_back_quarter, len(quarterly_data) - 1)
        assert min_back_quarter <= max_back_quarter

        for back_quarter in range(min_back_quarter, max_back_quarter):
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
                 agg_day_counts: List[Union[int, np.timedelta64]] = [100, 200], 
                 max_back_quarter: int=10,
                 min_back_quarter: int=0,
                 daily_index=None,
                 stats: Dict[str, Callable]={'mean': np.mean,
                                             'median': np.median,
                                             'max': np.max,
                                             'min': np.min,
                                             'std': np.std},
                 norm: bool=True,
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
            max bound of company slices in time. 
            If ``max_back_quarter = 1`` than features will be calculated
            for only current company quarter. 
            If max_back_quarter is larger than total number of
            quarters for company than features will be calculated 
            for all quarters 
        min_back_quarter:
            min bound of company slices in time. 
            If ``min_back_quarter = 0`` (default) than features will be calculated
            for all quarters. 
            If ``min_back_quarter = 2`` than current and previous quarter slices 
            will not be used for feature calculation 
        daily_index:
            indexes for ``data[daily_data_key]`` dataloader. 
            If ``None`` than index will be the same as for ``data[quarterly]``.
            I.e. if you want to use this class for calculating 
            commodities features, ``daily_index`` may be 
            list of interesting commodities codes.
            If you want want to use it i.e. for calculating daily price 
            features, ``daily_index`` should be ``None``
        stats:
            aggregation functions for features calculation.
            Should be as ``Dict[str, Callable]``.
            Keys of this dict will be used as features names prefixes.
            Values of this dict should implement 
            ``foo(x:List) -> float`` interface
        norm:
            normalize daily stats or not
        n_jobs:
            number of threads for calculation         
        '''
        self.daily_data_key = daily_data_key
        self.quarterly_data_key = quarterly_data_key
        self.columns = columns
        self.agg_day_counts = agg_day_counts
        self.max_back_quarter = max_back_quarter
        self.min_back_quarter = min_back_quarter
        self.daily_index = daily_index
        self.stats = stats
        self.norm = True
        self.n_jobs = n_jobs
        self._daily_data_loader = None
        self._quarterly_data_loader = None

        
    def _calc_series_feats(self, data: pd.DataFrame,
                           str_prefix: str='') -> Dict[str, float]:
        result = {}
        if len(data) == 0:
            return result
        for day_cnt in self.agg_day_counts:
            if type(day_cnt) == int:
                curr_data = data[:day_cnt]
            elif type(day_cnt) == np.timedelta64:
                daily_dates = data['date'].values
                curr_data = data[daily_dates > daily_dates[0] - day_cnt]

            for col in self.columns:
                series = curr_data[col].values[::-1].astype('float')
                name_prefix = '{}_days{}_{}'.format(str_prefix, str(day_cnt), col)
                feats = calc_series_stats(series=series,
                                          stats=self.stats,
                                          name_prefix=name_prefix,
                                          norm=self.norm)

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
        min_back_quarter = min(self.min_back_quarter, len(quarterly_data) - 1)
        assert min_back_quarter <= max_back_quarter
        
        for back_quarter in range(min_back_quarter, max_back_quarter):
            curr_data = quarterly_data[back_quarter:]
            curr_date = np.datetime64(curr_data['date'].values[0])
            
            feats = {}
            feats['ticker'] = ticker
            feats['date'] = curr_date
            for idx in daily_data.keys():
                if daily_data[idx] is not None:
                    daily_dates = daily_data[idx]['date'].values      
                else:
                    continue
                
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


class RelativeGroupFeatures:
    '''
    Feature calculator for features relative to some group median.
    I.e. calculate revenue growth relative to median in sector/industry.
    '''
    def __init__(self,
                 feature_calculator, 
                 group_data_key: str,
                 group_col: str,
                 relation_foo = lambda x, y: x - y,
                 keep_group_feats=False,
                ):
        '''     
        Parameters
        ----------
        feature_calculator:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.DailyAggQuarterFeatures.calculate` 
            for daily data loading
        group_data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.features.RelativeGroupFeatures.calculate`
            for loading data having ``group_col``
        group_col:
            column name for groups in which median values will be calculated
        relation_foo:
            function implementing ``foo(x, y) -> z`` interface.
            E.g. if foo = lambda x: x - y, than resulted features will be 
            calculated as difference between current company features 
            and group median features.
        keep_group_feats:
            return group median features or not 
        '''
        self.feature_calculator = feature_calculator
        self.group_data_key = group_data_key
        self.group_col = group_col
        self.relation_foo = relation_foo
        self.keep_group_feats = keep_group_feats
        
        
    def calculate(self, data, index):
        '''     
        Interface to calculate features for tickers 
        based on data
        
        Parameters
        ----------
        data:
            dict having fields named as values in ``group_data_key`` and 
            necessary for ``feature_calculator`` keys.
            This fields should contain classes implementing
            ``load(index) -> pd.DataFrame`` interfaces
        index:
            index needed for ``feature_calculator.calculate()``
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted features with index as in 
            ''feature_calculator.calculate``.
        '''

        X = self.feature_calculator.calculate(data, index)
        index_cols = list(X.index.names)
        cols = X.columns

        group_df = data[self.group_data_key].load(index)[['ticker',
                                                          self.group_col]]
        X = pd.merge(X.reset_index(), group_df, on='ticker', how='left')
        X['q_idx'] = X['date'].apply(lambda x: get_quarter_idx(x))

        mean_df = X.groupby([self.group_col, 'q_idx']).median()
        mean_df.columns = ['{}_median_{}'.format(self.group_col, x) 
                                            for x in mean_df.columns]
        mean_df = mean_df.reset_index()
        X = pd.merge(X, mean_df, on=[self.group_col, 'q_idx'], how='left')

        for col in cols:
            new_col = 'rel_to_{}_{}'.format(self.group_col, col)
            mean_col = '{}_median_{}'.format(self.group_col, col)
            X[new_col] = self.relation_foo(X[col], X[mean_col])
            
        keep_cols = [x for x in X.columns if 'rel_to_' in x]
        if self.keep_group_feats:
            keep_cols += [x for x in X.columns 
                            if '{}_median_'.format(self.group_col) in x]
            
        keep_cols += index_cols
        
        return X[keep_cols].set_index(index_cols)



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
 




    























