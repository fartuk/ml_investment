import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable



class QuarterlyTarget:
    '''
    Calculator of target represented as column in quarter-based data.
    Work with quarterly slices of company.
    '''
    def __init__(self, 
                 data_key: str,
                 col: str, 
                 quarter_shift: int=0,
                 n_jobs: int=cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.QuarterlyTarget.calculate`
        col:
            column name for target calculation(like marketcap, revenue)
        quarter_shift:
            number of quarters to shift. 
            e.g. if ``quarter_shift = 0`` than value for current quarter 
            will be returned. 
            If ``quarter_shift = 1`` than value for next quarter 
            will be returned.
            If ``quarter_shift = -1`` than value for previous quarter 
            will be returned.
        '''
        self.data_key = data_key
        self.col = col
        self.quarter_shift = quarter_shift
        self.n_jobs = n_jobs
        self._data_loader = None
        
        
    def _single_ticker_target(self, 
                              ticker_and_dates: Tuple[str,
                                                      List]) -> pd.DataFrame:
        ticker, dates = ticker_and_dates
        quarterly_data = self._data_loader.load([ticker])[::-1]
        quarter_dates = quarterly_data['date'].astype(np.datetime64).values
        vals = []
        for date in dates:
            assert np.datetime64(date) in quarter_dates
            curr_date_mask = quarter_dates == np.datetime64(date)
            curr_quarter_idx = np.where(curr_date_mask)[0][0]
            idx = curr_quarter_idx + self.quarter_shift
            if idx >= 0 and idx < len(quarterly_data):
                value = quarterly_data[self.col].values[idx]
            else:
                value = np.nan
                
            vals.append(value)

        result = pd.DataFrame()
        result['y'] = vals
        result['date'] = dates
        result['ticker'] = ticker

        return result        
        

    def calculate(self, data: Dict, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for dates and tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.QuarterlyTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers and dates
            to calculate targets for. 
            Should have columns: ``["ticker", "date"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company 
            at ``date`` quarter
        '''
        self._data_loader = data[self.data_key]
        grouped = index.groupby('ticker')['date'].apply(lambda x:
                  x.tolist()).reset_index()
        params = [(ticker, dates) for ticker, dates in grouped.values]

        with Pool(self.n_jobs) as p:
            result = []
            for ticker_result in tqdm(p.imap(self._single_ticker_target, params)):
                result.append(ticker_result)

        result = pd.concat(result, axis=0)
        result = result.drop_duplicates(['ticker', 'date'])
        result = pd.merge(index, result, on=['ticker', 'date'], how='left')
        result = result.set_index(['ticker', 'date'])
        result = result.infer_objects()
        
        return result


class QuarterlyDiffTarget:
    '''
    Calculator of target represented as difference between column values
    in current and previous quarter.
    Work with quarterly slices of company.
    '''
    def __init__(self, 
                 data_key: str,
                 col: str,
                 norm: bool=True,
                 n_jobs: int = cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.QuarterlyDiffTarget.calculate`
        col:
            column name for target calculation(like marketcap, revenue)
        norm:
            normalize difference to previous quarter or not
        n_jobs:
            number of threads for calculation         
        '''
        self.curr_target = QuarterlyTarget(data_key=data_key,
                                           col=col,
                                           quarter_shift=0, 
                                           n_jobs=n_jobs)

        self.last_target = QuarterlyTarget(data_key=data_key,
                                           col=col, 
                                           quarter_shift=-1,
                                           n_jobs=n_jobs)
        self.norm = norm

    
    def calculate(self, data: Dict, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for dates and tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.QuarterlyTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers and dates
            to calculate targets for. 
            Should have columns: ``["ticker", "date"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company 
            at ``date`` quarter
        '''
        curr_df = self.curr_target.calculate(data, index)
        last_df = self.last_target.calculate(data, index)
        curr_df['y'] = curr_df['y'] - last_df['y']
        if self.norm:
            curr_df['y'] = curr_df['y'] / np.abs(last_df['y'])

        return curr_df


class QuarterlyBinDiffTarget:
    '''
    Calculator of target represented as binary difference 
    between column values in current and previous quarter.
    Work with quarterly slices of company.
    '''
    def __init__(self, data_key: str, col: str, n_jobs: int=cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.QuarterlyBinDiffTarget.calculate`
        col:
            column name for target calculation(like marketcap, revenue)
        n_jobs:
            number of threads for calculation         
        '''
        self.target = QuarterlyDiffTarget(data_key=data_key,
                                          col=col,
                                          norm=False,
                                          n_jobs=n_jobs)
    
    def calculate(self, data: Dict, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for dates and tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.QuarterlyTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers and dates
            to calculate targets for. 
            Should have columns: ``["ticker", "date"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company 
            at ``date`` quarter
        '''
        target_df = self.target.calculate(data, index)
        target_df.loc[target_df['y'].isnull() == False, 'y'] = \
            target_df.loc[target_df['y'].isnull() == False, 'y'] > 0
        target_df['y'] = target_df['y'].astype(float)
        
        return target_df




class DailyAggTarget:
    '''
    Calculator of target represented as aggregation function of daily values.
    Work with daily slices of company.
    '''
    def __init__(self,
                 data_key: str,
                 col: str,
                 horizon: int=100,
                 foo: Callable=np.mean,
                 n_jobs: int = cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.DailyAggTarget.calculate`
        col:
            column name for target calculation(like marketcap, pe)
        horizon:
            number of days for target calculation.
            If ``horizon > 0`` than values will be get 
            from the future of current date.
            If ``horizon < 0`` than values will be get 
            from the past of current date
        foo:
            function processing target aggregation
        n_jobs:
            number of threads for calculation         
        '''
        self.data_key = data_key
        self.col = col
        self.horizon = horizon
        self.foo = foo
        self.n_jobs = n_jobs
        self._data_loader = None
        
        
    def _single_ticker_target(self, 
                              ticker_and_dates: Tuple[str,
                                                      List]) -> pd.DataFrame:
        ticker, dates = ticker_and_dates
        result = pd.DataFrame()
        result['date'] = dates
        result['ticker'] = ticker
        result['y'] = None

        daily_data = self._data_loader.load([ticker])
        if daily_data is None:
            return result
        daily_data = daily_data[::-1]
        daily_dates = daily_data['date'].astype(np.datetime64).values
        vals = []
        for date in dates:
            if self.horizon >= 0:
                series = daily_data[daily_dates >= np.datetime64(date)]
                series = series[self.col].values[:self.horizon]
            else:
                series = daily_data[daily_dates < np.datetime64(date)]
                series = series[self.col].values[self.horizon:]                
                               
            vals.append(self.foo(series.astype(float)))

        result['y'] = vals

        return result        
        

    def calculate(self, data: Dict, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for dates and tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.DailyAggTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers and dates
            to calculate targets for. 
            Should have columns: ``["ticker", "date"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company 
            at ``date`` day
        '''
        self._data_loader = data[self.data_key]
        grouped = index.groupby('ticker')['date'].apply(lambda x:
                  x.tolist()).reset_index()
        params = [(ticker, dates) for ticker, dates in grouped.values]

        with Pool(self.n_jobs) as p:
            result = []
            for ticker_result in tqdm(p.imap(self._single_ticker_target, params)):
                result.append(ticker_result)

        result = pd.concat(result, axis=0)
        result = result.drop_duplicates(['ticker', 'date'])
        result = pd.merge(index, result, on=['ticker', 'date'], how='left')
        result = result.set_index(['ticker', 'date'])
        
        return result


class DailySmoothedQuarterlyDiffTarget:
    '''
    Feature calculator getting difference between current and last quarter
    smoothed daily column values. Work with company quarter slices.
    '''
    def __init__(self, 
                 daily_data_key: str,
                 quarterly_data_key: str,
                 col: str, 
                 smooth_horizon: int=30,
                 norm: bool=True,
                 n_jobs: int = cpu_count()):
        '''     
        Parameters
        ----------
        daily_data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.DailySmoothedQuarterlyDiffTarget.calculate` 
            for daily data loading
        quarterly_data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.DailySmoothedQuarterlyDiffTarget.calculate` 
            for quarterly data loading
        col:
            column name for target calculation(like marketcap, pe)
        smooth_horizon:
            number of days for target calculation.
            If ``smooth_horizon > 0`` than values for smoothing wiil be get 
            from future of quarter date.
            If ``smooth_horizon < 0`` than values for smoothing will be get 
            from the past of quarter date
        norm:
            normalize result or not
        n_jobs:
            number of threads for calculation         
        '''

        self.norm = norm
        self.daily_target = DailyAggTarget(data_key=daily_data_key,
                                           col=col,
                                           horizon=smooth_horizon,
                                           foo=np.mean,
                                           n_jobs=n_jobs)

        self.prev_quarter_date_target = QuarterlyTarget(
                                            data_key=quarterly_data_key,
                                            col='date',
                                            quarter_shift=-1,
                                            n_jobs=n_jobs)

    def calculate(self, data: Dict, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for dates and tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.DailySmoothedQuarterlyDiffTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers and dates
            to calculate targets for. 
            Should have columns: ``["ticker", "date"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company 
            at ``date`` quarter
        '''
        last_date_df = self.prev_quarter_date_target.calculate(data, index)
        last_date_df = last_date_df.reset_index()
        last_date_df['date'] = last_date_df['y']
        del last_date_df['y']

        curr_df = self.daily_target.calculate(data, index)
        last_df = self.daily_target.calculate(data, last_date_df)

        result = curr_df.copy()
        result['y'] = (curr_df['y'].values - last_df['y'].values)
        if self.norm:
            result['y'] = result['y'].values / last_df['y'].values

        return result




class ReportGapTarget:
    '''
    Calculator of target represented as smoothed gap 
    at some date(i.e. report date).
    Work with daily slices of company.
    '''
    def __init__(self, 
                 data_key: str,
                 col: str,
                 smooth_horizon: int=1,
                 norm: bool=True,
                 n_jobs: int = cpu_count()):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.ReportGapTarget.calculate`
        col:
            column name for target calculation(like marketcap, pe)
        smooth_horizon:
            number of days for column smoothing
        norm:
            normalize gap value or not
        n_jobs:
            number of threads for calculation         
        '''
        self.curr_target = DailyAggTarget(data_key=data_key,
                                          col=col, 
                                          horizon=smooth_horizon,
                                          foo=np.mean,
                                          n_jobs=n_jobs)

        self.last_target = DailyAggTarget(data_key=data_key,
                                          col=col, 
                                          horizon=-smooth_horizon,
                                          foo=np.mean,
                                          n_jobs=n_jobs)
        self.norm = norm
        
        
    def calculate(self, data: Dict, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for dates and tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.ReportGapTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers and dates
            to calculate targets for. 
            Should have columns: ``["ticker", "date"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company 
            at ``date`` time
        '''
        curr_df = self.curr_target.calculate(data, index)
        last_df = self.last_target.calculate(data, index)
        curr_df['y'] = curr_df['y'] - last_df['y']
        if self.norm:
            curr_df['y'] = curr_df['y'] / np.abs(last_df['y'])

        return curr_df        
        
        
        
class BaseInfoTarget:
    '''
    Calculator of target represented by base company information
    '''
    def __init__(self, data_key: str, col: str):
        '''     
        Parameters
        ----------
        data_key:
            key of dataloader in ``data`` argument during 
            :func:`~ml_investment.targets.BaseInfoTarget.calculate`
        col:
            column name for target calculation(like sector, industry)
        '''
        self.data_key = data_key
        self.col = col
        
        
    def calculate(self, data, index: pd.DataFrame) -> pd.DataFrame:
        '''     
        Interface to calculate targets for tickers 
        in index parameter based on data
        
        Parameters
        ----------
        data:
            dict having field named as value in ``data_key`` param of 
            :func:`~ml_investment.targets.BaseInfoTarget.__init__`
            This field should contain class implementing
            ``load(index) -> pd.DataFrame`` interface
        index:
            ``pd.DataFrame`` containing information of tickers
            to calculate targets for. 
            Should have columns: ``["ticker"]``         
                        
        Returns
        -------
        ``pd.DataFrame``
            targets having 'y' column. Index of this dataframe has the same
            values as ``index`` param.
            Each row contains target for ``ticker`` company
        '''
        base_df = data[self.data_key].load(index['ticker'].values)[['ticker', self.col]]
        result = pd.merge(index, base_df, on='ticker', how='left')
        result = result.rename({self.col: 'y'}, axis=1)
        result = result[['ticker', 'y']]
        result = result.set_index(['ticker'])

        return result             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        













