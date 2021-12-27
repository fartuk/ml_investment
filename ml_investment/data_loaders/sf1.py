'''
Loaders for dataset provided by https://www.quandl.com/databases/SF1/data. Data may be downloaded by script 
:func:`~ml_investment.download_scripts.download_sf1.main`

Expected structure of dataset
    | SF1
    | ├── core_fundamental
    | │   ├── AAPL.json
    | │   ├── FB.json
    | │   └── ...
    | ├── daily
    | │   ├── AAPL.json
    | │   ├── FB.json
    | │   └── ...
    | └── tickers.zip       
'''

import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from ..utils import load_json, load_config




def _load_df(json_path: str) -> pd.DataFrame:
    '''
    Load data from SF1 .json file and convert it to pd.DataFrame
    
    Parameters
    ----------
    json_path:
        path to SF1 .json path
        
    Returns
    -------
        ``pd.DataFrame`` content of file
    '''
    data = load_json(json_path)
    df = pd.DataFrame(data['datatable']['data'])
    if len(df) == 0:
        columns = [x['name'] for x in data['datatable']['columns']]
        df = pd.DataFrame(columns=columns, dtype=object)
    else:
        df.columns = [x['name'] for x in data['datatable']['columns']]

    df = df.infer_objects()
    
    return df



def translate_currency(df: pd.DataFrame, columns: Optional[List[str]] = None):
    '''
    Translate currency of columns to USD according course information
    in appropriate columns(like debtusd-debt)
    
    Parameters
    ----------
    df:
        quarterly-based data 
    columns:
        columns to translate currency
        
    Returns
    -------
    ``pd.DataFrame`` 
        result with the same columns and shapes but with 
        converted currency in columns
    '''
    if columns is None:
        no_translate_cols = ['ticker', 'dimension', 'calendardate', 'datekey',
                             'reportperiod', 'date', 'marketcap', 'lastupdated']
        no_translate_cols += [x for x in df.columns if 'usd' in x]
        columns = [x for x in df.columns if x not in no_translate_cols]
    
    df = df.infer_objects()
    usd_cols = ['equityusd','epsusd','revenueusd','netinccmnusd',
                'cashnequsd','debtusd','ebitusd','ebitdausd']
                
    usd_cols = list(set(df.columns).intersection(set(usd_cols)))
    assert len(usd_cols) > 0
                
    rows = np.array([(df[col.replace('usd', '')] / df[col]).values 
                     for col in usd_cols])
    df['trans_currency'] = np.nanmax(rows, axis=0).astype('float32')
    df['trans_currency'] = df['trans_currency'].interpolate()    
    for col in columns:
        df[col] = df[col] / df['trans_currency']
    
    return df



class SF1BaseData: 
    '''
    Load base information about company(like sector, industry etc)
    '''
    def __init__(self, data_path: Optional[str]=None):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.sf1` dataset folder
            If None, than will be used ``sf1_data_path``
            from `~/.ml_investment/config.json`
        '''
        if data_path is None:
            data_path = load_config()['sf1_data_path']
        self.data_path = data_path


    def load(self, index: Optional[List[str]]=None) -> pd.DataFrame:
        '''
        Parameters
        ----------
        index:
            list of ticker to load data for, i.e. ``['AAPL', 'TSLA']`` 
            OR ``None`` (loading for all possible tickers)
        Returns
        -------
        ``pd.DataFrame`` 
            base companies information
        '''
        path = '{}/tickers.zip'.format(self.data_path)
        tickers_df = pd.read_csv(path)
        tickers_df = tickers_df[tickers_df['table'] == 'SF1']
        if index is not None:
            tmp = pd.DataFrame()
            tmp['ticker'] = index
            tmp['flag'] = True
            tickers_df = pd.merge(tickers_df, tmp, on='ticker', how='left')
            tickers_df['flag'] = tickers_df['flag'].fillna(False)
            tickers_df = tickers_df[tickers_df['flag']]
            del tickers_df['flag']

        return tickers_df.reset_index(drop=True)

    def existing_index(self):
        '''  
        Returns
        -------
        ``List``
            existing index values that can pe pushed to `load`
        '''
        path = '{}/tickers.zip'.format(self.data_path)
        tickers_df = pd.read_csv(path)
        tickers_df = tickers_df[tickers_df['table'] == 'SF1']
        index = list(tickers_df['ticker'].unique())
        return index 


class SF1QuarterlyData: 
    '''
    Loader for quartely fundamental information about
    companies(debt, revenue etc)
    '''
    def __init__(self,
                 data_path: Optional[str]=None,
                 quarter_count: Optional[int]=None,
                 dimension: Optional[str]='ARQ'):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.sf1` dataset folder
            If None, than will be used ``sf1_data_path``
            from `~/.ml_investment/config.json`
        quarter_count:
            maximum number of last quarters to return. 
            Resulted number may be less due to short history in some companies
        dimension: 
            one of ``['MRY', 'MRT', 'MRQ', 'ARY', 'ART', 'ARQ']``.
            SF1 dataset-based parameter
        '''
        if data_path is None:
            data_path = load_config()['sf1_data_path']
        self.data_path = data_path
        self.quarter_count = quarter_count
        self.dimension = dimension


    def load(self, index: List[str]) -> pd.DataFrame:
        '''    
        Parameters
        ----------
        index:
            list of tickers to load data for, i.e. ``['AAPL', 'TSLA']``
           
        Returns
        -------
        ``pd.DataFrame``
            quarterly information about companies
        '''
        result = []
        for ticker in index:
            path = '{}/core_fundamental/{}.json'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            df = _load_df(path)
            df = df[df['dimension'] == self.dimension]
            if self.quarter_count is not None:
                df = df[:self.quarter_count]

            df['date'] = df['datekey']
            df = df.sort_values('date', ascending=False)
            #df = translate_currency(df)
            result.append(df)
        
        if len(result) == 0:
            return None

        result = pd.concat(result, axis=0).reset_index(drop=True)
        result = result.infer_objects()
        result['date'] = result['date'].astype(np.datetime64) 
     
        return result


    def existing_index(self):
        '''  
        Returns
        -------
        ``List``
            existing index values that can pe pushed to `load`
        '''
        dir_path = '{}/core_fundamental'.format(self.data_path)
        index = [x.split('.json')[0] for x in os.listdir(dir_path)]
        return index 


        
class SF1DailyData():
    '''
    Load daily information about company(marketcap, pe etc)
    '''
    def __init__(self, 
                 data_path: Optional[str]=None,
                 days_count: Optional[int]=None):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.sf1` dataset folder
            If None, than will be used ``sf1_data_path``
            from `~/.ml_investment/config.json`
        days_count:
            maximum number of last days to return. 
            Resulted number may be less due to short history in some companies
        '''
        if data_path is None:
            data_path = load_config()['sf1_data_path']
        if days_count is None:
            days_count = int(1e5)    

        self.data_path = data_path
        self.days_count = days_count


    def load(self, index: List[str]) -> pd.DataFrame:
        '''    
        Parameters
        ----------
        index:
            list of ticker to load data for, i.e. ``['AAPL', 'TSLA']`` 
             
        Returns
        -------
        ``pd.DataFrame``
            daily information about companies
        '''                        
        result = []
        for ticker in index:
            path = '{}/daily/{}.json'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            daily_df = _load_df(path)[:self.days_count]
            result.append(daily_df)
        
        if len(result) == 0:
            return None

        result = pd.concat(result, axis=0).reset_index(drop=True)
        result = result.infer_objects()
        result['marketcap'] = result['marketcap'].astype(float) * 1e6
        result['date'] = result['date'].astype(np.datetime64) 
                
        return result


    def existing_index(self):
        '''  
        Returns
        -------
        ``List``
            existing index values that can pe pushed to `load`
        '''
        dir_path = '{}/daily'.format(self.data_path)
        index = [x.split('.json')[0] for x in os.listdir(dir_path)]
        return index 


class SF1SNP500Data: 
    '''
    S&P500 historical constituents
    '''
    def __init__(self, data_path: Optional[str]=None):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.sf1` dataset folder
            If None, than will be used ``sf1_data_path``
            from `~/.ml_investment/config.json`
        '''
        if data_path is None:
            data_path = load_config()['sf1_data_path']
        self.data_path = data_path


    def load(self, index: Optional[List[np.datetime64]]=None) -> pd.DataFrame:
        '''
        Parameters
        ----------
        index:
            list of dates to load constituents for,
            i.e. ``[np.datetime64('2018-01-01'), np.datetime64('2018-05-10')]`` 
            If there are no such date, than nearest past date will be used.
            OR ``None`` (loading for all dates when constituents was changed)
        Returns
        -------
        ``pd.DataFrame`` 
            constituents information
        '''
        path = '{}/snp500.zip'.format(self.data_path)
        df = pd.read_csv(path)
        df['date'] = df['date'].astype(np.datetime64)
        df = df[df['action'] == 'historical']
        
        if index is None:
            return df

        
        result = []
        for date in index:
            tmp = df[(df['date'] <= date)]
            result.append(tmp[tmp['date'] == tmp['date'].max()])
        
        if len(result) == 0:
            return

        result = pd.concat(result, axis=0)
        
        return result
    

    def existing_index(self):
        '''  
        Returns
        -------
        ``List``
            existing index values that can pe pushed to `load`
        '''
        path = '{}/snp500.zip'.format(self.data_path)
        df = pd.read_csv(path)
        df['date'] = df['date'].astype(np.datetime64)
        df = df[df['action'] == 'historical']
        index = list(df['date'].unique())
        return index

