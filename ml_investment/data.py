import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from .utils import load_json



class SF1Data: 
    '''
    Loading data provided by https://www.quandl.com/databases/SF1
    '''
    def __init__(self, data_path: str):
        '''
        Parameters
        ----------
        data_path:
            path to SF1 data folder with structure
            SF1
            ├── core_fundamental
            │   ├── AAPL.json
            │   ├── FB.json
            │   └── ...
            ├── daily
            │   ├── AAPL.json
            │   ├── FB.json
            │   └── ...
            └── tickers.csv       
        '''
        self.data_path = data_path


    def _load_df(self, json_path: str) -> pd.DataFrame:
        '''
        Load data from SF1 .json file and convert it to pd.DataFrame
        
        Parameters
        ----------
        json_path:
            path to SF1 .json path
            
        Returns
        -------
            pd.DataFrame content of file
        '''
        data = load_json(json_path)
        df = pd.DataFrame(data['datatable']['data'])
        if len(df) == 0:
            columns = [x['name'] for x in data['datatable']['columns']]
            df = pd.DataFrame(columns=columns)
        else:
            df.columns = [x['name'] for x in data['datatable']['columns']]

        df = df.infer_objects()
 
        return df
        
        
    def load_base_data(self, 
                       currency: Optional[str]=None,
                       scalemarketcap: Optional[Union[str, 
                                                      List[str]]
                                                ]=None) -> pd.DataFrame:
        '''
        Load base information about company(like sector, industry etc)
        
        Parameters
        ----------
        currency:
            currency of returned companies  
        scalemarketcap: possible values ['1 - Nano', '2 - Micro', '3 - Small',
                                         '4 - Mid', '5 - Large', '6 - Mega']
            scalemarketcap of returned companies
            
        Returns
        -------
            pd.DataFrame with base information
        '''
        path = '{}/tickers.csv'.format(self.data_path)
        tickers_df = pd.read_csv(path)
        if currency is not None:
            tickers_df = tickers_df[tickers_df['currency'] == currency]
            
        if type(scalemarketcap) == str:
            tickers_df = tickers_df[tickers_df['scalemarketcap'] 
                                    == scalemarketcap]
        if type(scalemarketcap) == list:
            tickers_df = tickers_df[tickers_df['scalemarketcap'].apply(lambda x: 
                                    x in scalemarketcap)]
        return tickers_df.reset_index(drop=True)
        
        
    def load_quarterly_data(self, 
                            tickers: List[str], 
                            quarter_count: Optional[int]=None,
                            dimension: Optional[str]='ARQ') -> pd.DataFrame:
        '''
        Load quartely fundamental information about companies(debt, revenue etc)
        
        Parameters
        ----------
        tickers:
            tickers of returned companies
        quarter_count:
            maximum last quarter to return  
        dimension: one of ['MRY', 'MRT', 'MRQ', 'ARY', 'ART', 'ARQ']
            SF1 dataset-based parameter
            
        Returns
        -------
            pd.DataFrame with quarterly information
        '''
        result = []
        for ticker in tickers:
            path = '{}/core_fundamental/{}.json'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            df = self._load_df(path)
            df = df[df['dimension'] == dimension]
            if quarter_count is not None:
                df = df[:quarter_count]
            df['date'] = df['datekey']
            result.append(df)
           
        
        try:
            result = pd.concat(result, axis=0).reset_index(drop=True)
        except:
            print(tickers)
        result['date'] = result['date'].astype(np.datetime64) 
         
        return result


    def load_daily_data(self, tickers: List[str], 
                        back_days:Optional[int]=None) -> pd.DataFrame:
        '''
        Load daily information about company(marketcap, pe etc)
        
        Parameters
        ----------
        tickers:
            tickers of returned companies 
        back_days:
            SF1 dataset-based parameter
            
        Returns
        -------
            pd.DataFrame with daily information
        '''                        
        if back_days is None:
            back_days = int(1e5)    
        result = []
        for ticker in tickers:
            path = '{}/daily/{}.json'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            daily_df = self._load_df(path)[:back_days]
            result.append(daily_df)
            
        result = pd.concat(result, axis=0).reset_index(drop=True)
        result = result.infer_objects()
        result['marketcap'] = result['marketcap'].astype(float) * 1e6
        result['date'] = result['date'].astype(np.datetime64) 
                
        return result


    @classmethod
    def translate_currency(cls, df:pd.DataFrame, columns:List[str]):
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
            pd.DataFrame with the same columns and shapes but with converted 
            currency in columns
        '''  
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


class ComboData:
    '''
    Class to combine data loaders in single object with union of all 
    methods.
    '''
    def __init__(self, data_loaders_list: List):
        '''
        Parameters
        ----------
        data_loaders_list:
            list of classes implementing different data loading interfaces. 
            If there are several the same names, then method belongs
            to the earlier class in data_loaders_list will be used
        '''
        self.data_loaders_list = data_loaders_list
        for data_loader in self.data_loaders_list[::-1]:
            for f_name in dir(data_loader):
                if callable(getattr(data_loader, f_name)) and \
                        not f_name.startswith("_"):
                    setattr(self, f_name, getattr(data_loader, f_name))
    
    
    
