'''
Loader for daily bars price information. 
Data may be downloaded by script
:func:`~ml_investment.download_scripts.download_daily_bars.main`

Expected dataset structure
        | daily_bars
        | ├── AAPL.csv
        | ├── TSLA.csv
        | └── ...       
'''

import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from ..utils import load_json, load_config


class DailyBarsData:
    '''
    Loader for daywise price bars.
    '''
    def __init__(self,
                 data_path: Optional[str]=None,
                 days_count: Optional[int]=None):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.daily_bars`
            dataset folder
            If None, than will be used ``daily_bars_data_path``
            from `~/.ml_investment/config.json`
        days_count:
            maximum number of last days to return. 
            Resulted number may be less due to short history in some companies
        '''
        if data_path is None:
            data_path = load_config()['daily_bars_data_path']

        if days_count is None:
            days_count = int(1e5)

        self.data_path = data_path
        self.days_count = days_count


    def load(self, index: List[str]) -> pd.DataFrame:
        '''
        Load daily price bars
        
        Parameters
        ----------
        index:
            list of tickers to load data for, i.e. ``['AAPL', 'TSLA']``
            
        Returns
        -------
        ``pd.DataFrame``
            daily price bars
        '''                        
        result = []
        for ticker in index:
            path = '{}/{}.csv'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            daily_df = pd.read_csv(path)[::-1][:self.days_count]
            daily_df['ticker'] = ticker
            result.append(daily_df)
        if len(result) > 0:    
            result = pd.concat(result, axis=0).reset_index(drop=True)
            result = result.infer_objects()
            result['date'] = result['Date'].astype(np.datetime64) 
        else:
            return None

        return result

