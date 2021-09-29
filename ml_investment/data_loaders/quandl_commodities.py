'''
Loader for commodities price information from 
https://blog.quandl.com/api-for-commodity-data.
Data may be downloaded by script
:func:`~ml_investment.download_scripts.download_commodities.main`

Expected dataset structure
        | commodities
        | ├── LBMA_GOLD.json
        | ├── CHRIS_CME_CL1.json
        | └── ...       
'''
 
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from ..utils import load_json, load_config



class QuandlCommoditiesData:
    '''
    Loader for commodities price information. 
    '''
    def __init__(self, data_path: Optional[str]=None):
        '''
        data_path:
            path to :mod:`~ml_investment.data_loaders.quandl_commodities`
            dataset folder
            If None, than will be used ``commodities_data_path``
            from `~/.ml_investment/config.json`
        '''
        if data_path is None:
            data_path = load_config()['commodities_data_path']
        self.data_path = data_path

        
    def load(self, index: List[str]) -> pd.DataFrame:
        '''
        Load time-series information about commodity price
        
        Parameters
        ----------
        index:
            list of commodities codes to load data for, i.e. 
            ``['LBMA/GOLD', 'JOHNMATT/PALL']``
            
        Returns
        -------
        ``pd.DataFrame``
            time series price information
        '''  
        result = []
        for code in index:
            path = '{}/{}.json'.format(self.data_path, code.replace('/', '_'))
            if not os.path.exists(path):
                continue
            data = load_json(path)
            data = np.array(data['dataset']['data'])
            df = pd.DataFrame()
            df['date'] = data[:, 0].astype(np.datetime64)
            df['price'] = data[:, 1].astype('float')
            df['date'] = df['date'].astype(np.datetime64)
            df['commodity_code'] = code
            result.append(df)
        
        if len(result) == 0:
            return None

        result = pd.concat(result, axis=0)
        
        return result    
    

