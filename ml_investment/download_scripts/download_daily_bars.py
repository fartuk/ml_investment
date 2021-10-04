import argparse
import os
import time
import numpy as np
import pandas_datareader.data as web

from typing import List, Optional
from tqdm import tqdm
from multiprocessing import Pool
from ml_investment.download import TinkoffDownloader
from ml_investment.utils import load_config, load_tickers


# Due to tqdm not work with multiple parameters in Pool
global _data_path
_data_path = None

global _from_date
_from_date = None
    
global _to_date
_to_date = None

def _single_ticker_download(ticker):
    global _data_path
    global _from_date
    global _to_date
    for _ in range(3):
        try:
            df = web.DataReader(ticker, "yahoo", 
                                _from_date, _to_date)
            df.to_csv('{}/{}.csv'.format(_data_path, ticker))          
            time.sleep(np.random.uniform(0.2, 1.0))
            break
        except:
            None


def main(data_path: str=None, 
         tickers: Optional[List]=None,
         from_date: Optional[np.datetime64]=None,
         to_date: Optional[np.datetime64]=None):
    '''
    Download daily price bars for base US stocks and indexes. 

    Parameters
    ----------
    data_path:
        path to folder in which downloaded data will be stored.
        OR ``None`` (downloading path will be as ``daily_bars_data_path`` from 
        `~/.ml_investment/config.json`
    '''
    if data_path is None:
        data_path = load_config()['daily_bars_data_path']
    
    if tickers is None:
        tickers = load_tickers()['base_us_stocks'] + ['SPY', 'TLT', 'QQQ']
    
    if from_date is None:
        from_date = np.datetime64('2010-01-01')

    if to_date is None:
        to_date = np.datetime64('now')

    global _data_path
    _data_path = data_path
    os.makedirs(data_path, exist_ok=True)

    global _from_date
    _from_date = from_date
    
    global _to_date
    _to_date = to_date

    with Pool(4) as p:
        for _ in tqdm(p.imap(_single_ticker_download, tickers)):
            None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    args = parser.parse_args()
    main(args.data_path)


