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



def _single_ticker_download(ticker):
    global _data_path
    global _from_date
    global _to_date
    success = False
    for _ in range(3):
        try:
            df = web.DataReader(ticker, "yahoo", 
                                _from_date, _to_date)
            df.to_csv('{}/{}.csv'.format(_data_path, ticker))          
            time.sleep(np.random.uniform(0.2, 1.0))
            success = True
            break
        except:
            None
            
    if not success and _verbose:
        print('Can not download {}'.format(ticker))


def main(data_path: str=load_config()['daily_bars_data_path'], 
         tickers: Optional[List]=load_tickers()['base_us_stocks'] + \
                                 ['SPY', 'TLT', 'QQQ'],
         from_date: Optional[np.datetime64]=np.datetime64('2010-01-01'),
         to_date: Optional[np.datetime64]=np.datetime64('now'),
         verbose: bool=False):
    '''
    Download daily price bars for base US stocks and indexes. 

    Parameters
    ----------
    data_path:
        path to folder in which downloaded data will be stored.
        OR ``None`` (downloading path will be as ``daily_bars_data_path`` from 
        `~/.ml_investment/config.json`
    tickers:
        tickers to download daily bars for
    from_date:
        start date for loading data
    to_date:
        end day for loading data
    verbose:
        show progress or not
    '''
    # Due to tqdm not work with multiple parameters in Pool
    global _data_path
    _data_path = data_path

    global _from_date
    _from_date = from_date
    
    global _to_date
    _to_date = to_date

    global _verbose
    _verbose = verbose

    os.makedirs(data_path, exist_ok=True)
    
    print('Start daily bars downloading: {}'.format(
            str(np.datetime64(int(time.time() * 1000), 'ms'))))
    with Pool(4) as p:
        for _ in tqdm(p.imap(_single_ticker_download, tickers),
                      disable=not verbose):
            None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    arg('--verbose', type=bool)
    args = vars(parser.parse_args())
    args = {key:args[key] for key in args if args[key] is not None}  
    main(**args)


