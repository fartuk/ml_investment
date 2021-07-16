import argparse
import os
import numpy as np
import pandas_datareader.data as web
from tqdm import tqdm
from multiprocessing import Pool
from ml_investment.download import TinkoffDownloader
from ml_investment.utils import load_config, load_tickers


# Due to tqdm not work with multiple parameters in Pool
global _data_path
_data_path = None

def _single_ticker_download(ticker):
    global _data_path
    for _ in range(3):
        try:
            df = web.DataReader(ticker, "yahoo", 
                                np.datetime64('2010-01-01'), np.datetime64('now'))
            df.to_csv('{}/{}.csv'.format(_data_path, ticker))          
            break
        except:
            print(ticker)


def main(data_path: str=None):
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
        config = load_config()
        data_path = config['daily_bars_data_path']

    global _data_path
    _data_path = data_path
    tickers = load_tickers()['base_us_stocks']
    index_tickers = ['SPY', 'TLT', 'QQQ']
    os.makedirs(data_path, exist_ok=True)
    
    p = Pool(8)
    for _ in tqdm(p.imap(_single_ticker_download,
                         tickers + index_tickers)):
        None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    args = parser.parse_args()
    main(args.data_path)


