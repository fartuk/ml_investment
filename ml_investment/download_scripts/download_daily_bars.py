import argparse
import os
import numpy as np
import pandas_datareader.data as web
from tqdm import tqdm
from multiprocessing import Pool
from ml_investment.download import TinkoffDownloader
from ml_investment.utils import load_config, load_tickers




def _single_ticker_download(ticker):
    config = load_config()
    try:
        df = web.DataReader(ticker, "yahoo", np.datetime64('2017-01-01'), np.datetime64('now'))
        df.to_csv('{}/{}.csv'.format(config['daily_bars_data_path'], ticker))          
    except:
        print(ticker)


def main():
    '''
    Download daily price bars for base US stocks and indexes. 
    Downloading path ``daily_bars_data_path`` may be configured at `~/.ml_investment/config.json`
    '''
    config = load_config()
    tickers = load_tickers()['base_us_stocks']
    index_tickers = ['SPY', 'TLT', 'QQQ']
    os.makedirs(config['daily_bars_data_path'], exist_ok=True)
    
    p = Pool(6)
    for _ in tqdm(p.imap(_single_ticker_download,
                         tickers + index_tickers)):
        None




if __name__ == '__main__':
    main()


