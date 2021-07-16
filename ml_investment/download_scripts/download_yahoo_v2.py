import argparse
import os
import time
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from ml_investment.download import YahooDownloader
from ml_investment.utils import load_config, load_tickers, save_json


import yfinance as yf
# Due to tqdm not work with multiple parameters in Pool
global _data_path
_data_path = None

def _single_ticker_download(ticker):
    try:
        ticker_data = yf.Ticker(ticker)
        quarterly_df = ticker_data.quarterly_financials.T
        quarterly_df['date'] = quarterly_df.index
        quarterly_df.to_csv('{}/quarterly/{}.csv'.format(_data_path, ticker))

        save_json('{}/base/{}.json'.format(_data_path, ticker),
                  ticker_data.info)

        time.sleep(np.random.uniform(0.1, 0.5))
    except:
        None


def main(data_path:str=None):
    '''
    Download quarterly and base data from https://finance.yahoo.com

    Parameters
    ----------
    data_path:
        path to folder in which downloaded data will be stored.
        OR ``None`` (downloading path will be as ``yahoo_data_path`` from 
        `~/.ml_investment/config.json`
    '''
    if data_path is None:
        config = load_config()
        data_path = config['yahoo_data_path']

    global _data_path
    _data_path = data_path
    tickers = load_tickers()['base_us_stocks']
    os.makedirs('{}/quarterly'.format(data_path), exist_ok=True)
    os.makedirs('{}/base'.format(data_path), exist_ok=True)

    p = Pool(12)
    for _ in tqdm(p.imap(_single_ticker_download, tickers)):
        None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    args = parser.parse_args()
    main(args.data_path)
 
