import argparse
import os
import numpy as np
import pandas_datareader.data as web
from tqdm import tqdm
from multiprocessing import Pool
from ml_investment.download import TinkoffDownloader
from ml_investment.utils import load_json, save_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    arg('--secrets_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
    secrets = load_json(args.secrets_path)  
    
    def foo(ticker):
        try:
            df = web.DataReader(ticker, "yahoo", np.datetime64('2017-01-01'), np.datetime64('now'))
            df.to_csv('{}/{}.csv'.format(config['daily_bars_data_path'], ticker))          
        except:
            print(ticker)

    tinkoff_downloader = TinkoffDownloader(secrets)
    tinkoff_tickers = [x['ticker'] for x in tinkoff_downloader.get_stocks()['payload']['instruments']]
    index_tickers = ['SPY', 'TLT', 'QQQ']
    os.makedirs(config['daily_bars_data_path'], exist_ok=True)
    
    p = Pool(6)
    for _ in tqdm(p.imap(foo, tinkoff_tickers[:0] + index_tickers)):
        None

    # for ticker in tqdm(tinkoff_tickers[:5]):
    #     df = web.DataReader(ticker, "yahoo", np.datetime64('2017-01-01'), np.datetime64('now'))
    #     df.to_csv('{}/{}.csv'.format(config['daily_bars_data_path'], ticker))          
    #
