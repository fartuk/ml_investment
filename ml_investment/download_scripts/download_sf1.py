import argparse
import time
import numpy as np
from ml_investment.data_loaders.sf1 import SF1BaseData
from ml_investment.download import QuandlDownloader
from ml_investment.utils import load_config, save_json


def main(data_path: str=load_config()['sf1_data_path'],
         verbose: bool=False):
    '''
    Download quarterly fundamental data from
    https://www.quandl.com/databases/SF1/data

    Note:
        SF1 is paid, so you need to subscribe 
        and paste quandl token to `~/.ml_investment/secrets.json`
        ``quandl_api_key``

    Parameters
    ----------
    data_path:
        path to folder in which downloaded data will be stored.
        OR ``None`` (downloading path will be as ``sf1_data_path`` from 
        `~/.ml_investment/config.json`
    verbose:
        show progress or not
    '''
    downloader = QuandlDownloader(sleep_time=0.8)


    print('Start SF1 base downloading: {}'.format(
            str(np.datetime64(int(time.time() * 1000), 'ms'))))
    downloader.zip_download(
            base_url_route='datatables/SHARADAR/TICKERS?qopts.export=true',
            save_filepath='{}/tickers.zip'.format(data_path))


    print('Start SF1 snp500 downloading: {}'.format(
            str(np.datetime64(int(time.time() * 1000), 'ms'))))
    downloader.zip_download(
            base_url_route='datatables/SHARADAR/SP500?qopts.export=true',                    
            save_filepath='{}/snp500.zip'.format(data_path))
    

    base_df = SF1BaseData(data_path).load()
    tickers = base_df['ticker'].unique().tolist()
    

    print('Start SF1 quarterly downloading: {}'.format(
            str(np.datetime64(int(time.time() * 1000), 'ms'))))
    downloader.ticker_download(
            base_url_route='datatables/SHARADAR/SF1?ticker={ticker}',
            tickers=tickers, 
            save_dirpath='{}/core_fundamental'.format(data_path), 
            skip_exists=False,
            batch_size=2,
            n_jobs=4,
            verbose=verbose)


    print('Start SF1 daily downloading: {}'.format(
            str(np.datetime64(int(time.time() * 1000), 'ms'))))
    downloader.ticker_download(
            base_url_route='datatables/SHARADAR/DAILY?ticker={ticker}',
            tickers=tickers,
            save_dirpath='{}/daily'.format(data_path), 
            skip_exists=False,
            batch_size=2,
            n_jobs=4,
            verbose=verbose)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    arg('--verbose', type=bool)
    args = vars(parser.parse_args())
    args = {key:args[key] for key in args if args[key] is not None}  
    main(**args)

