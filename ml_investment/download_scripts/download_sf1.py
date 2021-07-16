import argparse
from ml_investment.data_loaders.sf1 import SF1BaseData
from ml_investment.download import QuandlDownloader
from ml_investment.utils import load_config, save_json


def main(data_path :str=None):
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
    '''
    if data_path is None:
        config = load_config()
        data_path = config['sf1_data_path']

    downloader = QuandlDownloader(sleep_time=0.8)
    downloader.zip_download('datatables/SHARADAR/TICKERS?qopts.export=true',
                            '{}/tickers.zip'.format(data_path))

    base_df = SF1BaseData(data_path).load()
    tickers = base_df['ticker'].unique().tolist()
    
    downloader.ticker_download('datatables/SHARADAR/SF1?ticker={ticker}', tickers, 
                               save_dirpath='{}/core_fundamental'.format(data_path), 
                               skip_exists=False,  batch_size=2, n_jobs=4)

    downloader.ticker_download('datatables/SHARADAR/DAILY?ticker={ticker}', tickers, 
                               save_dirpath='{}/daily'.format(data_path), 
                               skip_exists=False, batch_size=2, n_jobs=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    args = parser.parse_args()
    main(args.data_path)
