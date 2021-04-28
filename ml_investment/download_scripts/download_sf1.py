import argparse
from ml_investment.data import SF1Data
from ml_investment.download import QuandlDownloader
from ml_investment.utils import load_config, save_json


def main():
    config = load_config()
    downloader = QuandlDownloader(sleep_time=0.8)
    downloader.zip_download('datatables/SHARADAR/TICKERS?qopts.export=true',
                            '{}/tickers.zip'.format(config['sf1_data_path']))

    data_loader = SF1Data(config['sf1_data_path'])
    ticker_list = data_loader.load_base_data()['ticker'].unique().tolist()
    
    downloader.ticker_download('datatables/SHARADAR/SF1?ticker={ticker}', ticker_list, 
                               save_dirpath='{}/core_fundamental'.format(config['sf1_data_path']), 
                               skip_exists=False,  batch_size=10, n_jobs=4)

    downloader.ticker_download('datatables/SHARADAR/DAILY?ticker={ticker}', ticker_list, 
                               save_dirpath='{}/daily'.format(config['sf1_data_path']), 
                               skip_exists=False, batch_size=5, n_jobs=4)



if __name__ == '__main__':
    main()
