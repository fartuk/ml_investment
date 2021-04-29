from ml_investment.download import YahooDownloader
from ml_investment.utils import load_tickers



def main():
    '''
    Download quarterly and base data from https://finance.yahoo.com
    Downloading path ``yahoo_data_path`` may be configured at ~/.ml_investment/config.json
    '''
    tickers = load_tickers()['base_us_stocks']
    downloader = YahooDownloader()
    downloader.download_quarterly_data(tickers)
    downloader.download_base_data(tickers)



if __name__ == '__main__':
    main()
 
