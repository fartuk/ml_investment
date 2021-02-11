import os
import requests
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
from utils import load_json, save_json


def download_json(url, save_filepath=None):
    response = requests.get(url)
    data = response.json()
    if save_filepath is not None:
        save_json(save_filepath, data)
        
    return data, response


def form_quandl_url(route, config):
    url = "{}/{}&api_key={}".format(config['quandl_api_url'], route, config['quandl_api_key'])
    return url 


def single_ticker_download(base_url_route, ticker, save_dirpath, retry_cnt=100, sleep_time=0.5):
    for _ in range(retry_cnt):
        url = base_url_route.format(ticker=ticker)
        save_filepath = '{}/{}.json'.format(save_dirpath, ticker)
        data, response = download_json(url, save_filepath)
        if response.status_code == 200:
            break
        else:
            time.sleep(sleep_time)

            
def multi_ticker_download(base_url_route, ticker_list, save_dirpath, skip_exists=False, n_jobs=12):
    os.makedirs(save_dirpath, exist_ok=True)
    if skip_exists:
        exist_tickers = [x.split('.')[0] for x in os.listdir(save_dirpath)]
        ticker_list = list(set(ticker_list).difference(set(exist_tickers)))

    p = Pool(n_jobs)
    params_gen = zip(repeat(base_url_route), ticker_list, repeat(save_dirpath))
    for _ in tqdm(p.starmap(single_ticker_download, params_gen)):
        None
        

def get_tinkoff_portfolio(config):
    headers = {"Authorization": "Bearer {}".format(config['tinkoff_token'])}
    url = 'https://api-invest.tinkoff.ru/openapi/portfolio?brokerAccountId=2001883988'
    response = requests.get(url, headers=headers)
    tinkoff_portfolio = response.json()
    
    return tinkoff_portfolio['payload']['positions']


def get_tinkoff_price(ticker, config):
    headers = {"Authorization": "Bearer {}".format(config['tinkoff_token'])}
    url = 'https://api-invest.tinkoff.ru/openapi/market/search/by-ticker?ticker={}'.format(ticker)
    response = requests.get(url, headers=headers)
    figi = response.json()['payload']['instruments'][0]['figi']
    
    price_url = 'https://api-invest.tinkoff.ru/openapi/market/candles?figi={}&from={}&to={}&interval=day'

    end = np.datetime64('now')
    start = np.datetime64('now') - np.timedelta64(7, 'D')

    price_url = price_url.format(figi, str(start) + '%2B00%3A00', str(end) + '%2B00%3A00')

    response = requests.get(price_url, headers=headers)
    val = response.json()['payload']['candles'][-1]['c']
    
    return val

