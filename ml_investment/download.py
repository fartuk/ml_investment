import os
import requests
import time
import numpy as np
import pandas as pd
import copy
import json
from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool
from itertools import repeat
from .utils import load_config, load_secrets, load_json, save_json



class QuandlDownloader:
    def __init__(self, retry_cnt=10, sleep_time=1.4):
        self.secrets = load_secrets()
        self.retry_cnt = retry_cnt
        self.sleep_time = sleep_time
        self._save_dirpath = None
        self._base_url_route = None
                

    def _form_quandl_url(self, route):
        url = "https://www.quandl.com/api/v3/{}&api_key={}".format(
                route, 
                self.secrets['quandl_api_key'])
                
        return url 


    def _batch_ticker_download(self, ticker_list):
        time.sleep(np.random.uniform(0, self.sleep_time))
        url = self._base_url_route.format(ticker=','.join(ticker_list))
        url = self._form_quandl_url(url)
        for _ in range(10):
            try:
                response = requests.get(url)
                break
            except:
                print(11)
                time.sleep(np.random.uniform(0, self.sleep_time))

        if response.status_code != 200:
            return
        data = response.json()
        datatable_data = np.array(data['datatable']['data'])
        ticker_seq = np.array([x[0] for x in data['datatable']['data']])
        
        curr_data = copy.deepcopy(data)
        curr_data['datatable']['data'] = []

        for ticker in ticker_list:
            curr_datatable_data = datatable_data[ticker_seq == ticker].tolist()
            curr_data['datatable']['data'] = curr_datatable_data

            save_filepath = '{}/{}.json'.format(self._save_dirpath, ticker)
            save_json(save_filepath, curr_data)            

            
    def ticker_download(self, base_url_route, ticker_list, save_dirpath,
                 skip_exists=False, batch_size=5, n_jobs=12):
        self._save_dirpath = save_dirpath
        self._base_url_route = base_url_route
        os.makedirs(save_dirpath, exist_ok=True)
        if skip_exists:
            exist_tickers = [x.split('.')[0] for x in os.listdir(save_dirpath)]
            ticker_list = list(set(ticker_list).difference(set(exist_tickers)))
        
        batches = [ticker_list[k:k+batch_size] 
                        for k in range(0, len(ticker_list), batch_size)]
        with Pool(n_jobs) as p:
            for _ in tqdm(p.imap(self._batch_ticker_download, batches)):
#         for batch in tqdm(batches):
#             self._batch_ticker_download(batch)
                None
            

    def single_download(self, base_url_route, save_filepath):
        if '?' not in base_url_route:
            base_url_route = base_url_route + '?'
        url = self._form_quandl_url(base_url_route)
        for _ in range(10):
            try:
                response = requests.get(url)
                break
            except:
                print(11)
                time.sleep(np.random.uniform(0, 2))    
        if response.status_code != 200:
            print(12)
        data = response.json()
        save_json(save_filepath, data)
    
    
    def zip_download(self, base_url_route, save_filepath):
        url = self._form_quandl_url(base_url_route)
        info_response = requests.get(url)
        zip_link = info_response.json()['datatable_bulk_download']['file']['link']
        data_response = requests.get(zip_link)
        
        if '/' in save_filepath:
            folder_path = '/'.join(save_filepath.split('/')[:-1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
        with open(save_filepath, 'wb') as f:
            f.write(data_response.content)


            
class YahooDownloaderV1:
    DEFAULT_TYPE_LIST = [
        'quarterlyTotalCapitalization',
        'quarterlyTotalRevenue',
        'quarterlyNetIncome',
        'quarterlyFreeCashFlow',
        'quarterlyTotalAssets',
        'quarterlyEBITDA',
        'quarterlyNetDebt',
        'quarterlyGrossProfit',
        'quarterlyWorkingCapital',
        'quarterlyCashAndCashEquivalents',
        'quarterlyResearchAndDevelopment',
        'quarterlyCashDividendsPaid',
    ]   
    def __init__(self):
        self.secrets = load_secrets()
        
    def _parse_quarterly_json(self, json_data):
        json_data = json_data['timeseries']['result']
        dfs = []
        for data in json_data:
            name_set = set(data.keys()).intersection(set(self.type_list))
            if len(name_set) == 1:
                name = list(name_set)[0]
                new_data = [{'date': row['asOfDate'], name: row['reportedValue']['raw']}
                            for row in data[name]]
                dfs.append(pd.DataFrame(new_data))
        if len(dfs) == 0:
            return
        result = reduce(lambda l, r: pd.merge(l, r, on='date', how='left'), dfs)
        for key in set(self.type_list).difference(set(result.columns)):
            result[key] = None
                    
        return result

    def _parse_base_json(self, json_data):
        new_row = {}
        for key in json_data.keys():
            if type(json_data[key]) == dict and 'raw' in json_data[key]:
                new_row[key] = json_data[key]['raw']
                continue
            if type(json_data[key]) in [list, dict] and len(json_data[key]) == 0:
                new_row[key] = None
                continue
            new_row[key] = json_data[key]

        return new_row


    def _download_quarterly_data_single(self, ticker):
        try:
            base_url = 'https://query{query_id}.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries'
            base_url += '/{ticker}?lang=en-US&region=US&padTimeSeries=false&type={type_str}'
            base_url += '&merge=false&period1=493590046&period2={period2}&corsDomain=finance.yahoo.com'
            url = base_url.format(query_id=2,
                                  ticker=ticker, 
                                  type_str=','.join(self.type_list),
                                  period2=int(time.time()))
            
            r = requests.get(url)
            if r.status_code != 200:
                print(r.status_code, ticker)
                print(url)
                return
            json_data = r.json()
            
            quarterly_df = self._parse_quarterly_json(json_data)
            if quarterly_df is None:
                print('Empty', ticker)
                return
            quarterly_df['date'] = quarterly_df['date'].astype(np.datetime64)
            quarterly_df = quarterly_df.sort_values('date', ascending=False)
            
            filepath = '{}/quarterly/{}.csv'.format(self._data_path, ticker)
            quarterly_df.to_csv(filepath, index=False)

            time.sleep(np.random.uniform(0.1, 0.5))

        except:
            print('AAAA')
            time.sleep(np.random.uniform(0.1, 1))

    
    def _download_base_data_single(self, ticker):
        try:
            url = 'https://query{query_id}.finance.yahoo.com/v10/finance/quoteSummary/{ticker}'
            url += '?modules=summaryProfile,defaultKeyStatistics&corsDomain=finance.yahoo.com'

            r = requests.get(url.format(query_id=2,
                                        ticker=ticker))
            if r.status_code != 200:
                print(r.status_code, ticker)
                return
            json_data = r.json()['quoteSummary']['result'][0]

            base_data = {}
            b1 = self._parse_base_json(json_data['summaryProfile'])
            b2 = self._parse_base_json(json_data['defaultKeyStatistics'])
            base_data.update(b1)
            base_data.update(b2)   
            filepath = '{}/base/{}.json'.format(self._data_path, ticker)
            save_json(filepath, base_data)            
            
            time.sleep(np.random.uniform(0.1, 0.5))
        except:
            time.sleep(np.random.uniform(0.1, 1))


    def download_quarterly_data(self, data_path, tickers, type_list=DEFAULT_TYPE_LIST, n_jobs=2):
        self._data_path = data_path
        self.type_list = type_list
        os.makedirs('{}/quarterly'.format(data_path), exist_ok=True)
        for t in tqdm(tickers):
            self._download_quarterly_data_single(t)

        # with Pool(n_jobs) as p:
        #     for _ in tqdm(p.imap(self._download_quarterly_data_single, tickers)):
        #         None        
        #
    def download_base_data(self, data_path, tickers, n_jobs=4):
        self._data_path = data_path
        os.makedirs('{}/base'.format(data_path), exist_ok=True)
        with Pool(n_jobs) as p:
            for _ in tqdm(p.imap(self._download_base_data_single, tickers)):
                None        
        

class YahooDownloader:
    def __init__(self):
        self.secrets = load_secrets()
        self.config = load_config()

    def _parse_quarterly_json(self, json_data):
        new_data = []
        for row in json_data:
            new_row = {}
            for key in row.keys():
                if key == 'endDate':
                    new_row['date'] = row[key]['fmt']
                    continue
                if type(row[key]) == dict and 'raw' in row[key]:
                    new_row[key] = row[key]['raw']
                    continue
                if type(row[key]) == dict and len(row[key]) == 0:
                    new_row[key] = None
                    continue
                new_row[key] = row[key]
            new_data.append(new_row)
        df = pd.DataFrame(new_data)
        return df
    
    def _parse_base_json(self, json_data):
        new_row = {}
        for key in json_data.keys():
            if type(json_data[key]) == dict and 'raw' in json_data[key]:
                new_row[key] = json_data[key]['raw']
                continue
            if type(json_data[key]) in [list, dict] and len(json_data[key]) == 0:
                new_row[key] = None
                continue
            new_row[key] = json_data[key]

        return new_row


    def _download_quarterly_data_single(self, ticker):
        try:
            url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}'
            url += '?modules=incomeStatementHistoryQuarterly'
            url += ',balanceSheetHistoryQuarterly'
            url += ',cashflowStatementHistoryQuarterly'

            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

            r = requests.get(url.format(ticker=ticker), headers=headers)
            if r.status_code != 200:
                print(r.status_code, ticker)
                return
            try:
                json_data = r.json()['quoteSummary']['result'][0]
            except:
                return
            if len(json_data['incomeStatementHistoryQuarterly']['incomeStatementHistory']) == 0:
                return
            
            q1 = self._parse_quarterly_json(json_data['incomeStatementHistoryQuarterly']['incomeStatementHistory'])
            q2 = self._parse_quarterly_json(json_data['balanceSheetHistoryQuarterly']['balanceSheetStatements'])
            q3 = self._parse_quarterly_json(json_data['cashflowStatementHistoryQuarterly']['cashflowStatements'])

            quarterly_df = pd.merge(q1, q2, on='date', how='left', suffixes=('', '_y'))
            quarterly_df = pd.merge(quarterly_df, q3, on='date', how='left', suffixes=('', '_z'))
            
            filepath = '{}/quarterly/{}.csv'.format(self._data_path, ticker)
            quarterly_df.to_csv(filepath, index=False)

            time.sleep(np.random.uniform(0, 0.5))
        except:
            time.sleep(np.random.uniform(0, 0.5))


        
    def _download_base_data_single(self, ticker):
        try:
            url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}'
            url += '?modules=summaryProfile,defaultKeyStatistics'

            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

            r = requests.get(url.format(ticker=ticker), headers=headers)
            if r.status_code != 200:
                print(r.status_code, ticker)
                return
            try:
                json_data = r.json()['quoteSummary']['result'][0]
            except:
                return
            base_data = {}
            b1 = self._parse_base_json(json_data['summaryProfile'])
            b2 = self._parse_base_json(json_data['defaultKeyStatistics'])
            base_data.update(b1)
            base_data.update(b2)   
            filepath = '{}/base/{}.json'.format(self._data_path, ticker)
            save_json(filepath, base_data)            
            
            time.sleep(np.random.uniform(0, 0.5))
        except:
            time.sleep(np.random.uniform(0, 0.5))
            

    def download_quarterly_data(self, data_path, tickers, n_jobs=8):
        self._data_path = data_path
        os.makedirs('{}/quarterly'.format(self._data_path), exist_ok=True)
        with Pool(n_jobs) as p:
            for _ in tqdm(p.imap(self._download_quarterly_data_single, tickers)):
                None        
        
    def download_base_data(self, data_path, tickers, n_jobs=8):
        self._data_path = data_path
        os.makedirs('{}/base'.format(self._data_path), exist_ok=True)
        with Pool(n_jobs) as p:
            for _ in tqdm(p.imap(self._download_base_data_single, tickers)):
                None       


            
class TinkoffDownloader:
    def __init__(self):
        self.config = load_config()
        self.secrets = load_secrets()
        self.headers = {"Authorization": 
                        "Bearer {}".format(self.secrets['tinkoff_token'])}
        
    def get_stocks(self):
        url = 'https://api-invest.tinkoff.ru/openapi/market/stocks'
        response = requests.get(url, headers=self.headers)
        result = response.json()        
        
        return result
        
        
    def get_portfolio(self):
        url = 'https://api-invest.tinkoff.ru/openapi/portfolio' \
               '?brokerAccountId={}'
        url = url.format(self.secrets['tinkoff_broker_account_id'])
        response = requests.get(url, headers=self.headers)
        portfolio = response.json()
        
        return portfolio['payload']['positions']
        
        
    def get_figi_by_ticker(self, ticker):
        url = 'https://api-invest.tinkoff.ru/' \
              'openapi/market/search/by-ticker?ticker={}'.format(ticker)
        response = requests.get(url, headers=self.headers)
        figi = response.json()['payload']['instruments'][0]['figi']            
    
        return figi
    
    def get_lot_by_ticker(self, ticker):
        url = 'https://api-invest.tinkoff.ru/' \
              'openapi/market/search/by-ticker?ticker={}'.format(ticker)
        response = requests.get(url, headers=self.headers)
        lot = response.json()['payload']['instruments'][0]['lot']            
    
        return lot
        
    def get_last_price(self, ticker):
        figi = self.get_figi_by_ticker(ticker)      
        url = 'https://api-invest.tinkoff.ru/openapi/market/candles' \
              '?figi={}&from={}&to={}&interval=day'

        end = np.datetime64('now')
        end = str(end) + '%2B00%3A00'
        start = np.datetime64('now') - np.timedelta64(3, 'D')
        start = str(start) + '%2B00%3A00'
        
        url = url.format(figi, start, end)

        response = requests.get(url, headers=self.headers)
        close_price = response.json()['payload']['candles'][-1]['c']
        
        return close_price
    
    def get_price_history(self, ticker):
        figi = self.get_figi_by_ticker(ticker)      
        url = 'https://api-invest.tinkoff.ru/openapi/market/candles' \
              '?figi={}&from={}&to={}&interval=day'

        end = np.datetime64('now')
        end = str(end) + '%2B00%3A00'
        start = np.datetime64('now') - np.timedelta64(365*4, 'D')
        start = str(start) + '%2B00%3A00'
        
        url = url.format(figi, start, end)

        response = requests.get(url, headers=self.headers)
        return response
        close_price = response.json()['payload']['candles'][-1]['c']
        
        return close_price            
        
    def post_market_order(self, ticker, side, lots):
        figi = self.get_figi_by_ticker(ticker)
        url = 'https://api-invest.tinkoff.ru/openapi/orders/market-order?figi={}&brokerAccountId={}'
        url = url.format(figi, self.secrets['tinkoff_broker_account_id'])
        data = {
                "operation": side,
                "lots": lots,
               }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        
        return response
