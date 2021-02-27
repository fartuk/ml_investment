import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_json



class SF1Data:
    def __init__(self, data_path):
        self.data_path = data_path


    def _load_df(self, json_path):
        data = load_json(json_path)
        df = pd.DataFrame(data['datatable']['data'])
        if len(df) == 0:
            columns = [x['name'] for x in data['datatable']['columns']]
            df = pd.DataFrame(columns=columns)
            df = df.infer_objects()
        else:
            df.columns = [x['name'] for x in data['datatable']['columns']]
 
        return df


    def load_quarterly_data(self, ticker_list, quarter_count=10, dimension='ARQ'):
        result = []
        for ticker in ticker_list:
            path = '{}/core_fundamental/{}.json'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            df = self._load_df(path)
            df = df[df['dimension'] == dimension][:quarter_count]
            df['date'] = df['datekey']
            data = json.loads(df.to_json(orient='records'))
            result.extend(data)
                  
        return result

    
    def load_tickers(self, currency=None,
                     scalemarketcap=None):
        path = '{}/tickers.csv'.format(self.data_path)
        tickers_df = pd.read_csv(path)
        if currency is not None:
            tickers_df = tickers_df[tickers_df['currency'] == currency]
        if type(scalemarketcap) == str:
            tickers_df = tickers_df[tickers_df['scalemarketcap'] == scalemarketcap]
        if type(scalemarketcap) == list:
            tickers_df = tickers_df[tickers_df['scalemarketcap'].apply(lambda x: 
                                    x in scalemarketcap)]
        return tickers_df


    def load_daily_data(self, ticker_list, back_days=None):
        if back_days is None:
            back_days = int(1e9)    
        result = []
        for ticker in tqdm(ticker_list):
            path = '{}/daily/{}.json'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            daily_df = self._load_df(path)[:back_days]
            result.append(daily_df)
            
        result = pd.concat(result, axis=0)
        
        return result


    def translate_currency(self, data, columns):
        df = pd.DataFrame(data)
        df = df.infer_objects()
        usd_cols = ['equityusd','epsusd','revenueusd','netinccmnusd',
                    'cashnequsd','debtusd','ebitusd','ebitdausd']
        rows = np.array([(df[col.replace('usd', '')] / df[col]).values 
                         for col in usd_cols])
        df['trans_currency'] = np.nanmax(rows, axis=0).astype('float32')
        df['trans_currency'] = df['trans_currency'].interpolate()    
        for col in columns:
            df[col] = df[col] * df['trans_currency']

        data = json.loads(df.to_json(orient='records'))
        
        return data



    
    
    
    
