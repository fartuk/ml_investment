import json
import os
import numpy as np
import pandas as pd
from utils import load_json



def load_cf1_df(json_path):
    data = load_json(json_path)
    df = pd.DataFrame(data['datatable']['data'])
    if len(df) == 0:
        df = pd.DataFrame(columns=[x['name'] for x in data['datatable']['columns']])
        df = df.infer_objects()
    else:
        df.columns = [x['name'] for x in data['datatable']['columns']]
    
    return df


def load_quarterly_data_cf1(ticker, data_path, dimension='ARQ'):
    path = '{}/cf1/core_fundamental/{}.json'.format(data_path, ticker)
    if not os.path.exists(path):
        return []
    df = load_cf1_df(path)
    df = df[df['dimension'] == dimension]
    df['date'] = df['datekey']
    data = json.loads(df.to_json(orient='records'))
    
    return data

  
def translate_currency_cf1(data, columns):
    df = pd.DataFrame(data)
    df = df.infer_objects()
    usd_cols = ['equityusd','epsusd','revenueusd','netinccmnusd','cashnequsd','debtusd','ebitusd','ebitdausd']
    #columns = df.columns[df.dtypes == 'float64']
    rows = [(df[col.replace('usd', '')] / df[col]).values for col in usd_cols]
    df['trans_currency'] = np.nanmax(np.array(rows), axis=0).astype('float32')
    df['trans_currency'] = df['trans_currency'].interpolate()    
    for col in columns:
        df[col] = df[col] * df['trans_currency']
        
    data = json.loads(df.to_json(orient='records'))
    
    return data
    

    
    
    
    
    
