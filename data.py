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


def load_quarterly_data_cf1(ticker, config, dimension='ARQ'):
    path = '{}/cf1/core_fundamental/{}.json'.format(config['data_path'], ticker)
    if not os.path.exists('{}/cf1/core_fundamental/{}.json'.format(config['data_path'], ticker)):
        return []
    df = load_cf1_df(path)
    df = df[df['dimension'] == dimension]
    df['date'] = df['datekey']
    data = json.loads(df.to_json(orient='records'))
    
    return data

    
def load_quarterly_data_fmpapi(ticker, config):
    list_of_datas = []
    for foldername in ['income_statement', 'cash_flow_statement', 'enterprise_values']:
        data = load_json('{}/fmpapi/{}/{}.json'.format(config['data_path'], foldername, ticker))
        list_of_datas.append(data)
        
    list_of_set_data = []
    for data in list_of_datas:
        list_of_set_data.append({x['date']:x for x in data})
            
    common_dates = set.intersection(*[set(x.keys()) for x in list_of_set_data]) 
    try:
        common_dates = sorted([np.datetime64(x) for x in common_dates])
    except:
        return []
        
    common_dates = [str(x) for x in common_dates][::-1]
    
    result = []
    for date in common_dates:
        quarter_set = {}
        for set_data in list_of_set_data:
            quarter_set.update(set_data[date])
            
        result.append(quarter_set)
        
    return result
    
    
    
    
    
    