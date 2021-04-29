import json
import os
import hashlib
import pandas as pd
import numpy as np
from copy import deepcopy


def check_create_folder(file_path):
    if '/' in file_path:
        folder_path = '/'.join(file_path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def save_json(file_path, data):
    check_create_folder(file_path)
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        
        
def load_json(path):
    with open(path, "r") as read_file:
        in_data = json.load(read_file)
        
    return in_data


def copy_repeat(data, cnt: int):
    result = [deepcopy(data) for _ in range(cnt)]
    
    return result

def int_hash_of_str(text:str):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)



def load_config():
    _base_dir = os.path.expanduser('~')
    _ml_investments_dir = os.path.join(_base_dir, '.ml_investment')
    _config_path = os.path.join(_ml_investments_dir, 'config.json')
    config = load_json(_config_path)
    return config


def load_secrets():
    _base_dir = os.path.expanduser('~')
    _ml_investments_dir = os.path.join(_base_dir, '.ml_investment')
    _secrets_path = os.path.join(_ml_investments_dir, 'secrets.json')
    secrets = load_json(_secrets_path)
    return secrets


def load_tickers():
    _base_dir = os.path.expanduser('~')
    _ml_investments_dir = os.path.join(_base_dir, '.ml_investment')
    _tickers_path = os.path.join(_ml_investments_dir, 'tickers.json')
    tickers = load_json(_tickers_path)
    return tickers



def make_step_function(df, x_col, y_col):
    '''

    '''
    part_1 = df[[x_col, y_col]]
    part_1['idx'] = 2
    part_2 = part_1.copy()
    part_2[y_col] = part_2[y_col].shift(-1)
    part_2['idx'] = 1

    result = pd.concat([part_1, part_2], axis=0)
    result = result.sort_values([x_col, 'idx'])
    del result['idx']

    last_val = result[y_col].values[-1]
    curr_df = pd.DataFrame()
    curr_df[y_col] = [last_val]
    curr_df[x_col] = np.datetime64('now')
    
    result = pd.concat([result, curr_df], axis=0)
    
    return result


