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



def nan_mask(arr):
    return np.isnan(arr) == False


def bound_filter_foo_gen(min_bound, max_bound):
    def foo(arr):
        result = np.isnan(arr) == False
        if min_bound is not None:
            result = result * (arr > min_bound)
        if max_bound is not None:
            result = result * (arr < max_bound)
        return result 
    return foo


def get_quarter_idx(date: np.datetime64):
    date = np.datetime64(date)
    year = date.astype(object).year
    month = date.astype(object).month
    bounds = np.array([3, 6, 9, 12])
    idx = np.where(bounds >= month)[0][0] + 1
    q_idx = '{}q{}'.format(year, idx)
    return q_idx





