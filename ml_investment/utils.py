import json
import os
import hashlib
import pandas as pd
import numpy as np
from copy import deepcopy


def save_json(path, data):
    if '/' in path:
        folder_path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    with open(path, "w") as write_file:
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


