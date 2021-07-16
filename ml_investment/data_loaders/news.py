import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from ..utils import load_json


class NewsData():
    '''
    Load news
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path


    def load(self, index: List[str]) -> pd.DataFrame:                     
        result = []
        for ticker in index:
            path = '{}/{}.csv'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            daily_df = pd.read_csv(path)
            result.append(daily_df)
        
        if len(result) == 0:
            return None

        result = pd.concat(result, axis=0).reset_index(drop=True)
        result = result.infer_objects()
        result['date'] = result['date'].astype(np.datetime64) 
                
        return result