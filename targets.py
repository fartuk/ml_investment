import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from data import load_quarterly_data_cf1





class QuarterlyTarget:
    def __init__(self, col, quarter_shift:int=0):
        self.col = col
        self.quarter_shift = quarter_shift
        self._data_path = None
        
        
    def _single_ticker_target(self, ticker_and_dates):
        ticker, dates = ticker_and_dates
        quarterly_data = load_quarterly_data_cf1(ticker, self._data_path)[::-1]
        quarter_dates = [np.datetime64(x['date']) for x in quarterly_data]
        quarter_dates = np.array(quarter_dates)
        vals = []
        for date in dates:
            curr_date_mask = quarter_dates == np.datetime64(date)
            curr_quarter_idx = np.where(curr_date_mask)[0][0]
            idx = curr_quarter_idx + self.quarter_shift
            if idx >= 0 and idx < len(quarterly_data):
                value = quarterly_data[idx][self.col]
            else:
                value = np.nan
                
            vals.append(value)

        result = pd.DataFrame()
        result['y'] = vals
        result['date'] = dates
        result['ticker'] = ticker

        return result        
        

    def calculate(self, data_path, info_df):
        '''
        info_df: pd.DataFrame. Should have columns: ["ticker", "date"] 
        '''
        self._data_path = data_path
        grouped = info_df.groupby('ticker')['date'].apply(lambda x: x.tolist()).reset_index()
        params = [(ticker, dates) for ticker, dates in grouped.values]

        n_jobs=10
        p = Pool(n_jobs)
        result = []
        for ticker_result in tqdm(p.imap(self._single_ticker_target, params)):
            result.append(ticker_result)

        result = pd.concat(result, axis=0)
        result = pd.merge(info_df, result, on=['ticker', 'date'], how='left')
        
        return result










