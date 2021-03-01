import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from data import SF1Data





class QuarterlyTarget:
    def __init__(self, col, quarter_shift:int=0):
        self.col = col
        self.quarter_shift = quarter_shift
        self._data_loader = None
        
        
    def _single_ticker_target(self, ticker_and_dates):
        ticker, dates = ticker_and_dates
        quarterly_data = self._data_loader.load_quarterly_data([ticker])[::-1]
        quarter_dates = quarterly_data['date'].astype(np.datetime64).values
        vals = []
        for date in dates:
            curr_date_mask = quarter_dates == np.datetime64(date)
            curr_quarter_idx = np.where(curr_date_mask)[0][0]
            idx = curr_quarter_idx + self.quarter_shift
            if idx >= 0 and idx < len(quarterly_data):
                value = quarterly_data[self.col].values[idx]
            else:
                value = np.nan
                
            vals.append(value)

        result = pd.DataFrame()
        result['y'] = vals
        result['date'] = dates
        result['ticker'] = ticker

        return result        
        

    def calculate(self, data_loader, info_df):
        '''
        info_df: pd.DataFrame. Should have columns: ["ticker", "date"] 
        '''
        self._data_loader = data_loader
        grouped = info_df.groupby('ticker')['date'].apply(lambda x:
                  x.tolist()).reset_index()
        params = [(ticker, dates) for ticker, dates in grouped.values]

        n_jobs=10
        p = Pool(n_jobs)
        result = []
        for ticker_result in tqdm(p.imap(self._single_ticker_target, params)):
            result.append(ticker_result)

        result = pd.concat(result, axis=0)
        result = pd.merge(info_df, result, on=['ticker', 'date'], how='left')
        result = result.set_index(['ticker', 'date'])
        
        return result


class QuarterlyDiffTarget:
    def __init__(self, col, norm=True):
        self.curr_target = QuarterlyTarget(col=col, quarter_shift=0)
        self.last_target = QuarterlyTarget(col=col, quarter_shift=-1)
        self.norm = norm

    
    def calculate(self, data_loader, info_df):
        curr_df = self.curr_target.calculate(data_loader, info_df)
        last_df = self.last_target.calculate(data_loader, info_df)
        curr_df['y'] = curr_df['y'] - last_df['y']
        if self.norm:
            curr_df['y'] = curr_df['y'] / last_df['y']

        return curr_df


class QuarterlyBinDiffTarget:
    def __init__(self, col, norm=True):
        self.target = QuarterlyDiffTarget(col=col, norm=False)
    
    def calculate(self, data_loader, info_df):
        target_df = self.curr_target.calculate(data_loader, info_df)
        target_df['y'] = target_df['y'] > 0

        return target_df
























