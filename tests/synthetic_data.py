import numpy as np
import pandas as pd
from ml_investment.utils import int_hash_of_str            



class GeneratedData:
    def __init__(self):
        self.quarterly_columns = ['marketcap', 'ebit', 'debt', 'netinc', \
                                  'ncf', 'fcf', 'revenue', 'ebitda']
        self.daily_columns = ['marketcap', 'pe']
        self.cat_columns = ['sector', 'sicindustry']
        self.tickers = ['AAPL', 'TSLA', 'NVDA', 'WORK', 'ZLG']
        self.quart_cnt = 50
        self.day_cnt = 10000
    
    def load_quarterly_data(self, tickers, quarter_count=None):
        df = pd.DataFrame()
        df['ticker'] = [x for x in tickers for _ in range(self.quart_cnt)]
        df['date'] = [np.datetime64('2020-01-17') - 90 * np.timedelta64(k, 'D')
                      for k in range(self.quart_cnt)] * len(tickers)
        np.random.seed(int_hash_of_str(str(tickers)))
        for col in self.quarterly_columns:
            if col == 'marketcap':
                df[col] = np.random.uniform(1000, 1e5, self.quart_cnt * len(tickers))            
            else:
                df[col] = np.random.uniform(-1e5, 1e5, self.quart_cnt * len(tickers))
        
        return df
        
    def load_daily_data(self, tickers):
        df = pd.DataFrame()
        df['ticker'] = [x for x in tickers for _ in range(self.day_cnt)]
        df['date'] = [np.datetime64('2020-01-17') - np.timedelta64(k, 'D')
                      for k in range(self.day_cnt)] * len(tickers)
        
        np.random.seed(int_hash_of_str(str(tickers)))
        for col in self.daily_columns:
            df[col] = np.random.uniform(-1e5, 1e5, self.day_cnt * len(tickers))
        
        return df        
        
        
    def load_base_data(self):
        df = pd.DataFrame()
        df['ticker'] = self.tickers
        for col in self.cat_columns:
            np.random.seed(int_hash_of_str(col))
            df[col] = np.random.randint(-2, 2, len(self.tickers))

        return df
    
    
class PreDefinedData:
    def load_quarterly_data(self, tickers):
        df = pd.DataFrame()
        df['date'] = ['2018-12-05', '2018-11-05', '2018-10-05',
                      '2018-09-05', '2018-08-05', '2018-07-05'] 
        df['ticker'] = ['A'] * 6
        df['marketcap'] = [10, 3, -5, 25, 1e5, 2]
            
        return df


    def load_daily_data(self, tickers):
        df = pd.DataFrame()
        df['date'] = ['2018-11-08', '2018-11-07', '2018-11-06', '2018-11-05',
                      '2018-11-04', '2018-11-03', '2018-11-02', '2018-11-01'] 
        df['ticker'] = ['A'] * 8
        df['marketcap'] = [10, 3, -5, 25, 100, 2, 23., -7]
            
        return df    
    
    
    
    
    
    
    
    