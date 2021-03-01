import pytest
import hashlib
import pandas as pd
import numpy as np
from data import SF1Data
from features import calc_series_stats, QuarterlyFeatures
from utils import load_json
config = load_json('config.json')





@pytest.mark.parametrize(
    ["series", "expected"],
    [([10, 0, 1], 
      {'_mean': 3.6666666666666665,
       '_median': 1.0,
       '_max': 10.0,
       '_min': 0.0,
       '_std': 4.4969125210773475}),
     ([10, -30, 1, 4, 15.2], 
      {'_mean': 0.039999999999999855,
       '_median': 4.0,
       '_max': 15.2,
       '_min': -30.0,
       '_std': 15.798936673080249}), 
     ([1],
      {'_mean': 1.0,
       '_median': 1.0,
       '_max': 1.0,
       '_min': 1.0,
       '_std': 0.0} )]
)
def test_calc_series_stats(series, expected):
    result = calc_series_stats(series)
    assert type(result) == dict
    assert len(result) == len(expected)
    assert result.keys() == expected.keys()
    for key in result:
        assert np.isclose(result[key], expected[key])
        
    np.random.seed(0)
    np.random.shuffle(series)
    result = calc_series_stats(series)
    for key in result:
        assert np.isclose(result[key], expected[key])


def test_calc_series_stats_nans():
    assert calc_series_stats([np.nan, 10, 0, 1]) == calc_series_stats([10, 0, 1])
    assert calc_series_stats([None, 10, 0, 1]) == calc_series_stats([10, 0, 1])
    assert calc_series_stats([10, 0, np.nan, 1]) == calc_series_stats([10, 0, 1])
    
    result = calc_series_stats([])
    for key in result:
        assert np.isnan(result[key])
        
    result = calc_series_stats([np.nan, None])
    for key in result:
        assert np.isnan(result[key])        


def int_hash(text):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
     
        
class Data:
    def __init__(self, columns):
        self.columns = columns
    
    def load_quarterly_data(self, tickers, quarter_count=None):
        size=20
        df = pd.DataFrame()
        df['ticker'] = tickers * size
        df['date'] = np.nan
        np.random.seed(int_hash(str(tickers)))
        for col in self.columns:
            df[col] = np.random.uniform(-1e5, 1e5, size)
        
        return df
        

class TestQuarterlyFeatures:
    @pytest.mark.parametrize(
        ["tickers", "columns", "quarter_counts", "max_back_quarter"],
        [(['AAPL', 'TSLA'], ['ebit'], [2], 10), 
         (['NVDA', 'TSLA'], ['ebit'], [2, 4], 5), 
         (['AAPL', 'NVDA', 'TSLA', 'WORK'], ['ebit', 'debt'], [2, 4, 10], 5)]
    )
    def test_calculate_synthetic(self, tickers, columns, 
                                 quarter_counts, max_back_quarter):
        fc = QuarterlyFeatures(columns=columns,
                               quarter_counts=quarter_counts,
                               max_back_quarter=max_back_quarter)
                            
        loaders = [Data(columns), SF1Data(config['sf1_data_path'])]   
        for data_loader in loaders:
            X = fc.calculate(data_loader, tickers)
            
            assert type(X) == pd.DataFrame
            assert X.shape[0] == max_back_quarter * len(tickers)
            assert X.shape[1] == 2 * len(calc_series_stats([])) * \
                                 len(columns) * len(quarter_counts)
                             
            # Minimum can not be lower with reduction of quarter_count    
            sorted_quarter_counts = np.sort(quarter_counts)
            for col in columns:
                for k in range(len(sorted_quarter_counts) - 1):
                    lower_count = sorted_quarter_counts[k]
                    higher_count = sorted_quarter_counts[k + 1]
                    l_col = 'quarter{}_{}_min'.format(lower_count, col)
                    h_col = 'quarter{}_{}_min'.format(higher_count, col)
                    
                    assert (X[h_col] <= X[l_col]).min()

            # Maximum can not be higher with reduction of quarter_count    
            sorted_quarter_counts = np.sort(quarter_counts)
            for col in columns:
                for k in range(len(sorted_quarter_counts) - 1):
                    lower_count = sorted_quarter_counts[k]
                    higher_count = sorted_quarter_counts[k + 1]
                    l_col = 'quarter{}_{}_max'.format(lower_count, col)
                    h_col = 'quarter{}_{}_max'.format(higher_count, col)
                    
                    assert (X[h_col] >= X[l_col]).min()                

            std_cols = [x for x in X.columns if '_std' in x]
            for col in std_cols:
                assert X[col].min() >= 0

            for col in columns:
                for count in quarter_counts:
                    min_col = 'quarter{}_{}_min'.format(count, col)
                    max_col = 'quarter{}_{}_max'.format(count, col)
                    mean_col = 'quarter{}_{}_mean'.format(count, col)
                    median_col = 'quarter{}_{}_median'.format(count, col)
                    assert (X[max_col] >= X[min_col]).min()                
                    assert (X[max_col] >= X[mean_col]).min()                
                    assert (X[max_col] >= X[median_col]).min()                
                    assert (X[mean_col] >= X[min_col]).min()                
                    assert (X[median_col] >= X[min_col]).min()                

































































