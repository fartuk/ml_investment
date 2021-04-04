import pytest
import pandas as pd
import numpy as np
from ml_investment.data import SF1Data
from ml_investment.features import calc_series_stats, QuarterlyFeatures, BaseCompanyFeatures,\
                     QuarterlyDiffFeatures, FeatureMerger, \
                     DailyAggQuarterFeatures
from ml_investment.utils import load_json, int_hash_of_str
from synthetic_data import GeneratedData
config = load_json('config.json')

loaders = [GeneratedData()]
if config['sf1_data_path'] is not None:
    loaders.append(SF1Data(config['sf1_data_path']))
    

@pytest.mark.parametrize(
    ["series", "norm", "expected"],
    [([10, 0, 1], False,
      {'_mean': 3.6666666666666665,
       '_median': 1.0,
       '_max': 10.0,
       '_min': 0.0,
       '_std': 4.4969125210773475}),
     ([10, -30, 1, 4, 15.2],  False,
      {'_mean': 0.039999999999999855,
       '_median': 4.0,
       '_max': 15.2,
       '_min': -30.0,
       '_std': 15.798936673080249}), 
     ([1],  False,
      {'_mean': 1.0,
       '_median': 1.0,
       '_max': 1.0,
       '_min': 1.0,
       '_std': 0.0} ),
     ([10, -30, 1, 4, 15.2],  True,
      {'_mean': 0.0039999999999999855,
       '_median': .4,
       '_max': 1.52,
       '_min': -3.0,
       '_std': 1.5798936673080249})]
)
def test_calc_series_stats(series, norm, expected):
    result = calc_series_stats(series, norm=norm)
    assert type(result) == dict
    assert len(result) == len(expected)
    assert result.keys() == expected.keys()
    for key in result:
        assert np.isclose(result[key], expected[key])
        
    if norm == False:
        np.random.seed(0)
        np.random.shuffle(series)
        result = calc_series_stats(series, norm=norm)
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



        

class TestQuarterlyFeatures:
    @pytest.mark.parametrize('data_loader', loaders)    
    @pytest.mark.parametrize(
        ["tickers", "columns", "quarter_counts", "max_back_quarter"],
        [(['AAPL', 'TSLA'], ['ebit'], [2], 10), 
         (['NVDA', 'TSLA'], ['ebit'], [2, 4], 5), 
         (['AAPL', 'NVDA', 'TSLA', 'WORK'], ['ebit', 'debt'], [2, 4, 10], 10), 
         (['AAPL', 'ZLG'], ['ebit', 'debt'], [2, 4, 10], 5)]
    )
    def test_calculate(self, data_loader, tickers, columns, 
                                 quarter_counts, max_back_quarter):
        fc = QuarterlyFeatures(columns=columns,
                               quarter_counts=quarter_counts,
                               max_back_quarter=max_back_quarter)
                            
        X = fc.calculate(data_loader, tickers)

        assert type(X) == pd.DataFrame
        assert 'ticker' in X.index.names
        assert 'date' in X.index.names

        if type(data_loader) == GeneratedData:
            assert X.shape[0] == max_back_quarter * len(tickers)
        else:
            assert X.shape[0] <= max_back_quarter * len(tickers)

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



class TestQuarterlyDiffFeatures:
    @pytest.mark.parametrize('data_loader', loaders) 
    @pytest.mark.parametrize(
        ["tickers", "columns", "compare_quarter_idxs", "max_back_quarter"],
        [(['AAPL', 'TSLA'], ['ebit'], [1], 10), 
         (['NVDA', 'TSLA'], ['ebit'], [1, 4], 5), 
         (['AAPL', 'NVDA', 'TSLA', 'WORK'], ['ebit', 'debt'], [1, 4, 10], 10), 
         (['AAPL', 'ZLG'], ['ebit', 'debt'], [1, 4, 10], 5)]
    )
    def test_calculate(self, data_loader, tickers, columns, 
                                 compare_quarter_idxs, max_back_quarter):
        fc = QuarterlyDiffFeatures(columns=columns,
                                   compare_quarter_idxs=compare_quarter_idxs,
                                   max_back_quarter=max_back_quarter)
                            
        X = fc.calculate(data_loader, tickers)

        assert type(X) == pd.DataFrame
        assert 'ticker' in X.index.names
        assert 'date' in X.index.names

        if type(data_loader) == GeneratedData:
            assert X.shape[0] == max_back_quarter * len(tickers)
        else:
            assert X.shape[0] <= max_back_quarter * len(tickers)

        assert X.shape[1] == len(compare_quarter_idxs) * len(columns)


class WrapData:
    def __init__(self, data_loader, tickers):
        self.data_loader = data_loader
        self.tickers = tickers
        
    def load_base_data(self):
        df = pd.DataFrame()
        df['ticker'] = self.tickers
        df = pd.merge(df, self.data_loader.load_base_data(), how='left') 
        return df
    

class TestBaseCompanyFeatures:
    @pytest.mark.parametrize('data_loader', loaders) 
    @pytest.mark.parametrize(
        ["tickers", "cat_columns"],
        [(['AAPL', 'TSLA'], ['sector']), 
         (['NVDA', 'TSLA'], ['sector', 'sicindustry']), 
         (['AAPL', 'NVDA', 'TSLA', 'WORK'], ['sector', 'sicindustry']), 
         (['AAPL', 'ZLG'], ['sector', 'sicindustry'])]
    )
    def test_calculate(self, data_loader, tickers, cat_columns):                             
        fc = BaseCompanyFeatures(cat_columns=cat_columns)
        X = fc.calculate(data_loader, tickers)

        assert type(X) == pd.DataFrame
        assert 'ticker' in X.index.names
        base_data = data_loader.load_base_data()
        for col in cat_columns:
            assert len(base_data[col].unique()) ==\
                   len(fc.col_to_encoder[col].classes_)

        # Reuse fitted after first calculate fc
        for col in cat_columns:
            assert col in fc.col_to_encoder
        new_X = fc.calculate(data_loader, tickers)
        for col in cat_columns:
            assert (new_X[col] == X[col]).min()

        wd = WrapData(data_loader, tickers)
        new_X = fc.calculate(wd, tickers)
        for col in cat_columns:
            assert (new_X[col] == X[col]).min()




class TestFeatureMerger:
    @pytest.mark.parametrize('data_loader', loaders) 
    @pytest.mark.parametrize(
        "tickers",
        [['AAPL', 'TSLA'], ['NVDA', 'TSLA'], 
        ['AAPL', 'NVDA', 'TSLA', 'WORK'], ['AAPL', 'ZLG']]
    )    
    def test_calculate(self, data_loader, tickers):                            
        fc1 = QuarterlyFeatures(columns=['ebit'],
                                quarter_counts=[2],
                                max_back_quarter=10)

        fc2 = QuarterlyDiffFeatures(columns=['ebit', 'debt'], 
                                    compare_quarter_idxs=[1, 4],
                                    max_back_quarter=10)

        fc3 = BaseCompanyFeatures(cat_columns=['sector', 'sicindustry'])

        X1 = fc1.calculate(data_loader, tickers)
        X2 = fc2.calculate(data_loader, tickers)
        X3 = fc3.calculate(data_loader, tickers)
        
        fm1 = FeatureMerger(fc1, fc2, on=['ticker', 'date'])
        Xm1 = fm1.calculate(data_loader, tickers)

        fm2 = FeatureMerger(fc1, fc3, on='ticker')
        Xm2 = fm2.calculate(data_loader, tickers)

        assert Xm1.shape[0] == X1.shape[0]
        assert Xm2.shape[0] == X1.shape[0]
        assert Xm1.shape[1] == X1.shape[1] + X2.shape[1]
        assert Xm2.shape[1] == X1.shape[1] + X3.shape[1]
        assert (Xm1.index == X1.index).min()
        assert (Xm2.index == X1.index).min()
        
        new_cols = Xm1.columns[:X1.shape[1]]
        old_cols = X1.columns
        for nc, oc in zip(new_cols, old_cols):
            assert (Xm1[nc] == X1[oc]).min()

        new_cols = Xm2.columns[:X1.shape[1]]
        old_cols = X1.columns
        for nc, oc in zip(new_cols, old_cols):
            assert (Xm2[nc] == X1[oc]).min()




class TestDailyAggQuarterFeatures:
    @pytest.mark.parametrize('data_loader', loaders) 
    @pytest.mark.parametrize(
        ["tickers", "columns", "agg_day_counts", "max_back_quarter"],
        [(['AAPL', 'TSLA'], ['marketcap'], [100], 10), 
         (['NVDA', 'TSLA'], ['marketcap'], [100, 200], 5), 
         (['AAPL', 'NVDA', 'TSLA', 'WORK'], ['marketcap', 'pe'], [50, 200], 10), 
         (['AAPL', 'ZLG'], ['marketcap', 'pe'], [50, 200], 5)]
    )
    def test_calculate(self, data_loader, tickers, columns, 
                       agg_day_counts, max_back_quarter):
        fc = DailyAggQuarterFeatures(columns=columns,
                                     agg_day_counts=agg_day_counts,
                                     max_back_quarter=max_back_quarter)
                            
        X = fc.calculate(data_loader, tickers)
                
        assert type(X) == pd.DataFrame
        assert 'ticker' in X.index.names
        assert 'date' in X.index.names
       
        assert X.shape[0] <= max_back_quarter * len(tickers)     
        assert X.shape[1] == len(calc_series_stats([])) * \
                             len(columns) * len(agg_day_counts)
                                    

        for col in columns:
            for count in agg_day_counts:
                min_col = 'days{}_{}_min'.format(count, col)
                max_col = 'days{}_{}_max'.format(count, col)
                mean_col = 'days{}_{}_mean'.format(count, col)
                median_col = 'days{}_{}_median'.format(count, col)
                assert (X[max_col] >= X[min_col]).min()                
                assert (X[max_col] >= X[mean_col]).min()                
                assert (X[max_col] >= X[median_col]).min()                
                assert (X[mean_col] >= X[min_col]).min()                
                assert (X[median_col] >= X[min_col]).min()  













































