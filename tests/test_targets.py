import pytest
import pandas as pd
import numpy as np
from ml_investment.data import SF1Data
from ml_investment.targets import QuarterlyTarget, QuarterlyDiffTarget, \
                    QuarterlyBinDiffTarget, DailyAggTarget, \
                    ReportGapTarget
from ml_investment.utils import load_json
from synthetic_data import PreDefinedData
config = load_json('config.json')


     
class TestQuarterlyTarget:       
    @pytest.mark.parametrize(
        ["ticker_dates", "quarter_shift", "expected"],
        [([['A', '2018-12-05'], ['A', '2018-11-05']], 0, 
          [10, 3]),
         ([['A', '2018-11-05'], ['A', '2018-11-05'], ['A', '2018-10-05']], 0, 
          [3, 3, -5]), 
         ([['A', '2018-09-05'], ['A', '2018-12-05'], ['A', '2018-11-05']], 0, 
          [25, 10, 3]),
         ([['A', '2018-09-05'], ['A', '2018-12-05'], ['A', '2018-11-05']], -1, 
          [1e5, 3, -5]),
         ([['A', '2018-12-05'], ['A', '2018-11-05']], 1, 
          [np.nan, 10]),    
         ([['A', '2018-12-05'], ['A', '2018-11-05'], ['A', '2018-10-05']], 2, 
          [np.nan, np.nan, 10]),      
          ]
    )        
    def test_calculate_synth(self, ticker_dates, quarter_shift, expected):
        data_loader = PreDefinedData()
        target = QuarterlyTarget('marketcap', quarter_shift=quarter_shift)
        info_df = pd.DataFrame(ticker_dates)
        info_df.columns = ['ticker', 'date']   
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values, expected)



    @pytest.mark.skipif(config['sf1_data_path'] is None, reason="There are no SF1 dataset")
    @pytest.mark.parametrize(
        "tickers",
        [['AAPL', 'TSLA'], ['NVDA', 'TSLA'], 
        ['AAPL', 'NVDA', 'TSLA', 'WORK'], ['AAPL', 'ZLG']]
    )
    def test_calculate(self, tickers):
        data_loader = SF1Data(config['sf1_data_path'])
        quarterly_df = data_loader.load_quarterly_data(tickers,
                                                       quarter_count=None)
                                                       
        target = QuarterlyTarget('marketcap', quarter_shift=0)
        info_df = quarterly_df.drop_duplicates('ticker', keep='first') \
                                        [['ticker', 'date', 'marketcap']]

        y = target.calculate(data_loader, info_df[['ticker', 'date']])
        assert type(y) == pd.DataFrame
        assert 'y' in y.columns
        np.testing.assert_array_equal(y['y'].values,
                                      info_df['marketcap'].values)
        
        info_df = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values,
                                      info_df['marketcap'].values)


        target = QuarterlyTarget('marketcap', quarter_shift=1)
        info_df = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values, 
                                      info_df.groupby('ticker')['marketcap']\
                                      .shift(1).astype('float').values)


        target = QuarterlyTarget('marketcap', quarter_shift=-3)
        info_df = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values, 
                                      info_df.groupby('ticker')['marketcap']\
                                      .shift(-3).astype('float').values)



class TestQuarterlyDiffTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "norm", "expected"],
        [([['A', '2018-12-05'], ['A', '2018-11-05']], False, 
          [7, 8]),
         ([['A', '2018-12-05'], ['A', '2018-11-05']], True, 
          [7/3, 8/5]), 
         ([['A', '2018-12-05'], ['A', '2018-11-05'], ['A', '2018-07-05']], True, 
          [7/3, 8/5, np.nan]),
         ([['A', '2018-10-05'], ['A', '2018-08-05']], False, 
          [-30, 1e5-2])]
    )        
    def test_calculate_synth(self, ticker_dates, norm, expected):
        data_loader = PreDefinedData()
        target = QuarterlyDiffTarget('marketcap', norm=norm)
        info_df = pd.DataFrame(ticker_dates)
        info_df.columns = ['ticker', 'date']   
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values, expected)

    @pytest.mark.skipif(config['sf1_data_path'] is None, reason="There are no SF1 dataset")
    @pytest.mark.parametrize(
        "tickers",
        [['AAPL', 'TSLA'], ['NVDA', 'TSLA'], 
        ['AAPL', 'NVDA', 'TSLA', 'WORK'], ['AAPL', 'ZLG']]
    )
    def test_calculate(self, tickers):
        data_loader = SF1Data(config['sf1_data_path'])
        quarterly_df = data_loader.load_quarterly_data(tickers,
                                                       quarter_count=None)
                                                     
        target = QuarterlyDiffTarget('marketcap', norm=False)
        info_df = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(data_loader, info_df)
        assert type(y) == pd.DataFrame
        assert 'y' in y.columns
        assert len(y) == len(info_df)
        gt = info_df['marketcap'].astype('float') - \
             info_df.groupby('ticker')['marketcap'].shift(-1).astype('float')
        np.testing.assert_array_equal(y['y'].values, gt.values)
                                      



class TestQuarterlyBinDiffTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "expected"],
        [([['A', '2018-12-05'], ['A', '2018-11-05']], 
          [True, True]),
         ([['A', '2018-12-05'], ['A', '2018-11-05']],
          [True, True]),
         ([['A', '2018-12-05'], ['A', '2018-11-05'], ['A', '2018-07-05']], 
          [True, True, np.nan])]
    )        
    def test_calculate_synth(self, ticker_dates, expected):
        data_loader = PreDefinedData()
        target = QuarterlyBinDiffTarget('marketcap')
        info_df = pd.DataFrame(ticker_dates)
        info_df.columns = ['ticker', 'date']   
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)


class TestDailyAggTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "horizon", "foo", "expected"],
        [([['A', '2018-11-05']], 1, np.mean, [25]), 
         ([['A', '2018-11-05']], 3, np.mean, [23 / 3]), 
         ([['A', '2018-11-05'], ['A', '2018-11-01']], -2, np.mean, 
          [51, np.nan]),
         ([['A', '2018-11-05'], ['A', '2018-11-01']], 3, np.max, 
          [25, 23])]
    )        
    def test_calculate_synth(self, ticker_dates, horizon, foo, expected):
        data_loader = PreDefinedData()
        target = DailyAggTarget('marketcap', horizon=horizon, foo=foo)
        info_df = pd.DataFrame(ticker_dates)
        info_df.columns = ['ticker', 'date']   
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)


class TestReportGapTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "norm", "expected"],
        [([['A', '2018-11-05']], False, [(25-100)]), 
         ([['A', '2018-11-05']], True, [(25-100)/100]), 
         ([['A', '2018-11-05'], ['A', '2018-11-01']], False, 
          [25-100, np.nan]),
         ([['A', '2018-11-05'], ['A', '2018-11-08']], False, 
          [25-100, 7])
        ]
    )        
    def test_calculate_synth(self, ticker_dates, norm, expected):
        data_loader = PreDefinedData()
        target = ReportGapTarget('marketcap', norm=norm)
        info_df = pd.DataFrame(ticker_dates)
        info_df.columns = ['ticker', 'date']   
        y = target.calculate(data_loader, info_df)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)







































































