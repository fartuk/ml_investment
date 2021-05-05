import pytest
import os
import pandas as pd
import numpy as np
from ml_investment.data_loaders.sf1 import SF1QuarterlyData, SF1BaseData
from ml_investment.targets import QuarterlyTarget, QuarterlyDiffTarget, \
                    QuarterlyBinDiffTarget, DailyAggTarget, \
                    ReportGapTarget, BaseInfoTarget,\
                    DailySmoothedQuarterlyDiffTarget
from ml_investment.utils import load_config
from synthetic_data import PredefQuarterlyData, PredefDailyData

config = load_config()

predefined_data = {
    'quarterly': PredefQuarterlyData(),   
    'daily': PredefDailyData(),    
}


sf1_data = {
    'quarterly': SF1QuarterlyData(config['sf1_data_path']), 
    'base': SF1BaseData(config['sf1_data_path']),
}

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
        target = QuarterlyTarget(data_key='quarterly',
                                 col='marketcap',
                                 quarter_shift=quarter_shift)
        info_df = pd.DataFrame(ticker_dates)
        info_df.columns = ['ticker', 'date']   
        y = target.calculate(predefined_data, info_df)
        np.testing.assert_array_equal(y['y'].values, expected)



    @pytest.mark.skipif(not os.path.exists(config['sf1_data_path']),
                        reason="There are no SF1 dataset")
    @pytest.mark.parametrize(
        "tickers",
        [['AAPL', 'TSLA'], ['NVDA', 'TSLA'], 
        ['AAPL', 'NVDA', 'TSLA', 'WORK'], ['AAPL', 'ZLG']]
    )
    def test_calculate(self, tickers):
        quarterly_df = sf1_data['quarterly'].load(tickers)

        target = QuarterlyTarget(data_key='quarterly',
                                 col='marketcap',
                                 quarter_shift=0)

        # Firstly check only last quarter for base check
        index = quarterly_df.drop_duplicates('ticker', keep='first') \
                                        [['ticker', 'date', 'marketcap']]
        y = target.calculate(sf1_data, index[['ticker', 'date']])
        assert type(y) == pd.DataFrame
        assert 'y' in y.columns
        np.testing.assert_array_equal(y['y'].values, index['marketcap'].values)

        # Check all quarters too
        index = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(sf1_data, index)
        np.testing.assert_array_equal(y['y'].values, index['marketcap'].values)


        target = QuarterlyTarget(data_key='quarterly', 
                                 col='marketcap',
                                 quarter_shift=1)
        index = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(sf1_data, index)
        np.testing.assert_array_equal(y['y'].values, 
                                      index.groupby('ticker')['marketcap']\
                                      .shift(1).astype('float').values)


        target = QuarterlyTarget(data_key='quarterly',
                                 col='marketcap', 
                                 quarter_shift=-3)
        index = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(sf1_data, index)
        np.testing.assert_array_equal(y['y'].values, 
                                      index.groupby('ticker')['marketcap']\
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
        target = QuarterlyDiffTarget(data_key='quarterly',
                                     col='marketcap', 
                                     norm=norm)
        index = pd.DataFrame(ticker_dates)
        index.columns = ['ticker', 'date']   
        y = target.calculate(predefined_data, index)
        np.testing.assert_array_equal(y['y'].values, expected)

    @pytest.mark.skipif(not os.path.exists(config['sf1_data_path']), 
                        reason="There are no SF1 dataset")
    @pytest.mark.parametrize(
        "tickers",
        [['AAPL', 'TSLA'], ['NVDA', 'TSLA'], 
        ['AAPL', 'NVDA', 'TSLA', 'WORK'], ['AAPL', 'ZLG']]
    )
    def test_calculate(self, tickers):
        quarterly_df = sf1_data['quarterly'].load(tickers)

        target = QuarterlyDiffTarget(data_key='quarterly',
                                     col='marketcap',
                                     norm=False)
        index = quarterly_df[['ticker', 'date', 'marketcap']]
        y = target.calculate(sf1_data, index)
        assert type(y) == pd.DataFrame
        assert 'y' in y.columns
        assert len(y) == len(index)
        gt = index['marketcap'].astype('float') - \
             index.groupby('ticker')['marketcap'].shift(-1).astype('float')
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
        target = QuarterlyBinDiffTarget(data_key='quarterly', col='marketcap')
        index = pd.DataFrame(ticker_dates)
        index.columns = ['ticker', 'date']   
        y = target.calculate(predefined_data, index)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)


class TestDailyAggTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "horizon", "foo", "expected"],
        [([['A', '2018-11-05']], 1, np.mean, [25]), 
         ([['A', '2018-11-05']], 3, np.mean, [23 / 3]), 
         ([['A', '2018-11-05'], ['A', '2018-10-03']], -2, np.mean, 
          [51, np.nan]),
         ([['A', '2018-11-05'], ['A', '2018-11-01']], 3, np.max, 
          [25, 23])]
    )        
    def test_calculate_synth(self, ticker_dates, horizon, foo, expected):
        target = DailyAggTarget(data_key='daily',
                                col='marketcap',
                                horizon=horizon,
                                foo=foo)
        index = pd.DataFrame(ticker_dates)
        index.columns = ['ticker', 'date']   
        y = target.calculate(predefined_data, index)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)


class TestReportGapTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "norm", "expected"],
        [([['A', '2018-11-05']], False, [(25-100)]), 
         ([['A', '2018-11-05']], True, [(25-100)/100]), 
         ([['A', '2018-11-05'], ['A', '2018-10-03']], False, 
          [25-100, np.nan]),
         ([['A', '2018-11-05'], ['A', '2018-11-08']], False, 
          [25-100, 7])
        ]
    )        
    def test_calculate_synth(self, ticker_dates, norm, expected):
        target = ReportGapTarget(data_key='daily', col='marketcap', norm=norm)
        index = pd.DataFrame(ticker_dates)
        index.columns = ['ticker', 'date']   
        y = target.calculate(predefined_data, index)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)



class TestBaseInfoTarget:       
    @pytest.mark.skipif(not os.path.exists(config['sf1_data_path']),
                        reason="There are no SF1 dataset")
    @pytest.mark.parametrize(
        "tickers",
        [['AAPL', 'TSLA'], ['NVDA', 'TSLA'], 
        ['AAPL', 'NVDA', 'TSLA', 'WORK'], ['AAPL', 'ZLG']]
    )
    def test_calculate(self, tickers):
        quarterly_df = sf1_data['quarterly'].load(tickers)

        target = BaseInfoTarget(data_key='base', col='sector')
        index = quarterly_df.drop_duplicates('ticker', keep='first') \
                                        [['ticker', 'date']]

        y = target.calculate(sf1_data, index[['ticker']])
        assert type(y) == pd.DataFrame
        assert 'y' in y.columns


class TestDailySmoothedQuarterlyDiffTarget:
    @pytest.mark.parametrize(
        ["ticker_dates", "smooth_horizon", "norm", "expected"],
        [([['A', '2018-11-05']], 1, False, [(23)],),
         ([['A', '2018-11-05']], 1, True, [(11.5)],),
         ([['A', '2018-11-05']], 2, False, [(10-1.5)],),
         ([['A', '2018-11-05']], 2, True, [(10-1.5)/1.5],),
        ] 
    )        
    def test_calculate_synth(self, ticker_dates, smooth_horizon, norm, expected):
        target = DailySmoothedQuarterlyDiffTarget(
                    daily_data_key='daily',
                    quarterly_data_key='quarterly',
                    col='marketcap',
                    norm=norm,
                    smooth_horizon=smooth_horizon)
        index = pd.DataFrame(ticker_dates)
        index.columns = ['ticker', 'date']   
        y = target.calculate(predefined_data, index)
        np.testing.assert_array_equal(y['y'].values.astype('float'), expected)






































