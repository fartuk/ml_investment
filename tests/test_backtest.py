import pytest
import pandas as pd
import numpy as np
import os


from ml_investment.backtest.strategy import Strategy, Order
from ml_investment.utils import load_json, load_config, load_secrets
from pymongo import MongoClient


config = load_config()
secrets = load_secrets()



class DfData:
    def __init__(self, df):
        self.df = df

    def load(self, index):
        return self.df


class TestStrategy:
    @pytest.mark.parametrize(
        ["step_dates_path", "df_path", "expected_path"],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 'data/expected1.csv'),
            ('data/step_dates2.csv', 'data/df2.csv', 'data/expected2.csv'),
            ('data/step_dates3.csv', 'data/df3.csv', 'data/expected3.csv'),

        ]
      )
    def test__cast_data(self, step_dates_path, df_path, expected_path):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        
        df = pd.read_csv(df_path)
        df['date'] = df['date'].astype(np.datetime64)
        
        expected = pd.read_csv(expected_path)
        expected['date'] = expected['date'].astype(np.datetime64)
        
        strategy = Strategy()
        strategy.step_dates = step_dates
        cast_df = strategy._cast_data(df)
        pd.testing.assert_frame_equal(cast_df, expected)



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_path', 'expected_path',
         'date_col', 'price_col', 'return_col', 'return_format'],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 'data/expected4.csv',
             'date', 'price', 'return', 'ratio'),
            ('data/step_dates1.csv', 'data/df5.csv', 'data/expected5.csv',
             'date', 'price', 'return', 'ratio'),
            ('data/step_dates2.csv', 'data/df6.csv', 'data/expected6.csv',
             'date', 'price', 'return', 'ratio'),
            ('data/step_dates3.csv', 'data/df7.csv', 'data/expected7.csv',
             'date', 'price', 'return', 'ratio'),
            ('data/step_dates3.csv', 'data/df8.csv', 'data/expected7.csv',
             'Date', 'Close', 'return', 'ratio'),
            ('data/step_dates1.csv', 'data/df9.csv', 'data/expected5.csv',
             'date', 'price', 'return', 'change'),
            ('data/step_dates1.csv', 'data/df10.csv', 'data/expected5.csv',
             'date', 'price', 'adj_price', 'price'),
        ]
      )
    def test__check_create_ticker_data(self, step_dates_path, df_path, 
                                      expected_path, date_col, price_col,
                                      return_col, return_format):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        
        df = pd.read_csv(df_path)
        df[date_col] = df[date_col].astype(np.datetime64)
        
        expected = pd.read_csv(expected_path)
        expected['date'] = expected['date'].astype(np.datetime64)

        data_loader = DfData(df)

        strategy = Strategy()
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = date_col
        strategy.price_col = price_col
        strategy.return_col = return_col
        strategy.return_format = return_format
        strategy._check_create_ticker_data('AAPL')

        assert 'AAPL' in strategy._data
        result = strategy._data['AAPL']
        need_cols = ['price', 'return', 'missed', 'closed', 'prev_price']

        pd.testing.assert_frame_equal(result[need_cols], expected[need_cols])



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_path', 'step_idx', 'portfolio', 'cash', 'comission',
         'direction', 'expected'],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.,
             Order.BUY, 1),
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.6, 0.,
             Order.BUY, 0),
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.1, 
             Order.BUY, 0),
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.1,
             Order.SELL, 10),
            ('data/step_dates1.csv', 'data/df1.csv', 16, {'AAPL': 10}, 168.7, 0.,
             Order.BUY, None),
            ('data/step_dates2.csv', 'data/df2.csv', 16, {'AAPL': 10}, 168.7, 0.,
             Order.BUY, None),
            ('data/step_dates3.csv', 'data/df3.csv', 1, {'AAPL': 10}, 175., 0.,
             Order.BUY, 10),
            ('data/step_dates3.csv', 'data/df3.csv', 1, {'AAPL': 10}, 175., 0.0025,
             Order.BUY, 9),
        ]
      )
    def test__aposteriori_next_step_max_size(self, step_dates_path, df_path, step_idx, 
            portfolio, cash, direction, comission, expected):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]

        df = pd.read_csv(df_path)
        df['date'] = df['date'].astype(np.datetime64)

        data_loader = DfData(df)

        strategy = Strategy()
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy.comission = comission
        strategy._cash = cash
        strategy._check_create_ticker_data('AAPL')
        strategy.portfolio = portfolio
        strategy.step_idx = step_idx

        order = {'ticker': 'AAPL', 'direction': direction}
        result = strategy._aposteriori_next_step_max_size(order) 
        assert result == expected



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_path', 'step_idx', 'portfolio', 'cash', 'comission',
         'direction', 'size', 'allow_partial', 'creation_date', 'lifetime',
         'expected_portfolio', 'expected__cash', 'expected_location',
         'expected_size', 'expected_price', 'expected_status', 'expected_execution_date'],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.,
              Order.BUY, 1, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 11}, 0., 'orders', 1, 168.7,
              Order.COMPLETED, np.datetime64('2015-03-02')),
            
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.,
              Order.BUY, 3, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 11}, 0., 'orders', 1, 168.7,
              Order.PARTIAL, np.datetime64('2015-03-02')),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7*3+10., 0.,
              Order.BUY, 3, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 13}, 10., 'orders', 3, 168.7,
              Order.COMPLETED, np.datetime64('2015-03-02')),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7*3., 0.01,
              Order.BUY, 3, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 12}, 168.7*(3-2*1.01), 'orders', 2, 168.7,
              Order.PARTIAL, np.datetime64('2015-03-02')),
            
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 10, 0.,
              Order.SELL, 2, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 8}, 10 + 168.7*2, 'orders', 2, 168.7,
              Order.COMPLETED, np.datetime64('2015-03-02')),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 10, 0.,
              Order.SELL, 12, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 0}, 10 + 168.7*10, 'orders', 10, 168.7,
              Order.PARTIAL, np.datetime64('2015-03-02')),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 10, 0.,
              Order.SELL, 12, False, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 10}, 10, '_active_orders', 12, np.nan,
              np.nan, np.nan),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.,
              Order.BUY, 3, False, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 10}, 168.7, '_active_orders', 3, np.nan,
              np.nan, np.nan),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.6, 0.,
              Order.BUY, 3, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 10}, 168.6, '_active_orders', 3, np.nan,
              np.nan, np.nan),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 10}, 168.7, 0.01,
              Order.BUY, 3, True, np.datetime64('2015-02-27'), np.timedelta64(3, 'D'), 
              {'AAPL': 10}, 168.7, '_active_orders', 3, np.nan,
              np.nan, np.nan),

            ('data/step_dates1.csv', 'data/df1.csv', 12, {'AAPL': 10}, 243.3, 0.,
              Order.BUY, 1, True, np.datetime64('2015-03-16'), np.timedelta64(3, 'D'), 
              {'AAPL': 10}, 243.3, '_active_orders', 1, np.nan,
              np.nan, np.nan),

            ('data/step_dates1.csv', 'data/df1.csv', 11, {'AAPL': 10}, 243.3, 0.,
              Order.BUY, 1, True, np.datetime64('2015-03-15'), np.timedelta64(4, 'D'), 
              {'AAPL': 11}, 0., 'orders', 1, 243.3,
              Order.COMPLETED, np.datetime64('2015-03-16')),

            ('data/step_dates1.csv', 'data/df1.csv', 11, {'AAPL': 10}, 243.3, 0.,
              Order.BUY, 1, True, np.datetime64('2015-03-12'), np.timedelta64(4, 'D'), 
              {'AAPL': 11}, 0., 'orders', 1, 243.3,
              Order.COMPLETED, np.datetime64('2015-03-16')),

            ('data/step_dates1.csv', 'data/df1.csv', 11, {'AAPL': 10}, 243.3, 0.,
              Order.BUY, 1, True, np.datetime64('2015-03-09'), np.timedelta64(4, 'D'), 
              {'AAPL': 10}, 243.3, 'orders', 1, np.nan,
              Order.EXPIRED, np.nan),
        ]
      )
    def test__execute_market_order(self, 
            step_dates_path, df_path, step_idx, portfolio, cash, comission,
            direction, size, allow_partial, creation_date, lifetime,
            expected_portfolio, expected__cash, expected_location,
            expected_size, expected_price, expected_status, expected_execution_date):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]

        df = pd.read_csv(df_path)
        df['date'] = df['date'].astype(np.datetime64)

        data_loader = DfData(df)

        strategy = Strategy()
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy.comission = comission
        strategy._cash = cash
        strategy._check_create_ticker_data('AAPL')
        strategy.portfolio = portfolio
        strategy.step_idx = step_idx

        order = {'ticker': 'AAPL', 'direction': direction, 'size': size,
                 'allow_partial': allow_partial, 'creation_date': creation_date,
                 'lifetime': lifetime}

        strategy._execute_market_order(order) 
        
        if expected_location == '_active_orders':
            assert len(strategy._active_orders) == 1
            assert len(strategy.orders) == 0
            result = strategy._active_orders[0]
            assert 'execution_date' not in result
            result['execution_date'] = np.nan
            assert 'price' not in result
            result['price'] = np.nan
            assert 'status' not in result
            result['status'] = np.nan

        if expected_location == 'orders':
            assert len(strategy.orders) == 1
            assert len(strategy._active_orders) == 0
            result = strategy.orders[0]

        assert strategy.portfolio == expected_portfolio
        np.testing.assert_almost_equal(strategy._cash, expected__cash)

        np.testing.assert_almost_equal(result['size'], expected_size)
        np.testing.assert_almost_equal(result['price'], expected_price)
        np.testing.assert_equal(result['status'], expected_status)
        np.testing.assert_equal(result['execution_date'], expected_execution_date)


















