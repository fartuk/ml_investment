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
        if type(self.df) == dict:
            return self.df[index[0]]
        else:
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
        strategy.latency = np.timedelta64(1, 'h')
        strategy._cash = cash
        strategy._check_create_ticker_data('AAPL')
        strategy.portfolio = portfolio
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]

        order = {'ticker': 'AAPL', 'direction': direction, 'size': size,
                 'allow_partial': allow_partial, 'creation_date': creation_date,
                 'submit_date': creation_date + strategy.latency, 'lifetime': lifetime}

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


    @pytest.mark.parametrize(
        ['step_dates_path', 'df_pathes', 'step_idx', 'portfolio', 'cash', 
         'expected__cash'],
        [
            ('data/step_dates1.csv', ['data/df1.csv', 'data/df2.csv'], 4,
             {'AAPL0': 2, 'AAPL1': 10}, 120, 120.),

            ('data/step_dates1.csv', ['data/df1.csv', 'data/df2.csv'], 9,
             {'AAPL0': 2, 'AAPL1': 10}, 120, 120.),

            ('data/step_dates1.csv', ['data/df1.csv', 'data/df2.csv'], 9,
             {'AAPL0': 0, 'AAPL1': 10}, 120, 120.),

            ('data/step_dates1.csv', ['data/df1.csv', 'data/df5.csv'], 5,
             {'AAPL0': 2, 'AAPL1': 10}, 120, 285.5),

            ('data/step_dates1.csv', ['data/df6.csv', 'data/df5.csv'], 5,
             {'AAPL0': 2, 'AAPL1': 10}, 0., 169.26),
        ]
    )
    def test__receive_dividends(self, step_dates_path, df_pathes, step_idx, 
                                portfolio, cash, expected__cash):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        dfs = {}
        for k, df_path in enumerate(df_pathes):
            df = pd.read_csv(df_path)
            df['date'] = df['date'].astype(np.datetime64)
            dfs['AAPL{}'.format(k)] = df
        data_loader = DfData(dfs)

        strategy = Strategy()
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy._cash = cash
        strategy.portfolio = portfolio
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]

        for k, df_path in enumerate(df_pathes):
            strategy._check_create_ticker_data('AAPL{}'.format(k))

        strategy._receive_dividends()

        np.testing.assert_almost_equal(strategy._cash, expected__cash)



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_pathes', 'step_idx', 'portfolio', 'cash', 
         'expected_equity'],
        [
            ('data/step_dates1.csv', ['data/df1.csv', 'data/df2.csv'], 4,
             {'AAPL0': 2, 'AAPL1': 10}, 120, 634.),
            
            ('data/step_dates1.csv', ['data/df1.csv', 'data/df2.csv'], 9,
             {'AAPL0': 2, 'AAPL1': 10}, 120, 746.4),

            ('data/step_dates1.csv', ['data/df1.csv', 'data/df2.csv'], 9,
             {'AAPL0': 0, 'AAPL1': 10}, 120, 262.),

            ('data/step_dates1.csv', ['data/df1.csv', 'data/df5.csv'], 5,
             {'AAPL0': 2, 'AAPL1': 10}, 120, 2144.4),

            ('data/step_dates1.csv', ['data/df6.csv', 'data/df5.csv'], 5,
             {'AAPL0': 2, 'AAPL1': 10}, 0., 1713.8),
        ]
    )
    def test__calc_equity(self, step_dates_path, df_pathes, step_idx, 
                          portfolio, cash, expected_equity):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        dfs = {}
        for k, df_path in enumerate(df_pathes):
            df = pd.read_csv(df_path)
            df['date'] = df['date'].astype(np.datetime64)
            dfs['AAPL{}'.format(k)] = df
        data_loader = DfData(dfs)

        strategy = Strategy()
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy._cash = cash
        strategy.portfolio = portfolio
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]

        for k, df_path in enumerate(df_pathes):
            strategy._check_create_ticker_data('AAPL{}'.format(k))

        equity = strategy._calc_equity()

        np.testing.assert_almost_equal(equity, expected_equity)



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_pathes', 'step_idx', 'portfolio', 
         'expected_orders'],
        [
            ('data/step_dates3.csv', ['data/df2.csv', 'data/df6.csv'], 10,
             {'AAPL0': 2, 'AAPL1': 10}, 
             [{'ticker': 'AAPL0', 'direction': Order.SELL, 'size': 2},
              {'ticker': 'AAPL1', 'direction': Order.SELL, 'size': 10}]),

            ('data/step_dates3.csv', ['data/df2.csv', 'data/df5.csv'], 8,
             {'AAPL0': 2, 'AAPL1': 10}, 
             [{'ticker': 'AAPL1', 'direction': Order.SELL, 'size': 10}]),
            
            ('data/step_dates3.csv', ['data/df2.csv', 'data/df5.csv'], 7,
             {'AAPL0': 2, 'AAPL1': 10}, 
             []),
        ]
    )
    def test__post_close_orders(self, step_dates_path, df_pathes, step_idx, 
                                portfolio, expected_orders):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        dfs = {}
        for k, df_path in enumerate(df_pathes):
            df = pd.read_csv(df_path)
            df['date'] = df['date'].astype(np.datetime64)
            dfs['AAPL{}'.format(k)] = df
        data_loader = DfData(dfs)

        strategy = Strategy()
        strategy.latency = np.timedelta64(1, 'h')
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy.portfolio = portfolio
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]
        strategy.verbose = False

        for k, df_path in enumerate(df_pathes):
            strategy._check_create_ticker_data('AAPL{}'.format(k))

        strategy._post_close_orders()
        
        assert len(strategy._active_orders) == len(expected_orders)
        for order, expected in zip(strategy._active_orders, expected_orders):
            np.testing.assert_equal(order['ticker'], expected['ticker'])
            np.testing.assert_equal(order['direction'], expected['direction'])
            np.testing.assert_equal(order['size'], expected['size'])



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_path', 'step_idx', 'direction', 'value', 
         'expected_orders'],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 3, Order.BUY, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 6}]),

            ('data/step_dates1.csv', 'data/df1.csv', 3, Order.BUY, 960.,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 5}]),

            ('data/step_dates1.csv', 'data/df1.csv', 8, Order.BUY, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 5}]),

            ('data/step_dates1.csv', 'data/df1.csv', 8, Order.SELL, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.SELL, 'size': 5}]),
     
        ]
    )
    def test_post_order_value(self, step_dates_path, df_path, step_idx, 
                              direction, value, expected_orders):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        df = pd.read_csv(df_path)
        df['date'] = df['date'].astype(np.datetime64)
        data_loader = DfData(df)

        strategy = Strategy()
        strategy.latency = np.timedelta64(1, 'h')
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]
        strategy.verbose = False

        strategy.post_order_value(ticker='AAPL',
                                  direction=direction,
                                  order_type=Order.MARKET,
                                  value=value,
                                  lifetime=np.timedelta64(300, 'D'),
                                  allow_partial=True)

        
        assert len(strategy._active_orders) == len(expected_orders)
        for order, expected in zip(strategy._active_orders, expected_orders):
            np.testing.assert_equal(order['ticker'], expected['ticker'])
            np.testing.assert_equal(order['direction'], expected['direction'])
            np.testing.assert_equal(order['size'], expected['size'])



    @pytest.mark.parametrize(
        ['step_dates_path', 'step_idx', 'portfolio', 'size', 'expected_orders'],
        [
            ('data/step_dates1.csv', 4, {'AAPL': 3}, 10,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 7}]),

            ('data/step_dates1.csv', 4, {'AAPL': 0}, 10,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 10}]),
    
            ('data/step_dates1.csv', 4, {'AAPL': 10}, 4,
             [{'ticker': 'AAPL', 'direction': Order.SELL, 'size': 6}]),

            ('data/step_dates1.csv', 4, {}, 10,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 10}]),

            ('data/step_dates1.csv', 4, {'AAPL': 10}, 10,
             []),
        ]
    )
    def test_post_portfolio_size(self, step_dates_path, step_idx, portfolio, 
                                 size, expected_orders):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]

        strategy = Strategy()
        strategy.step_dates = step_dates
        strategy.step_date = strategy.step_dates[0]
        strategy.latency = np.timedelta64(1, 'h')
        strategy.portfolio = portfolio

        strategy.post_portfolio_size(ticker='AAPL',
                                     size=size,
                                     lifetime=np.timedelta64(300, 'D'),
                                     allow_partial=True)

        
        assert len(strategy._active_orders) == len(expected_orders)
        for order, expected in zip(strategy._active_orders, expected_orders):
            np.testing.assert_equal(order['ticker'], expected['ticker'])
            np.testing.assert_equal(order['direction'], expected['direction'])
            np.testing.assert_equal(order['size'], expected['size'])



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_path', 'step_idx', 'portfolio', 'value', 
         'expected_orders'],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 0}, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 6}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 0}, 20.,
             []),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 6}, 1000.,
             []),

            ('data/step_dates1.csv', 'data/df11.csv', 2, {}, 1000.,
             []),

            ('data/step_dates1.csv', 'data/df1.csv', 12, {}, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 4}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 2}, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 4}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 20}, 1000.,
             [{'ticker': 'AAPL', 'direction': Order.SELL, 'size': 14}]),

    
        ]
    )
    def test_post_portfolio_value(self, step_dates_path, df_path, step_idx, 
                                  portfolio, value, expected_orders):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        df = pd.read_csv(df_path)
        df['date'] = df['date'].astype(np.datetime64)
        data_loader = DfData(df)

        strategy = Strategy()
        strategy.latency = np.timedelta64(1, 'h')
        strategy.portfolio = portfolio
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]
        strategy.verbose = False

        strategy.post_portfolio_value(ticker='AAPL',
                                      value=value,
                                      lifetime=np.timedelta64(300, 'D'),
                                      allow_partial=True)
        
        assert len(strategy._active_orders) == len(expected_orders)
        for order, expected in zip(strategy._active_orders, expected_orders):
            np.testing.assert_equal(order['ticker'], expected['ticker'])
            np.testing.assert_equal(order['direction'], expected['direction'])
            np.testing.assert_equal(order['size'], expected['size'])



    @pytest.mark.parametrize(
        ['step_dates_path', 'df_path', 'step_idx', 'portfolio', 'step_equity', 'part', 
         'expected_orders'],
        [
            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 0}, 1000., 1.0,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 6}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 0}, 1000., 0.5,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 3}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 0}, 1000., 0.3,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 2}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 1}, 1000., 0.3,
             [{'ticker': 'AAPL', 'direction': Order.BUY, 'size': 1}]),

            ('data/step_dates1.csv', 'data/df1.csv', 4, {'AAPL': 5}, 1000., 0.3,
             [{'ticker': 'AAPL', 'direction': Order.SELL, 'size': 3}]),

    
        ]
    )
    def test_post_portfolio_part(self, step_dates_path, df_path, step_idx, 
                                 portfolio, step_equity, part, expected_orders):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        df = pd.read_csv(df_path)
        df['date'] = df['date'].astype(np.datetime64)
        data_loader = DfData(df)

        strategy = Strategy()
        strategy.latency = np.timedelta64(1, 'h')
        strategy.equity = [0] * len(step_dates)
        strategy.equity[step_idx] = step_equity
        strategy.portfolio = portfolio
        strategy.data_loader = data_loader
        strategy.step_dates = step_dates
        strategy.date_col = 'date'
        strategy.price_col = 'price'
        strategy.return_col = 'return'
        strategy.return_format = 'ratio'
        strategy.step_idx = step_idx
        strategy.step_date = strategy.step_dates[strategy.step_idx]
        strategy.verbose = False

        strategy.post_portfolio_part(ticker='AAPL',
                                     part=part,
                                     lifetime=np.timedelta64(300, 'D'),
                                     allow_partial=True)
        
        assert len(strategy._active_orders) == len(expected_orders)
        for order, expected in zip(strategy._active_orders, expected_orders):
            np.testing.assert_equal(order['ticker'], expected['ticker'])
            np.testing.assert_equal(order['direction'], expected['direction'])
            np.testing.assert_equal(order['size'], expected['size'])




    class Strategy1(Strategy):        
        def step(self):        
            if self.step_idx == 0:
                self.post_order(ticker='AAPL0',
                                size=10,
                                direction=Order.BUY,
                                lifetime=np.timedelta64(3, 'D'),
                                allow_partial=True)

                self.post_order(ticker='AAPL1',
                                size=5,
                                direction=Order.BUY,
                                lifetime=np.timedelta64(3, 'D'),
                                allow_partial=True)

            if self.step_idx == 7:
                self.post_portfolio_part(ticker='AAPL1',
                           part=0.99,
                           allow_partial=True)    

            if self.step_idx == 16:
                self.post_order(ticker='AAPL0',
                                size=20,
                                direction=Order.SELL,
                                lifetime=np.timedelta64(3, 'D'),
                                allow_partial=True)        

    @pytest.mark.parametrize(
        ['step_dates_path', 'df_pathes', 'strategy', 'cash', 'comission', 'latency',
         'expected_orders', 'expected_equity', 'expected_cash', 'expected_portfolio'],
        [
            ('data/step_dates1.csv', ['data/df6.csv', 'data/df3.csv'], Strategy1(),
             500., 0., np.timedelta64(1, 'h'),
             [{'ticker': 'AAPL0', 'direction': Order.BUY, 'size': 10, 
               'creation_date': np.datetime64('2015-02-18'),
               'execution_date': np.datetime64('2015-02-20'), 
               'status': Order.COMPLETED, 'price': 16.0},
              
              {'ticker': 'AAPL1', 'direction': Order.BUY, 'size': 5, 
               'creation_date': np.datetime64('2015-02-18'),
               'execution_date': np.nan, 
               'status': Order.EXPIRED, 'price': np.nan},

              {'ticker': 'AAPL1', 'direction': Order.BUY, 'size': 28, 
               'creation_date': np.datetime64('2015-03-05'),
               'execution_date': np.datetime64('2015-03-12'), 
               'status': Order.COMPLETED, 'price': 12.3},

              {'ticker': 'AAPL1', 'direction': Order.SELL, 'size': 28, 
               'creation_date': np.datetime64('2015-03-17'),
               'execution_date': np.datetime64('2015-03-18'), 
               'status': Order.COMPLETED, 'price': 19.3},

              {'ticker': 'AAPL0', 'direction': Order.SELL, 'size': 10, 
               'creation_date': np.datetime64('2015-03-21'),
               'execution_date': np.datetime64('2015-03-23'), 
               'status': Order.PARTIAL, 'price': 19.6},
             ],
             [500. , 500. , 455. , 498. , 498. , 492.8, 477.8, 496.8, 500.8, 500.8,
              540.8, 500.8, 500.8, 500.8, 696.8, 737.8, 719.1, 769.1, 769.1, 769.1],
             [500, 340.0, 340.0, 340.0, 340.0, 358.8, 358.8, 358.8, 358.8, 358.8,
              14.4, 14.4, 14.4, 14.4, 554.8, 554.8, 573.1, 769.1, 769.1, 769.1],
             {'AAPL0': 0, 'AAPL1': 0}
             ),


            ('data/step_dates1.csv', ['data/df6.csv', 'data/df3.csv'], Strategy1(),
             100., 0., np.timedelta64(49, 'h'),
             [{'ticker': 'AAPL0', 'direction': Order.BUY, 'size': 8, 
               'creation_date': np.datetime64('2015-02-18'),
               'execution_date': np.datetime64('2015-02-23'), 
               'status': Order.PARTIAL, 'price': 11.5},
              
              {'ticker': 'AAPL1', 'direction': Order.BUY, 'size': 5, 
               'creation_date': np.datetime64('2015-02-18'),
               'execution_date': np.nan, 
               'status': Order.EXPIRED, 'price': np.nan},

              {'ticker': 'AAPL1', 'direction': Order.BUY, 'size': 1, 
               'creation_date': np.datetime64('2015-03-05'),
               'execution_date': np.datetime64('2015-03-12'), 
               'status': Order.COMPLETED, 'price': 12.3},

              {'ticker': 'AAPL1', 'direction': Order.SELL, 'size': 1, 
               'creation_date': np.datetime64('2015-03-17'),
               'execution_date': np.datetime64('2015-03-18'), 
               'status': Order.COMPLETED, 'price': 19.3},

              {'ticker': 'AAPL0', 'direction': Order.SELL, 'size': 8, 
               'creation_date': np.datetime64('2015-03-21'),
               'execution_date': np.datetime64('2015-03-24'), 
               'status': Order.PARTIAL, 'price': 20.},
             ],
             [100., 100., 100., 134.4, 134.4, 130.24, 118.24, 133.44,
              136.64, 136.64, 168.64, 136.64, 136.64, 136.64, 143.64, 176.44,
              161.48, 201.48, 204.68, 204.68],
             [100., 100., 8., 8., 8., 23.04, 23.04, 23.04,
              23.04, 23.04, 10.74, 10.74, 10.74, 10.74, 30.04, 30.04,
              44.68, 44.68, 204.68, 204.68],
             {'AAPL0': 0, 'AAPL1': 0}
             ),

        ]
    )
    def test_backtest_simple_strategy(self, step_dates_path, df_pathes, strategy,
            cash, comission, latency, expected_orders, expected_equity, 
            expected_cash, expected_portfolio):
        step_dates = [np.datetime64(x) for x in 
                                        pd.read_csv(step_dates_path)['date']]
        dfs = {}
        for k, df_path in enumerate(df_pathes):
            df = pd.read_csv(df_path)
            df['date'] = df['date'].astype(np.datetime64)
            dfs['AAPL{}'.format(k)] = df
        data_loader = DfData(dfs)
        
        strategy.backtest(data_loader=data_loader,
                          date_col='date',
                          price_col='price',
                          return_col='return',
                          return_format='ratio',
                          step_dates=step_dates,
                          cash=cash,
                          comission=comission,
                          latency=latency)

        assert len(strategy.orders) == len(expected_orders)
        for order, expected in zip(strategy._active_orders, expected_orders):
            np.testing.assert_equal(order['ticker'], expected['ticker'])
            np.testing.assert_equal(order['direction'], expected['direction'])
            np.testing.assert_equal(order['size'], expected['size'])
            np.testing.assert_almost_equal(order['price'], expected['price'])
            np.testing.assert_equal(order['creation_date'], expected['creation_date'])
            np.testing.assert_equal(order['execution_date'], expected['execution_date'])


        np.testing.assert_array_almost_equal(strategy.equity, expected_equity)
        np.testing.assert_array_almost_equal(strategy.cash, expected_cash)

        assert strategy.portfolio == expected_portfolio







