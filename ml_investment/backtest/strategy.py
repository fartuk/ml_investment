import numpy as np
import pandas as pd
from tqdm import tqdm



class Order:
    BUY = 1
    SELL = -1
       
    MARKET = 0
    LIMIT = 1
        
    INIT = 0
    PARTIAL = 1
    COMPLETED = 2
    EXPIRED = 3
    REJECTED = 4



class Strategy:
    def __init__(self):
        self.data_loader = None
        self._data = {}
        self._cash = 0
        self._active_orders = []
        self._data_step_idxs = {}

        self.portfolio = {}
        self.step_dates = None
        self.step_date = None
        self.step_idx = None
        self.orders = []
              
        self.cash = []
        self.returns = []
        self.equity = []
        
        self.metrics = {}
        
    
    def _cast_data(self, df):
        date_df = pd.DataFrame()
        date_df['date'] = self.step_dates
        date_df['base_date'] = date_df['date']

        df = pd.merge_asof(df, date_df, on='date', direction='forward')
        df = df.groupby('base_date').agg({'price': 'last',
                                          'return': np.prod})
        df = df.reset_index().rename({'base_date': 'date'}, axis=1)
        df = pd.merge(date_df[['date']], df, on='date', how='left')

        return df


    def _check_create_ticker_data(self, ticker):
        if ticker not in self._data:
            df = self.data_loader.load([ticker])
            df[self.date_col] = df[self.date_col].astype(np.datetime64)
            df = df.sort_values(self.date_col)
            df.index = range(len(df))
            
            result = pd.DataFrame()
            result['date'] = df[self.date_col]
            result['price'] = df[self.price_col]

            if self.return_col is None:
                result['return'] = df[self.price_col] /\
                                   df[self.price_col].shift(1)

            if self.return_format == 'price':
                result['return'] = (df[self.return_col] /\
                                    df[self.return_col].shift(1)).fillna(1)

            if self.return_format == 'change':
                result['return'] = df[self.return_col] + 1  

            if self.return_format == 'ratio':
                result['return'] = df[self.return_col]
            
            result = self._cast_data(result)
            # If len of data larger than len of step dates, shuoldn't use 
            # older values as cumulative returns
            result.loc[0, 'return'] = 1

            result['missed'] = result['price'].isnull()
            # If last date of data less then some step_date
            result['closed'] = result['missed'][::-1].cumprod()[::-1]
            result['price'] = result['price'].ffill()
            
            result['price_return'] = (result['price'] /\
                                      result['price'].shift(1)).fillna(1)
            result['prev_price'] = result['price'].shift(1)
            result['dividend'] = result['return'] - result['price_return']
            result['dividend'] *= result['dividend'].abs() > 1e-5
            #result['missed'] = result['missed'].ffill()

            self._data[ticker] = result


    def _aposteriori_next_step_max_size(self, order):
        idx = self.step_idx + 1
        if len(self._data[order['ticker']]) <= idx:
            return
        
        missed = self._data[order['ticker']].loc[idx, 'missed']
        if missed:
            return

        price = self._data[order['ticker']].loc[idx, 'price']
        size = self.portfolio[order['ticker']]
        
        if order['direction'] == Order.BUY:
            result = self._cash // (price * (1 + self.comission))

        if order['direction'] == Order.SELL:
            result = size

        return result


    def _execute_market_order(self, order):
        idx = self.step_idx + 1
        if idx >= len(self.step_dates):
            order['status'] = Order.EXPIRED
            order['price'] = np.nan
            order['execution_date'] = np.nan
            self.orders.append(order)
            return

        execution_date = np.datetime64(
                            self._data[order['ticker']].loc[idx, 'date'])
        
        if order['creation_date'] + order['lifetime'] < execution_date:
            order['status'] = Order.EXPIRED
            order['price'] = np.nan
            order['execution_date'] = np.nan
            self.orders.append(order)
            return

        if self._data[order['ticker']].loc[idx, 'missed']:
            self._active_orders.append(order)
            return
 
        price = self._data[order['ticker']].loc[idx, 'price']
        possible_size = self._aposteriori_next_step_max_size(order)
        if possible_size == 0 or possible_size is None:
            self._active_orders.append(order)
            return

        if not order['allow_partial'] and order['size'] > possible_size:
            self._active_orders.append(order)
            return

        execution_size = min(order['size'], possible_size)
        self.portfolio[order['ticker']] += order['direction'] * execution_size
        self._cash -= order['direction'] * price *\
                      execution_size * (1 + self.comission)
        
        order['execution_date'] = execution_date
        order['price'] = price
        
        if execution_size == order['size']:
            order['status'] = Order.COMPLETED
        else:
            order['size'] = possible_size
            order['status'] = Order.PARTIAL
            
        self.orders.append(order)

        
    def _execute_orders(self):
        curr_orders = self._active_orders.copy()
        curr_orders = sorted(curr_orders, key=lambda x: x['direction'])
        self._active_orders = []
        for order in curr_orders:
            self._check_create_ticker_data(order['ticker'])
            
            if order['ticker'] not in self.portfolio:
                self.portfolio[order['ticker']] = 0
                
            if order['order_type'] == Order.MARKET:            
                self._execute_market_order(order)

            if order['order_type'] == Order.LIMIT:            
                raise NotImplementedError
               

    def _post_close_orders(self):
        if self.step_idx >= len(self.step_dates) - 2:
            return

        for ticker in self.portfolio.keys():
            size = self.portfolio[ticker]
            if size == 0:
                continue

            closed = self._data[ticker].loc[self.step_idx + 2, 'closed']
            if closed:
                if self.verbose:
                    print('Close ticker {}'.format(ticker))

                direction = Order.SELL if size > 0 else Order.BUY
                size = abs(size)
                self.post_order(ticker=ticker,
                                direction=direction,
                                size=size,
                                order_type=Order.MARKET,
                                lifetime=np.timedelta64(300, 'D'),
                                allow_partial=False)


    def _receive_dividends(self):
        for ticker in self.portfolio.keys():
            size = self.portfolio[ticker]
            if size == 0:
                continue

            dividend = self._data[ticker].loc[self.step_idx, 'dividend']
            if (not np.isnan(dividend)) and dividend != 0:
                prev_price = self._data[ticker].loc[self.step_idx, 'prev_price']
                self._cash += size * prev_price * dividend


    def _calc_equity(self):
        equity = 0
        for ticker in self.portfolio.keys():
            size = self.portfolio[ticker]
            if size == 0:
                continue

            price = self._data[ticker].loc[self.step_idx, 'price']
            equity += size * price
        
        equity += self._cash

        return equity


    def post_order(self,
                   ticker: str,
                   direction: int,
                   size: float,
                   order_type: int,
                   lifetime=None,
                   allow_partial=True):
        if size == 0:
            return
        self._active_orders.append({'ticker': ticker,
                                    'direction': direction,
                                    'size': size,
                                    'order_type': order_type,
                                    'lifetime': lifetime,
                                    'allow_partial': allow_partial,
                                    'creation_date': self.step_date})


    def post_order_value(self,
                         ticker,
                         direction,
                         order_type,
                         value,
                         lifetime,
                         allow_partial):
        self._check_create_ticker_data(ticker)
        price = self._data[ticker].loc[self.step_idx, 'price']

        size = round(value / price)
        self.post_order(ticker=ticker,
                        direction=direction,
                        size=size,
                        order_type=order_type,
                        lifetime=lifetime,
                        allow_partial=allow_partial)

       
    def post_portfolio_size(self,
                            ticker,
                            size,
                            lifetime,
                            allow_partial=True):
        if ticker not in self.portfolio:
            self.portfolio[ticker] = 0

        diff_size = size - self.portfolio[ticker] 
        if diff_size == 0:
            return

        direction = Order.BUY if diff_size > 0 else Order.SELL
        diff_size = abs(diff_size)
        self.post_order(ticker=ticker,
                        direction=direction,
                        size=diff_size,
                        order_type=Order.MARKET,
                        lifetime=lifetime,
                        allow_partial=allow_partial)

        
    def post_portfolio_value(self,
                             ticker,
                             value,
                             lifetime,
                             allow_partial):
        self._check_create_ticker_data(ticker)
        price = self._data[ticker].loc[self.step_idx, 'price']
        if np.isnan(price): 
            if self.verbose:
                print("There are no price for {} yet".format(ticker))
            return

        needed_size = round(value / price)
        self.post_portfolio_size(ticker=ticker,
                                 size=needed_size,
                                 lifetime=lifetime,
                                 allow_partial=allow_partial)


    def post_portfolio_part(self,
                            ticker,
                            part,
                            lifetime,
                            allow_partial):
        needed_value = self.equity[self.step_idx] * part
        self.post_portfolio_value(ticker=ticker,
                                  value=needed_value,
                                  lifetime=lifetime,
                                  allow_partial=allow_partial)




        
    def calc_metrics(self):
        {'annualized_return': 0.0030997070195237786,
         'max_drawdown': -0.14835668465595964,
         'total_return': 0.029301277098099817,
         'sharpe_ratio': 0.09598589832122632,
         'sortino_ratio': 0.1387087978643872,
         'beta': 0.0178287716829249,
         'alpha': 0.0038429722177439896}
        
        
        
    def step(self):
        None
        
        
    def backtest(self,
                 data_loader=None,
                 date_col=None,
                 price_col=None,
                 return_col=None,
                 return_format=None,
                 step_dates=None,
                 cash=100_000,
                 comission=0.00025,
                 latency=0,
                 allow_short=False,
                 metrics=None,
                 verbose=True,
                 preload=False):
        '''
        Backtesting
        '''
        self.data_loader = data_loader
        self.date_col = date_col
        self.price_col = price_col
        self.return_col = return_col
        self.return_format = return_format
        self.step_dates = step_dates
        self._cash = cash
        self.comission = comission
        self.latency = latency
        self.allow_short = allow_short
        self.verbose = verbose

        for self.step_idx, self.step_date in tqdm(enumerate(self.step_dates),
                                                  disable=not self.verbose):
            self._receive_dividends()
            self.equity.append(self._calc_equity())
            self.cash.append(self._cash)

            self.step()
            self._post_close_orders()
            self._execute_orders()

        self.equity = np.array(self.equity)
        self.returns = self.equity[1:] / self.equity[:-1]
        self.returns = np.insert(self.returns, 0, 1., axis=0)

#         calc_metrics()
        






