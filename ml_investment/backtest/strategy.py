import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict


class Order:
    BUY = 1
    SELL = -1
       
    MARKET = 0
    LIMIT = 1
    CLOSE = 2

    INIT = 0
    PARTIAL = 1
    COMPLETED = 2
    EXPIRED = 3
    REJECTED = 4



class Strategy:
    '''
    Base class for strategy backtesting. 
    It contains overrideble method ``step`` for defining user strategy.
    This class incapsulate backtesting and metrics calculation process and 
    also contains information about orders.
    '''
    def __init__(self):
        self.data_loader = None
        self._data = {}
        self._cash = 0
        self._active_orders = []
        self._data_step_idxs = {}

        self.portfolio = {}
        self.step_dates:List[np.datetime64] = None
        self.step_date = None
        self.step_idx = None
        self.orders = []
              
        self.cash: List = []
        self.returns: List[float] = []
        self.equity: List[float] = []
        
        self.metrics:Dict = {}
        
    
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
        

        if order['submit_date'] >= execution_date: 
            self._active_orders.append(order)
            return

        if order['submit_date'] + order['lifetime'] < execution_date:
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



    def _execute_close_order(self, order):
        idx = self.step_idx + 1
        execution_date = np.datetime64(
                                self._data[order['ticker']].loc[idx, 'date'])
        

        price = self._data[order['ticker']].loc[idx, 'price']
        self.portfolio[order['ticker']] = 0
        self._cash -= order['direction'] * price *\
                      order['size'] * (1 + self.comission)
        
        order['execution_date'] = execution_date
        order['price'] = price
        
        order['status'] = Order.COMPLETED
            
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

            if order['order_type'] == Order.CLOSE:            
                self._execute_close_order(order)

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
                                order_type=Order.CLOSE,
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
                   order_type: int=Order.MARKET,
                   lifetime: np.timedelta64=np.timedelta64(300, 'D'),
                   allow_partial: bool=True):
        '''     
        Post new order to backtest. 
        It may be used inside your strategy overriden ``step`` method.
        
        Parameters
        ----------
        ticker:
            ticker of company to post order for
        direction:
            one of ``Order.BUY`` (1), ``Order.SELL`` (-1)
        size:
            size of order in pieces
        order_type:
            one of ``Order.MARKET`` (0), ``Order.LIMIT`` (1)
        lifetime:
            amount of time before order closing if it can not be executed 
            (e.g. if unsatisfactory price lasts a long time)
        allow_partial:
            may order be executed with not full size or not
        '''
        if size == 0:
            return

        submit_date = self.step_date + self.latency
        if order_type == Order.CLOSE:
            submit_date = self.step_date

        self._active_orders.append({'ticker': ticker,
                                    'direction': direction,
                                    'size': size,
                                    'order_type': order_type,
                                    'lifetime': lifetime,
                                    'allow_partial': allow_partial,
                                    'creation_date': self.step_date,
                                    'submit_date': submit_date})


    def post_order_value(self,
                         ticker: str,
                         direction: int,
                         value: float,
                         order_type: int=Order.MARKET,
                         lifetime: np.timedelta64=np.timedelta64(300, 'D'),
                         allow_partial: bool=True):
        '''     
        Post new order by value (instead of size) to backtest.
        It may be used inside your strategy overriden ``step`` method.
        
        Parameters
        ----------
        ticker:
            ticker of company to post order for
        direction:
            one of ``Order.BUY`` (1), ``Order.SELL`` (-1)
        value:
            value of order in money
        order_type:
            one of ``Order.MARKET`` (0), ``Order.LIMIT`` (1)
        lifetime:
            amount of time before order closing if it can not be executed 
            (e.g. if unsatisfactory price lasts a long time)
        allow_partial:
            may order be executed with not full size or not
        '''
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
                            ticker: str,
                            size: int,
                            lifetime: np.timedelta64=np.timedelta64(300, 'D'),
                            allow_partial: bool=True):
        '''     
        Post order to backtest to have desired size in portfolio. 
        It will calculate difference between current and desired size 
        to create appropriate order.
        It may be used inside your strategy overriden ``step`` method.
        
        Parameters
        ----------
        ticker:
            ticker of company to post order for
        size:
            desired size in portfolio (in pieces)
        lifetime:
            amount of time before order closing if it can not be executed 
            (e.g. if unsatisfactory price lasts a long time)
        allow_partial:
            may order be executed with not full size or not
        '''
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
                             ticker: str,
                             value: float,
                             lifetime: np.timedelta64=np.timedelta64(300, 'D'),
                             allow_partial: bool=True):
        '''     
        Post order to backtest to have desired value in portfolio. 
        It will calculate difference between current and desired value
        to create appropriate order.
        It may be used inside your strategy overriden ``step`` method.
        
        Parameters
        ----------
        ticker:
            ticker of company to post order for
        value:
            desired value in portfolio (in money)
        lifetime:
            amount of time before order closing if it can not be executed 
            (e.g. if unsatisfactory price lasts a long time)
        allow_partial:
            may order be executed with not full size or not
        '''
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
                            ticker: str,
                            part: float,
                            lifetime: np.timedelta64=np.timedelta64(300, 'D'),
                            allow_partial: bool=True):
        '''     
        Post order to backtest to have desired part in portfolio. 
        It will calculate difference between current and desired part
        to create appropriate order.
        It may be used inside your strategy overriden ``step`` method.
        
        Parameters
        ----------
        ticker:
            ticker of company to post order for
        part:
            desired part in all equity including other stocks and cash
            in portfolio (value between 0 and 1)
        lifetime:
            amount of time before order closing if it can not be executed 
            (e.g. if unsatisfactory price lasts a long time)
        allow_partial:
            may order be executed with not full size or not
        '''

        needed_value = self.equity[self.step_idx] * part
        self.post_portfolio_value(ticker=ticker,
                                  value=needed_value,
                                  lifetime=lifetime,
                                  allow_partial=allow_partial)

    
    def calc_metrics(self, metrics:Dict):
        for key in metrics.keys():
            self.metrics[key] = metrics[key](self.step_dates, self.returns)
        
        
    def step(self):
        None
        
        
    def backtest(self,
                 data_loader,
                 date_col: str,
                 price_col: str,
                 return_col: str,
                 return_format: str,
                 step_dates: List[np.timedelta64]=None,
                 cash: float=100_000,
                 comission: float=0.00025,
                 latency: np.timedelta64=np.timedelta64(0, 'h'),
                 allow_short: bool=False,
                 metrics=None,
                 preload: bool=False,
                 verbose: bool=True):
        '''
        Backtest strategy on provided data and other parameters. 
        It will create and execute orders and calculate 
        resulted equity and metrics.

        Parameters
        ----------
        data_loader:
            class implementing
            ``load(index) -> pd.DataFrame`` interface.
            index in this case is list of tickers to load market data for.
        date_col:
            name of column containing date (time) information in market data
            provided by ``data_loader``.
        price_col:
            name of column containing price information in market data
            provided by ``data_loader``.
        return_col:
            name of column containing total return information in data
            provided by ``data_loader``. It may be differ from price due to
            dividends, stock splits and etc.
        return_format:
            format of data provided by ``return_col`` column.
            If ``return_format = 'ratio'`` than column should contain
            ratio between previous and current adjusted price. 
            E.g. 1.2 means growth by 20% from the previous step.
            If ``return_format = 'price'`` than column should contain
            adjusted price (price, including dividends and etc.)
            If ``return_format = 'change'`` than column should contain
            relative change between current and previous step.
            E.g. 0.2 means growth by 20% from the previous step.
        step_dates:
            dates in which all actions can be taken. 
            Include new market prices receiving, order creation and executing.
            ``step`` method will iterate over all those dates.
            If None than all possible dates, provided by ``date_col`` column in
            ``data_loader`` will be used. Possible only if ``preload = True``
            and ``data_loader`` have
            ``existing_index(index) -> List`` interface.
        cash:
            initial amount of cash
        comission:
            commission charged for each trade (in percent of order value)
        latency:
            time between current step date and actual order posting.
            It emulates delays during ``step`` logic and in the Internet
            connection with the exchange. 
        allow_short:
            allow short positions or not
        preload:
            load all data provided from ``data_loader`` to ram or not
        verbose:
            show progress or not
        '''
        if allow_short:
            raise NotImplementedError
        
        if preload:
            raise NotImplementedError

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
        self.returns = self.equity[1:] / self.equity[:-1] - 1
        self.returns = np.insert(self.returns, 0, 0., axis=0)

        if metrics is not None:
            self.calc_metrics(metrics)
        







