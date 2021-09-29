import pytest
import pandas as pd
import numpy as np
import os
from ml_investment.data_loaders.sf1 import SF1BaseData, SF1QuarterlyData,\
                                           SF1DailyData, translate_currency
from ml_investment.data_loaders.mongo import SF1BaseData as SF1BaseDataMongo,\
                        SF1QuarterlyData as SF1QuarterlyDataMongo,\
                        SF1DailyData as SF1DailyDataMongo,\
                        QuandlCommoditiesData as QuandlCommoditiesDataMongo,\
                        DailyBarsData as DailyBarsDataMongo

from ml_investment.data_loaders.yahoo import YahooBaseData, YahooQuarterlyData
from ml_investment.data_loaders.quandl_commodities import QuandlCommoditiesData
from ml_investment.data_loaders.daily_bars import DailyBarsData
from ml_investment.utils import load_json, load_config, load_secrets
from pymongo import MongoClient


config = load_config()
secrets = load_secrets()



class TestSF1BaseData:
    datas = []
    if os.path.exists(config['sf1_data_path']):
        datas.append(SF1BaseData())
    if secrets['mongodb_adminusername'] is not None:
        datas.append(SF1BaseDataMongo())

    @pytest.mark.parametrize('data_loader', datas)
    def test_load_no_param(self, data_loader):
        df = data_loader.load()
        assert type(df) == pd.DataFrame
        assert len(df) > 0
        assert 'ticker' in df.columns
        assert df['ticker'].isnull().max() == False

    @pytest.mark.parametrize('data_loader', datas)
    @pytest.mark.parametrize(
        ["tickers"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'],),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'],)]
    )
    def test_load(self, tickers, data_loader):
        df = data_loader.load(tickers)
        assert type(df) == pd.DataFrame
        assert len(df) > 0
        assert 'ticker' in df.columns
        assert len(set(df['ticker'].values).difference(set(tickers))) == 0



class TestSF1QuarterlyData:
    data_classes = []
    if os.path.exists(config['sf1_data_path']):
        data_classes.append(SF1QuarterlyData)
    if secrets['mongodb_adminusername'] is not None:
        data_classes.append(SF1QuarterlyDataMongo)

    @pytest.mark.parametrize('data_loader_class', data_classes)
    @pytest.mark.parametrize(
        ["tickers", "quarter_count", "dimension"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'], 10, 'ARQ'),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 5, 'ARQ'),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 5, 'ARQ'),
         (['NVDA'], 10, 'ARQ'),
         (['ZRAN'], 10, 'ARQ')],
    )
    def test_load(self, tickers, quarter_count, dimension, data_loader_class):
        data_loader = data_loader_class(quarter_count=quarter_count,
                                       dimension=dimension)
        quarterly_df = data_loader.load(tickers)
        
        assert type(quarterly_df) == pd.DataFrame
        assert 'ticker' in quarterly_df.columns
        assert 'date' in quarterly_df.columns
        
        # Data should be ordered by date inside ticker
        quarterly_df['date_'] = quarterly_df['date'].astype(np.datetime64)
        quarterly_df['def_order'] = range(len(quarterly_df))[::-1]
        expected_dates_order = quarterly_df.sort_values(['ticker', 'date_'],
                                            ascending=False)['date'].values
        real_dates_order = quarterly_df.sort_values(['ticker', 'def_order'], 
                                            ascending=False)['date'].values          
        np.testing.assert_array_equal(expected_dates_order, real_dates_order)
                             
        for cnt in quarterly_df.groupby('ticker').size():
            assert cnt <= quarter_count
                        
        assert (quarterly_df['dimension'] == dimension).min()     
              




class TestSF1DailyData:
    data_classes = []
    if os.path.exists(config['sf1_data_path']):
        data_classes.append(SF1DailyData)
    if secrets['mongodb_adminusername'] is not None:
        data_classes.append(SF1DailyDataMongo)

    @pytest.mark.parametrize('data_loader_class', data_classes)
    @pytest.mark.parametrize(
        ["tickers", "days_count"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'], 100),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 50),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], None),
         (['NVDA'], 100),
         (['ZRAN'], 10)],
    )    
    def test_load(self, tickers, days_count, data_loader_class):
        data_loader = data_loader_class(days_count=days_count)
        daily_df = data_loader.load(tickers)  
        assert type(daily_df) == pd.DataFrame
        assert 'ticker' in daily_df.columns
        assert 'date' in daily_df.columns
           
        # Data should be ordered by date inside ticker
        daily_df['date_'] = daily_df['date'].astype(np.datetime64)
        daily_df['def_order'] = range(len(daily_df))[::-1]
        expected_dates_order = daily_df.sort_values(['ticker', 'date_'],
                                            ascending=False)['date'].values
        real_dates_order = daily_df.sort_values(['ticker', 'def_order'], 
                                            ascending=False)['date'].values
        np.testing.assert_array_equal(expected_dates_order, real_dates_order)

        # Should not be large holes in date
        diffs = daily_df.groupby('ticker')['date_'].shift(1) - daily_df['date_']
        assert (diffs.dropna() <= np.timedelta64(14,'D')).min()
        
        if days_count is not None:
            for cnt in daily_df.groupby('ticker').size():
                assert cnt == days_count



@pytest.mark.skipif(not os.path.exists(config['sf1_data_path']), 
                    reason="There are no SF1 dataset")
class TestSF1TranslateCurrency:
    @pytest.mark.parametrize("cnt", [1, 3, 5, 10, 100])       
    def test_translate_currency_synthetic(self, cnt):
        np.random.seed(0)
        currency_arr = np.array(range(1, cnt + 1))
        df = pd.DataFrame()
        df['debtusd'] = np.random.uniform(-1e5, 1e5, cnt) 
        df['debt'] = df['debtusd'] * currency_arr
        df['ebitusd'] = np.random.uniform(-10, 10, cnt)
        noise = np.random.uniform(-0.1, 0.1, cnt) 
        df['ebit'] = df['ebitusd'] * (currency_arr + noise)
        del_proba = np.random.uniform(0, 0.3)
        drop_mask = np.random.choice([True, False], cnt, 
                                     p=[del_proba, 1 - del_proba])
        df.loc[drop_mask, 'ebitusd'] = None

        trans_df = translate_currency(df, ['debt', 'ebit'])
        for col in ['debt', 'ebit']:
            diff = trans_df['{}usd'.format(col)] - trans_df[col]
            diff = np.abs(diff.values / trans_df['{}usd'.format(col)].values)
            diff = diff[~np.isnan(diff)]
            assert diff.max() < 0.1


    @pytest.mark.parametrize("ticker", ['YNDX', 'NIO'])       
    def test_translate_currency_real(self, ticker):
        columns = ['equity','eps','revenue','netinccmn',
                    'cashneq','debt','ebit','ebitda']
        data_loader = SF1QuarterlyData(config['sf1_data_path'])
        quarterly_df = data_loader.load(ticker)
        trans_df = translate_currency(quarterly_df, columns)
        for col in columns:
            diff = trans_df['{}usd'.format(col)] - trans_df[col]
            diff = np.abs(diff.values / trans_df['{}usd'.format(col)].values)
            diff = diff[~np.isnan(diff)]
            assert diff.max() < 0.1

 


@pytest.mark.skipif(not os.path.exists(config['yahoo_data_path']), 
                    reason="There are no Yahoo dataset")
class TestYahooBaseData:
    def test_load_no_param(self):
        data_loader = YahooBaseData(config['yahoo_data_path'])
        df = data_loader.load()
        assert type(df) == pd.DataFrame
        assert len(df) > 0
        assert 'ticker' in df.columns
        assert df['ticker'].isnull().max() == False

    @pytest.mark.parametrize(
        ["tickers"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'],),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'],)]
    )
    def test_load(self, tickers):
        data_loader = YahooBaseData(config['yahoo_data_path'])
        df = data_loader.load(tickers)
        assert type(df) == pd.DataFrame
        assert len(df) > 0
        assert 'ticker' in df.columns
        assert len(set(df['ticker'].values).difference(set(tickers))) == 0



@pytest.mark.skipif(not os.path.exists(config['yahoo_data_path']), 
                    reason="There are no Yahoo dataset")
class TestYahooQuarterlyData:
    @pytest.mark.parametrize(
        ["tickers", "quarter_count"],
        [(['AAPL', 'MSFT', 'TSLA', 'WORK'], 3),
         (['INTC', 'MSFT', 'XRDC', 'XOM'], 3),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 2),
         (['NVDA'], 4)]
    )
    def test_load_quarterly_data(self, tickers, quarter_count):
        data_loader = YahooQuarterlyData(config['yahoo_data_path'], quarter_count=quarter_count)
        quarterly_df = data_loader.load(tickers)

        assert type(quarterly_df) == pd.DataFrame
        assert 'ticker' in quarterly_df.columns
        assert 'date' in quarterly_df.columns

        # Data should be ordered by date inside ticker
        quarterly_df['date_'] = quarterly_df['date'].astype(np.datetime64)
        quarterly_df['def_order'] = range(len(quarterly_df))[::-1]
        expected_dates_order = quarterly_df.sort_values(['ticker', 'date_'],
                                            ascending=False)['date'].values
        real_dates_order = quarterly_df.sort_values(['ticker', 'def_order'], 
                                            ascending=False)['date'].values          
        np.testing.assert_array_equal(expected_dates_order, real_dates_order)

        for cnt in quarterly_df.groupby('ticker').size():
            assert cnt <= quarter_count




class TestQuandlCommoditiesData:
    data_classes = []
    if os.path.exists(config['commodities_data_path']):
        data_classes.append(QuandlCommoditiesData)
    if secrets['mongodb_adminusername'] is not None:
        data_classes.append(QuandlCommoditiesDataMongo)

    @pytest.mark.parametrize('data_loader_class', data_classes)
    @pytest.mark.parametrize(
        ["commodities_codes"],
        [(['LBMA/GOLD', 'CHRIS/CME_CL1'], ),
         (['LBMA/GOLD'], )]
    )
    def test_load_commodities_data(self, commodities_codes, data_loader_class):
        data_loader = data_loader_class()
        commodities_codes = [x.replace('/', '_') for x in commodities_codes]
        df = data_loader.load(commodities_codes)
        assert type(df) == pd.DataFrame
        assert len(df) > 0
        assert 'commodity_code' in df.columns
        assert df['commodity_code'].isnull().max() == False
        assert len(set(df['commodity_code'].values).difference(set(commodities_codes))) == 0



class TestDailyBarsData:
    data_classes = []
    if os.path.exists(config['daily_bars_data_path']):
        data_classes.append(DailyBarsData)
    if secrets['mongodb_adminusername'] is not None:
        data_classes.append(DailyBarsDataMongo)

    @pytest.mark.parametrize('data_loader_class', data_classes)
    @pytest.mark.parametrize(
        ["tickers", "days_count"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'], 100),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 50),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], None),
         (['NVDA'], 100)],
    )    
    def test_load(self, tickers, days_count, data_loader_class):
        data_loader = data_loader_class(days_count=days_count)
        daily_df = data_loader.load(tickers)  
        assert type(daily_df) == pd.DataFrame
        assert 'ticker' in daily_df.columns
        assert 'date' in daily_df.columns
           
        # Data should be ordered by date inside ticker
        daily_df['date_'] = daily_df['date'].astype(np.datetime64)
        daily_df['def_order'] = range(len(daily_df))[::-1]
        expected_dates_order = daily_df.sort_values(['ticker', 'date_'],
                                            ascending=False)['date'].values
        real_dates_order = daily_df.sort_values(['ticker', 'def_order'], 
                                            ascending=False)['date'].values
        np.testing.assert_array_equal(expected_dates_order, real_dates_order)

        # Should not be large holes in date
        diffs = daily_df.groupby('ticker')['date_'].shift(1) - daily_df['date_']
        assert (diffs.dropna() <= np.timedelta64(14,'D')).min()
        
        if days_count is not None:
            for cnt in daily_df.groupby('ticker').size():
                assert cnt == days_count






