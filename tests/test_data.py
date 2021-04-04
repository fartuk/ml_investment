import pytest
import pandas as pd
import numpy as np
from ml_investment.data import SF1Data, ComboData
from ml_investment.utils import load_json
config = load_json('config.json')


@pytest.mark.skipif(config['sf1_data_path'] is None, reason="There are no SF1 dataset")
class TestSF1Data:
    def test_load_base_data(self):
        data_loader = SF1Data(config['sf1_data_path'])
        df = data_loader.load_base_data()
        assert type(df) == pd.DataFrame
        assert len(df) > 0
        assert 'ticker' in df.columns
        assert df['ticker'].isnull().max() == False


    @pytest.mark.parametrize(
        ["tickers", "quarter_count", "dimension"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'], 10, 'ARQ'),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 5, 'ARQ'),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 5, 'MRY'),
         (['NVDA'], 10, 'ARQ'),
         (['ZRAN'], 10, 'ARQ')],
    )
    def test_load_quarterly_data(self, tickers, quarter_count, dimension):
        data_loader = SF1Data(config['sf1_data_path'])
        quarterly_df = data_loader.load_quarterly_data(tickers, quarter_count,
                                                       dimension)
        
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
              
                    
    @pytest.mark.parametrize(
        ["tickers", "back_days"],
        [(['AAPL', 'ZRAN', 'TSLA', 'WORK'], 100),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], 50),
         (['INTC', 'ZRAN', 'XRDC', 'XOM'], None),
         (['NVDA'], 100),
         (['ZRAN'], 10)],
    )    
    def test_load_daily_data(self, tickers, back_days):
        data_loader = SF1Data(config['sf1_data_path'])
        daily_df = data_loader.load_daily_data(tickers, back_days=back_days)  
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
        
        if back_days is not None:
            for cnt in daily_df.groupby('ticker').size():
                assert cnt == back_days


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

        trans_df = SF1Data.translate_currency(df, ['debt', 'ebit'])
        for col in ['debt', 'ebit']:
            diff = trans_df['{}usd'.format(col)] - trans_df[col]
            diff = np.abs(diff.values / trans_df['{}usd'.format(col)].values)
            diff = diff[~np.isnan(diff)]
            assert diff.max() < 0.1


    @pytest.mark.parametrize("ticker", ['YNDX', 'NIO'])       
    def test_translate_currency_real(self, ticker):
        columns = ['equity','eps','revenue','netinccmn',
                    'cashneq','debt','ebit','ebitda']
        data_loader = SF1Data(config['sf1_data_path'])
        quarterly_df = data_loader.load_quarterly_data(ticker, 10)
        trans_df = SF1Data.translate_currency(quarterly_df, columns)
        for col in columns:
            diff = trans_df['{}usd'.format(col)] - trans_df[col]
            diff = np.abs(diff.values / trans_df['{}usd'.format(col)].values)
            diff = diff[~np.isnan(diff)]
            assert diff.max() < 0.1



class Cl1:
    def a(self):
        return 1
    def b(self):
        return 2
    def ba(self):
        return 20
    
class Cl2:
    def a(self):
        return 3
    def b(self):
        return 4
    def c(self):
        return 5


class TestComboData:
    def test_execute(self):
        cl1 = Cl1()
        cl2 = Cl2()
        combo = ComboData([cl1, cl2])
        assert combo.a() == 1
        assert combo.b() == 2 
        assert combo.c() == 5






















































