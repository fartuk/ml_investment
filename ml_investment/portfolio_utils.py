import time
import numpy as np
import pandas as pd
from tqdm import tqdm

def balance_portfolio_sec(portfolio_df, prop_col):
    sectors = [
            {'sector': 'Basic Materials', 'sector_part': 0.06931405604649113},
            {'sector': 'Communication Services', 'sector_part': 0.12239164380419218},
            {'sector': 'Consumer Cyclical', 'sector_part': 0.08560895832599877},
            {'sector': 'Consumer Defensive', 'sector_part': 0.13342669998029802},
            {'sector': 'Energy', 'sector_part': 0.055189129343203504},
            {'sector': 'Financial Services', 'sector_part': 0.07863521155719683},
            {'sector': 'Healthcare', 'sector_part': 0.05953360680576723},
            {'sector': 'Industrials', 'sector_part': 0.043649199983053354},
            {'sector': 'Real Estate', 'sector_part': 0.054883199803837886},
            {'sector': 'Technology', 'sector_part': 0.17515194593480288},
            {'sector': 'Utilities', 'sector_part': 0.12221634841515819}
        ]  
    sectors_df = pd.DataFrame(sectors)

    portfolio_df = pd.merge(portfolio_df, sectors_df, on='sector', how='left')

    tmp = portfolio_df.groupby('sector')[prop_col].sum().reset_index().rename({prop_col:'sum'}, axis=1)
    portfolio_df = pd.merge(portfolio_df, tmp, on='sector', how='left')
    portfolio_df['part'] = portfolio_df[prop_col] / portfolio_df['sum'] * portfolio_df['sector_part'] #* 52_000
    
    return portfolio_df['part'].values


def balance_portfolio(portfolio_df, prop_col):
    portfolio_df['part'] = portfolio_df[prop_col] / portfolio_df[prop_col].sum() #* 52_000
    
    return portfolio_df['part'].values

