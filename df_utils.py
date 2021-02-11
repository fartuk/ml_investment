import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from utils import load_json
from data import load_cf1_df

config = load_json("config.json")


def form_pred_df(X, pred, tickers_df):
    pred_df = pd.DataFrame()
    pred_df['ticker'] = X['ticker']
    pred_df['date'] = X['date'].apply(lambda x: np.datetime64(x))
    pred_df['marketcap'] = X['marketcap']
    pred_df['pred_marketcap'] = pred
    pred_df = pd.merge(pred_df, tickers_df[['ticker', 'sector', 'exchange', 'sicindustry']], on='ticker', how='left')
    pred_df['ratio'] = pred_df['pred_marketcap'] / pred_df['marketcap']
    
    return pred_df


def load_last_marketcap(ticker_list):
    last_marketcap_arr = []
    for ticker in tqdm(ticker_list):
        path = '{}/cf1/daily/{}.json'.format(config['data_path'], ticker)
        if not os.path.exists(path):
            continue
        daily_df = load_cf1_df(path)
        if len(daily_df) == 0:
            continue
        last_marketcap_arr.append({'ticker':ticker, 'last_marketcap':daily_df['marketcap'].values[0]})

    last_marketcap_df = pd.DataFrame(last_marketcap_arr)
    last_marketcap_df['last_marketcap'] = last_marketcap_df['last_marketcap'] * 1e6
    
    return last_marketcap_df


def form_last_quarter_df(pred_df):
    last_df = pred_df.sort_values('date', ascending=False)
    last_df = last_df.drop_duplicates('ticker', keep='first')
    last_df['ratio'] = last_df['pred_marketcap'] / last_df['marketcap']
    last_df = last_df[last_df['date'] > np.datetime64('2020-09-01')]

    last_marketcap_df = load_last_marketcap(pred_df['ticker'].unique())
    last_df = pd.merge(last_df, last_marketcap_df, on='ticker', how='left')
    last_df['last_ratio'] = last_df['pred_marketcap'] / last_df['last_marketcap']

    last_df = last_df.sort_values('last_ratio', ascending=False)

    ticker_mean_ratio = pred_df.groupby('ticker')['ratio'].mean().reset_index().rename({'ratio':'mean_ratio'}, axis=1)
    last_df = pd.merge(last_df, ticker_mean_ratio, on='ticker', how='left')
    last_df['ratio_ratio'] = last_df['last_ratio'] / last_df['mean_ratio']
    last_df = last_df.sort_values('ratio_ratio', ascending=False)
    last_df.index = range(len(last_df))
    
    return last_df


def form_portfolio_cumm_df(portfolio_df):
    arr = []
    for ticker in portfolio_df['ticker']:
        daily_df = load_cf1_df('{}/cf1/daily/{}.json'.format(config['data_path'], ticker))
        arr.append(daily_df)

    index_df = pd.DataFrame()
    index_df['date'] = list(set(np.concatenate([x['date'].values for x in arr], axis=0)))

    for ticker, part in portfolio_df[['ticker', 'part']].values:
        daily_df = load_cf1_df('{}/cf1/daily/{}.json'.format(config['data_path'], ticker))
        daily_df[ticker] = daily_df['marketcap'].values / daily_df['marketcap'].values[-1] * part

        index_df = pd.merge(index_df, daily_df[['date', ticker]], how='left')

    index_df['date'] = index_df['date'].apply(lambda x: np.datetime64(x))
    index_df = index_df.sort_values('date')
    index_df.index = range(len(index_df))

    index_df = index_df.interpolate().fillna(0)
    index_df['val'] = index_df[index_df.columns[1:]].sum(axis=1)
    
    return index_df

