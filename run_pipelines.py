import argparse
import os
import numpy as np
import pandas as pd
from ml_investment.utils import load_config, load_tickers, check_create_folder
from ml_investment.data_loaders.sf1 import SF1QuarterlyData, SF1DailyData
from ml_investment.pipelines import MergePipeline, LoadingPipeline
from ml_investment.applications.fair_marketcap_sf1 import FairMarketcapSF1
from ml_investment.applications.fair_marketcap_diff_sf1 import FairMarketcapDiffSF1
from ml_investment.applications.marketcap_down_std_sf1 import MarketcapDownStdSF1

config = load_config()


def pipeline_postprocessing(quarterly_df):    
    quarterly_df['fair_marketcap_via_diff_sf1'] = quarterly_df.groupby('ticker')\
        ['marketcap'].shift(-1) * (1 + quarterly_df['fair_marketcap_diff_sf1'])

    quarterly_df['down70'] = quarterly_df['marketcap'] * \
                            (1 - quarterly_df['marketcap_down_std_sf1'] * 1.04)
    quarterly_df['down90'] = quarterly_df['marketcap'] * \
                            (1 - quarterly_df['marketcap_down_std_sf1'] * 1.64)

    quarterly_df = quarterly_df.infer_objects()
    
    return quarterly_df


def calc_metrics(quarterly_df):
    q_df = quarterly_df.copy()
    q_df['fm_m_ratio'] = q_df['fair_marketcap_sf1'] / q_df['marketcap']
    q_df['fmd_m_ratio'] = q_df['fair_marketcap_via_diff_sf1'] / q_df['marketcap']
    
    metrics_df = q_df.drop_duplicates('ticker', keep='first')
    tmp = q_df.groupby('ticker')['fm_m_ratio'].mean().reset_index()
    tmp = tmp.rename({'fm_m_ratio':'mean_fm_m_ratio'}, axis=1)
    metrics_df = pd.merge(metrics_df, tmp, on='ticker', how='left')

    tickers = q_df['ticker'].unique()
    tmp = SF1DailyData(data_path=config['sf1_data_path'],
                       days_count=1).load(tickers)
    tmp = tmp[['ticker', 'marketcap', 'pe']]
    tmp['last_marketcap'] = tmp['marketcap'].astype(float)
    del tmp['marketcap']
    metrics_df = pd.merge(metrics_df, tmp, on='ticker', how='left')

    metrics_df['fm_lm_ratio'] = metrics_df['fair_marketcap_sf1'] / \
                                metrics_df['last_marketcap']
    metrics_df['fm_lm_ratio_rel'] = metrics_df['fm_lm_ratio'] / \
                                    metrics_df['mean_fm_m_ratio']
    metrics_df['date'] = metrics_df['date'].apply(lambda x: np.datetime64(x))
    metrics_df = metrics_df[metrics_df['date'] > np.datetime64('2020-09-01')]
    metrics_df = metrics_df[['ticker', 'fm_m_ratio', 'fmd_m_ratio', 
                             'fm_lm_ratio', 'fm_lm_ratio_rel']]
    
    return metrics_df


def main():
    tickers = load_tickers()['base_us_stocks'] + ['YNDX'] 

    p1 = FairMarketcapSF1()
    p2 = FairMarketcapDiffSF1()
    p3 = MarketcapDownStdSF1()
    p4 = LoadingPipeline(SF1QuarterlyData(config['sf1_data_path']),
                         ['ticker', 'date', 'marketcap'])

    pipeline = MergePipeline(
        pipeline_list=[p1, p2, p3, p4],
        execute_merge_on=['ticker', 'date'])

    quarterly_df = pipeline.execute(tickers)
    quarterly_df = pipeline_postprocessing(quarterly_df)

    metrics_df = calc_metrics(quarterly_df)

    quarterly_path = '{}/quarterly_df.csv'.format(config['out_path'])
    metrics_path = '{}/metrics_df.csv'.format(config['out_path'])
    os.makedirs(config['out_path'], exist_ok=True)

    quarterly_df.to_csv(quarterly_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)




if __name__ == '__main__':
    main()











