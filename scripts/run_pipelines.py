import argparse
import os
import numpy as np
import pandas as pd
from ml_investment.utils import load_json, check_create_folder
from ml_investment.data import SF1Data, QuandlCommoditiesData, ComboData
from ml_investment.pipelines import BasePipeline, ExecuteMergePipeline, \
                                    QuarterlyLoadPipeline



def pipeline_postprocessing(quarterly_df):    
    quarterly_df['fair_marketcap_via_diff'] = quarterly_df.groupby('ticker')\
        ['marketcap'].shift(-1) * (1 + quarterly_df['fair_marketcap_diff'])

    quarterly_df['down70'] = quarterly_df['marketcap'] * \
                            (1 - quarterly_df['marketcap_down_std'] * 1.04)
    quarterly_df['down90'] = quarterly_df['marketcap'] * \
                            (1 - quarterly_df['marketcap_down_std'] * 1.64)

    quarterly_df = quarterly_df.infer_objects()
    
    return quarterly_df


def calc_metrics(quarterly_df):
    q_df = quarterly_df.copy()
    q_df['fm_m_ratio'] = q_df['fair_marketcap'] / q_df['marketcap']
    q_df['fmd_m_ratio'] = q_df['fair_marketcap_via_diff'] / q_df['marketcap']
    
    metrics_df = q_df.drop_duplicates('ticker', keep='first')
    tmp = q_df.groupby('ticker')['fm_m_ratio'].mean().reset_index()
    tmp = tmp.rename({'fm_m_ratio':'mean_fm_m_ratio'}, axis=1)
    metrics_df = pd.merge(metrics_df, tmp, on='ticker', how='left')

    tmp = data_loader.load_daily_data(ticker_list, back_days=1)
    tmp = tmp[['ticker', 'marketcap', 'pe']]
    tmp['last_marketcap'] = tmp['marketcap'].astype(float)
    del tmp['marketcap']
    metrics_df = pd.merge(metrics_df, tmp, on='ticker', how='left')

    metrics_df['fm_lm_ratio'] = metrics_df['fair_marketcap'] / \
                                metrics_df['last_marketcap']
    metrics_df['fm_lm_ratio_rel'] = metrics_df['fm_lm_ratio'] / \
                                    metrics_df['mean_fm_m_ratio']
    metrics_df['date'] = metrics_df['date'].apply(lambda x: np.datetime64(x))
    metrics_df = metrics_df[metrics_df['date'] > np.datetime64('2020-09-01')]
    metrics_df = metrics_df[['ticker', 'fm_m_ratio', 'fmd_m_ratio', 
                             'fm_lm_ratio', 'fm_lm_ratio_rel']]
    
    return metrics_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
 
    dl1 = SF1Data(config['sf1_data_path'])
    dl2 = QuandlCommoditiesData(config['commodities_data_path'])
    data_loader = ComboData([dl1, dl2])

    tickers_df = data_loader.load_base_data()
    ticker_list = tickers_df['ticker'].unique().tolist()
    
    pipeline = ExecuteMergePipeline(
        pipeline_list=[
            BasePipeline.load('{}/fair_marketcap.pickle'.format(
                                                    config['models_path'])),
            BasePipeline.load('{}/fair_marketcap_diff.pickle'.format(
                                                    config['models_path'])),
            BasePipeline.load('{}/marketcap_down_std.pickle'.format(
                                                    config['models_path'])),
            QuarterlyLoadPipeline(['ticker', 'date', 'marketcap'])],
        on=['ticker', 'date'])

    quarterly_df = pipeline.execute(data_loader, ticker_list[:20])
    quarterly_df = pipeline_postprocessing(quarterly_df)

    metrics_df = calc_metrics(quarterly_df)

    quarterly_path = '{}/quarterly_df.csv'.format(config['out_path'])
    metrics_path = '{}/metrics_df.csv'.format(config['out_path'])
    os.makedirs(config['out_path'], exist_ok=True)

    quarterly_df.to_csv(quarterly_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)













