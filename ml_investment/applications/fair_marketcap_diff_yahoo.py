import argparse
import os
import lightgbm as lgbm
import catboost as ctb
from urllib.request import urlretrieve
from ml_investment.utils import load_config, load_tickers
from ml_investment.data_loaders.yahoo import YahooBaseData, YahooQuarterlyData
from ml_investment.data_loaders.daily_bars import DailyBarsData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, QuarterlyDiffFeatures
from ml_investment.targets import DailySmoothedQuarterlyDiffTarget
from ml_investment.models import GroupedOOFModel, EnsembleModel, LogExpModel
from ml_investment.metrics import median_absolute_relative_error
from ml_investment.pipelines import Pipeline
from ml_investment.download_scripts import download_yahoo, download_daily_bars

config = load_config()


URL = 'https://github.com/fartuk/ml_investment/releases/download/weights/fair_marketcap_diff_yahoo.pickle'
OUT_NAME = 'fair_marketcap_diff_yahoo'
FOLD_CNT = 5
QUARTER_COUNTS = [1, 2, 4]
COMPARE_QUARTER_IDXS = [1, 4]
CAT_COLUMNS = ['sector']
QUARTER_COLUMNS = [
    'totalRevenue',
    'netIncome',
    'cash',
    'totalAssets',
    'costOfRevenue',
    'grossProfit',
    'researchDevelopment',
    'totalOperatingExpenses',
    'ebit',
    'totalLiab',
    'discontinuedOperations',
]


def _check_download_data():
    if not os.path.exists(config['yahoo_data_path']):
        print('Downloading Yahoo data')
        download_yahoo.main()

    if not os.path.exists(config['daily_bars_data_path']):
        print('Downloading daily bars data')
        download_daily_bars.main()
     


def _create_data():
    data = {}
    data['quarterly'] = YahooQuarterlyData(config['yahoo_data_path'])
    data['daily'] = DailyBarsData(config['daily_bars_data_path'])
    data['base'] = YahooBaseData(config['yahoo_data_path'])
    
    return data


def _create_feature():
    fc1 = QuarterlyFeatures(data_key='quarterly',
                            columns=QUARTER_COLUMNS,
                            quarter_counts=QUARTER_COUNTS,
                            max_back_quarter=1)
    
    fc2 = BaseCompanyFeatures(data_key='base', cat_columns=CAT_COLUMNS)

    fc3 = QuarterlyDiffFeatures(data_key='quarterly',
                                columns=QUARTER_COLUMNS,
                                compare_quarter_idxs=COMPARE_QUARTER_IDXS,
                                max_back_quarter=1)

    feature = FeatureMerger(fc1, fc2, on='ticker')
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])

    return feature


def _create_target():
    target = DailySmoothedQuarterlyDiffTarget(daily_data_key='daily',
                                              quarterly_data_key='quarterly',
                                              col='Close',
                                              smooth_horizon=10)
    return target


def _create_model():
    model = GroupedOOFModel(
                base_model=LogExpModel(ctb.CatBoostRegressor(verbose=False)),
                group_column='ticker',
                fold_cnt=FOLD_CNT)
    
    return model



def FairMarketcapDiffYahoo(pretrained=True) -> Pipeline:
    '''
    Model is used to evaluate quarter-to-quarter(q2q) company
    fundamental progress. Model uses
    :class:`~ml_investment.features.QuarterlyDiffFeatures`
    (q2q results progress, e.g. 30% revenue increase,
    decrease in debt by 15% etc), 
    :class:`~ml_investment.features.BaseCompanyFeatures`,
    :class:`~ml_investment.features.QuarterlyFeatures`
    and trying to predict smoothed real q2q marketcap difference( 
    :class:`~ml_investment.targets.DailySmoothedQuarterlyDiffTarget` ).
    So model prediction may be interpreted as "fair" marketcap
    change according this q2q fundamental change.
    :mod:`~ml_investment.data_loaders.yahoo` and
    :mod:`~ml_investment.data_loaders.daily_bars`
    are used for loading data.

    Parameters
    ----------
    pretrained:
        use pretreined weights or not. If so,
        `fair_marketcap_diff_yahoo.pickle` will be downloaded. 
        Downloading directory path can be changed in
        `~/.ml_investment/config.json` ``models_path``
    '''
    _check_download_data()
    data = _create_data()
    feature = _create_feature()
    target = _create_target()
    model = _create_model()

    pipeline = Pipeline(feature=feature, 
                        target=target, 
                        model=model,
                        data=data,
                        out_name=OUT_NAME)
            
    core_path = '{}/{}.pickle'.format(config['models_path'], OUT_NAME)

    if pretrained:
        if not os.path.exists(core_path):
            print('Downloading pretrained model')
            urlretrieve(URL, core_path)       
        pipeline.load_core(core_path)

    return pipeline


def main():
    '''
    Default model training. Resulted model weights directory path 
    can be changed in `~/.ml_investment/config.json` ``models_path``
    '''
    pipeline = FairMarketcapDiffYahoo(pretrained=False)
    tickers = load_tickers()['base_us_stocks']
    result = pipeline.fit(tickers, median_absolute_relative_error)
    print(result)
    path = '{}/{}'.format(config['models_path'], OUT_NAME)
    pipeline.export_core(path)    


if __name__ == '__main__':
    main() 
    
