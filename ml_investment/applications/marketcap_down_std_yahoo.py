import argparse
import os
import lightgbm as lgbm
import catboost as ctb
from urllib.request import urlretrieve
from ml_investment.utils import load_config, load_tickers
from ml_investment.data_loaders.yahoo import YahooBaseData, YahooQuarterlyData
from ml_investment.data_loaders.daily_bars import DailyBarsData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, DailyAggQuarterFeatures, \
                                   QuarterlyDiffFeatures
from ml_investment.targets import DailyAggTarget
from ml_investment.models import TimeSeriesOOFModel, EnsembleModel, LogExpModel
from ml_investment.metrics import median_absolute_relative_error, down_std_norm
from ml_investment.pipelines import Pipeline
from ml_investment.download_scripts import download_yahoo, download_daily_bars

config = load_config()


URL = 'https://github.com/fartuk/ml_investment/releases/download/weights/marketcap_down_std_yahoo.pickle'
OUT_NAME = 'marketcap_down_std_yahoo'
TARGET_HORIZON = 90
MAX_BACK_QUARTER = 2
FOLD_CNT = 5
QUARTER_COUNTS = [1, 2, 4]
COMPARE_QUARTER_IDXS = [1, 4]
CAT_COLUMNS = ["sector"]
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
                            max_back_quarter=MAX_BACK_QUARTER)

    fc2 = BaseCompanyFeatures(data_key='base', cat_columns=CAT_COLUMNS)
        
    fc3 = QuarterlyDiffFeatures(data_key='quarterly',
                                columns=QUARTER_COLUMNS,
                                compare_quarter_idxs=COMPARE_QUARTER_IDXS,
                                max_back_quarter=MAX_BACK_QUARTER)
    
    feature = FeatureMerger(fc1, fc2, on='ticker')
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])

    return feature 


def _create_target():
    target = DailyAggTarget(data_key='daily',
                            col='Close',
                            horizon=TARGET_HORIZON,
                            foo=down_std_norm)
    return target


def _create_model():
    model = TimeSeriesOOFModel(
                base_model=LogExpModel(ctb.CatBoostRegressor(verbose=False)),
                time_column='date',
                fold_cnt=FOLD_CNT)

    return model



def MarketcapDownStdYahoo(pretrained=True) -> Pipeline:
    '''
    Model is used to predict future down-std value.
    Pipeline consist of time-series model training( 
    :class:`~ml_investment.models.TimeSeriesOOFModel` )
    and validation on real marketcap down-std values(
    :class:`~ml_investment.targets.DailyAggTarget` ).
    Model prediction may be interpreted as "risk" for the next quarter.
    :mod:`~ml_investment.data_loaders.yahoo`
    is used for loading data.

    Parameters
    ----------
    pretrained:
        use pretreined weights or not. If so,
        `marketcap_down_std_yahoo.pickle` will be downloaded. 
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
    pipeline = MarketcapDownStdYahoo(pretrained=False)
    tickers = load_tickers()['base_us_stocks']
    result = pipeline.fit(tickers, median_absolute_relative_error)
    print(result)
    path = '{}/{}'.format(config['models_path'], OUT_NAME)
    pipeline.export_core(path)    


if __name__ == '__main__':
    main() 
    
