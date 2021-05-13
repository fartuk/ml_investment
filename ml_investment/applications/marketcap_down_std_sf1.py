import argparse
import os
import lightgbm as lgbm
import catboost as ctb
from urllib.request import urlretrieve
from ml_investment.utils import load_config
from ml_investment.data_loaders.sf1 import SF1BaseData, SF1DailyData, \
                                           SF1QuarterlyData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, DailyAggQuarterFeatures, \
                                   QuarterlyDiffFeatures
from ml_investment.targets import DailyAggTarget
from ml_investment.models import TimeSeriesOOFModel, EnsembleModel, LogExpModel
from ml_investment.metrics import median_absolute_relative_error, down_std_norm
from ml_investment.pipelines import Pipeline
from ml_investment.download_scripts import download_sf1, download_commodities

config = load_config()


URL = 'https://github.com/fartuk/ml_investment/releases/download/weights/marketcap_down_std_sf1.pickle'
OUT_NAME = 'marketcap_down_std_sf1'
CURRENCY = 'USD'
TARGET_HORIZON = 90
MAX_BACK_QUARTER = 20
BAGGING_FRACTION = 0.7
MODEL_CNT = 20
FOLD_CNT = 20
QUARTER_COUNTS = [2, 4, 10]
COMPARE_QUARTER_IDXS = [1, 4]
AGG_DAY_COUNTS = [100, 200, 400, 800]
SCALE_MARKETCAP = ["4 - Mid", "5 - Large", "6 - Mega"]
DAILY_AGG_COLUMNS = ["marketcap", "pe"]
CAT_COLUMNS = ["sector", "sicindustry"]
QUARTER_COLUMNS = [
            "revenue",
            "netinc",
            "ncf",
            "assets",
            "ebitda",
            "debt",
            "fcf",
            "gp",
            "workingcapital",
            "cashneq",
            "rnd",
            "sgna",
            "ncfx",
            "divyield",
            "currentratio",
            "netinccmn"
         ]



def _check_download_data():
    if not os.path.exists(config['sf1_data_path']):
        print('Downloading sf1 data')
        download_sf1.main()


def _create_data():
    data = {}
    data['quarterly'] = SF1QuarterlyData(config['sf1_data_path'])
    data['base'] = SF1BaseData(config['sf1_data_path'])
    data['daily'] = SF1DailyData(config['sf1_data_path'])
    
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
    
    fc4 = DailyAggQuarterFeatures(daily_data_key='daily',
                                  quarterly_data_key='quarterly',
                                  columns=DAILY_AGG_COLUMNS,
                                  agg_day_counts=AGG_DAY_COUNTS,
                                  max_back_quarter=MAX_BACK_QUARTER)

    feature = FeatureMerger(fc1, fc2, on='ticker')
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])
    feature = FeatureMerger(feature, fc4, on=['ticker', 'date'])

    return feature


def _create_target():
    target = DailyAggTarget(data_key='daily',
                            col='marketcap',
                            horizon=TARGET_HORIZON,
                            foo=down_std_norm)
    return target


def _create_model():
    base_models = [LogExpModel(lgbm.sklearn.LGBMRegressor()),
                   LogExpModel(ctb.CatBoostRegressor(verbose=False))]
                   
    ensemble = EnsembleModel(base_models=base_models, 
                             bagging_fraction=BAGGING_FRACTION,
                             model_cnt=MODEL_CNT)

    model = TimeSeriesOOFModel(base_model=ensemble,
                               time_column='date',
                               fold_cnt=FOLD_CNT)

    return model
 


def MarketcapDownStdSF1(pretrained=True) -> Pipeline:
    '''
    Model is used to predict future down-std value.
    Pipeline consist of time-series model training( 
    :class:`~ml_investment.models.TimeSeriesOOFModel` )
    and validation on real marketcap down-std values(
    :class:`~ml_investment.targets.DailyAggTarget` ).
    Model prediction may be interpreted as "risk" for the next quarter.
    :mod:`~ml_investment.data_loaders.sf1`
    is used for loading data.

    Note:
        SF1 dataset is paid, so for using this model you need to subscribe 
        and paste quandl token to `~/.ml_investment/secrets.json`
        ``quandl_api_key``

    Parameters
    ----------
    pretrained:
        use pretreined weights or not. If so,
        `marketcap_down_std_sf1.pickle` will be downloaded. 
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
            urlretrieve(URL, core_path)       
        pipeline.load_core(core_path)

    return pipeline


 


def main():
    '''
    Default model training. Resulted model weights directory path 
    can be changed in `~/.ml_investment/config.json` ``models_path``
    '''
    pipeline = MarketcapDownStdSF1(pretrained=False)
    base_df = SF1BaseData(config['sf1_data_path']).load()
    tickers = base_df[(base_df['currency'] == CURRENCY) &\
                      (base_df['scalemarketcap'].apply(lambda x: x in SCALE_MARKETCAP))
                     ]['ticker'].values
    result = pipeline.fit(tickers[:200], median_absolute_relative_error)
    print(result)
    path = '{}/{}'.format(config['models_path'], OUT_NAME)
    pipeline.export_core(path)    


if __name__ == '__main__':
   main() 
    
