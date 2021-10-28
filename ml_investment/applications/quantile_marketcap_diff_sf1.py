import argparse
import sys
import os

sys.path.append(os.getcwd())
from ml_investment.data_loaders.quandl_commodities import QuandlCommoditiesData
from ml_investment.data_loaders.sf1 import SF1BaseData, SF1DailyData, \
    SF1QuarterlyData
from ml_investment.utils import load_config, bound_filter_foo_gen
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
    FeatureMerger, DailyAggQuarterFeatures, \
    QuarterlyDiffFeatures, RelativeGroupFeatures
from ml_investment.targets import QuarterlyDiffTarget, DailyDiffTarget
from ml_investment.models import GroupedOOFModel, EnsembleModel, LogExpModel, QuantileCatboostModel, \
    QuantileLightgbmModel
from ml_investment.metrics import median_absolute_relative_error, median_abs_diff
from ml_investment.pipelines import Pipeline
from ml_investment.download_scripts import download_sf1, download_commodities

config = load_config()

QUANTILES = [0.01, 0.1, 0.5, 0.9, 0.99]
OUT_NAME = 'quantile_marketcap_diff_sf1'
CURRENCY = 'USD'
MAX_BACK_QUARTER = 20
MIN_BACK_QUARTER = 0
MAX_TARGET_BOUND = 1.5
MIN_TARGET_BOUND = -0.9
BAGGING_FRACTION = 0.7
MODEL_CNT = 20
FOLD_CNT = 5
QUARTER_COUNTS = [2, 4, 10]
COMPARE_QUARTER_IDXS = [1, 4]
AGG_DAY_COUNTS = [100, 200, 400, 800]
SCALE_MARKETCAP = ["4 - Mid", "5 - Large", "6 - Mega"]
SCALE_REVENUE = ["1 - Nano", "2 - Micro", "3 - Small", "4 - Mid", "5 - Large", "6 - Mega"]
SCALE_MARKETCAP = SCALE_REVENUE = ["6 - Mega"]  # test
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
DEV_COLUMNS = ['rnd_invcap', 'capex_invcap', 'ebit_invcap', 'ev_ebitda', \
               'ev_ebit', 'debt_equity', 'grossmargin_ebitdamargin', 'debt_ebit']
COMMODITIES_CODES = [
    'LBMA/GOLD',
    'JOHNMATT/PALL',
]


def _check_download_data():
    if not os.path.exists(config['sf1_data_path']):
        print('Downloading sf1 data')
        download_sf1.main()

    if not os.path.exists(config['commodities_data_path']):
        print('Downloading commodities data')
        download_commodities.main()


def _create_data():
    data = {}
    data['quarterly'] = SF1QuarterlyData()
    data['base'] = SF1BaseData()
    data['daily'] = SF1DailyData()
    data['commodities'] = QuandlCommoditiesData()

    return data


def _preprocess(x):
    x['rnd_invcap'] = x['rnd'] / x['invcap']
    x['capex_invcap'] = x['capex'] / x['invcap']
    x['ebit_invcap'] = x['ebit'] / x['invcap']
    x['ev_ebitda'] = x['ev'] / x['ebitda']
    x['ev_ebit'] = x['ev'] / x['ebit']
    x['debt_equity'] = x['debt'] / x['equity']
    x['grossmargin_ebitdamargin'] = x['grossmargin'] / x['ebitdamargin']
    x['debt_ebit'] = x['debt'] / x['ebitda']

    return x


def _create_feature():
    fc1 = QuarterlyFeatures(data_key='quarterly',
                            columns=QUARTER_COLUMNS + DEV_COLUMNS,
                            quarter_counts=QUARTER_COUNTS,
                            max_back_quarter=MAX_BACK_QUARTER,
                            min_back_quarter=MIN_BACK_QUARTER,
                            calc_stats_on_diffs=True,
                            data_preprocessing=_preprocess)

    fc2 = QuarterlyDiffFeatures(data_key='quarterly',
                                columns=QUARTER_COLUMNS + DEV_COLUMNS,
                                compare_quarter_idxs=COMPARE_QUARTER_IDXS,
                                max_back_quarter=MAX_BACK_QUARTER,
                                min_back_quarter=MIN_BACK_QUARTER,
                                data_preprocessing=_preprocess)

    fc3 = DailyAggQuarterFeatures(daily_data_key='commodities',
                                  quarterly_data_key='quarterly',
                                  columns=['price'],
                                  agg_day_counts=AGG_DAY_COUNTS,
                                  max_back_quarter=MAX_BACK_QUARTER,
                                  min_back_quarter=MIN_BACK_QUARTER,
                                  daily_index=COMMODITIES_CODES)

    fc4 = RelativeGroupFeatures(feature_calculator=fc3,
                                group_data_key='base',
                                group_col='industry',
                                relation_foo=lambda x, y: x - y,
                                keep_group_feats=True)

    fc5 = RelativeGroupFeatures(feature_calculator=fc1,
                                group_data_key='base',
                                group_col='industry',
                                relation_foo=lambda x, y: x - y,
                                keep_group_feats=True)

    feature = FeatureMerger(fc1, fc2, on=['ticker', 'date'])
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])
    feature = FeatureMerger(feature, fc4, on=['ticker', 'date'])
    feature = FeatureMerger(feature, fc5, on=['ticker', 'date'])

    return feature


def _create_target():
    target = QuarterlyDiffTarget(data_key='quarterly', col='marketcap')
    #target = DailyDiffTarget(data_key='daily', col='marketcap')
    return target


def _create_model(verbose_level: int = 0):
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    lightgbm_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_data_in_leaf': 0,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': verbose_level,
    }
    # https://catboost.ai/en/docs/concepts/python-reference_catboostregressor
    catboost_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': None,
        'min_data_in_leaf': None,
        'num_leaves': None,
        'random_state': 42,
        'verbose': verbose_level,
    }

    model = QuantileCatboostModel(catboost_params, quantiles=QUANTILES)

    # base_models = [
    #     QuantileLightgbmModel(lightgbm_params, quantiles=QUANTILES),
    #     QuantileCatboostModel(catboost_params, quantiles=QUANTILES),
    # ]
    # ensemble = EnsembleModel(base_models=base_models,
    #                          bagging_fraction=BAGGING_FRACTION,
    #                          model_cnt=MODEL_CNT)
    # model = GroupedOOFModel(ensemble,
    #                         group_column='ticker',
    #                         fold_cnt=FOLD_CNT)

    return model


def QuantileMarketcapDiffSF1(max_back_quarter: int = None,
                           min_back_quarter: int = None,
                           pretrained: bool = True) -> Pipeline:
    '''
    Model is used to evaluate quarter-to-quarter(q2q) company
    fundamental progress. Model uses
    :class:`~ml_investment.features.QuarterlyDiffFeatures`
    (q2q results progress, e.g. 30% revenue increase,
    decrease in debt by 15% etc),
    :class:`~ml_investment.features.QuarterlyFeatures`
    and trying to predict real q2q marketcap difference(
    :class:`~ml_investment.targets.QuarterlyDiffTarget` ).
    So model prediction may be interpreted as "fair" marketcap
    change according this q2q fundamental change.
    :mod:`~ml_investment.data_loaders.sf1`
    is used for loading data.

    Note:
        SF1 dataset is paid, so for using this model you need to subscribe
        and paste quandl token to `~/.ml_investment/secrets.json`
        ``quandl_api_key``

    Parameters
    ----------
    max_back_quarter:
        max quarter number which will be used in model
    min_back_quarter:
        min quarter number which will be used in model
    data_source:
        which data use for model. One of ['sf1', 'mongo'].
        If 'mongo', than data will be loaded from db,
        credentials specified at `~/.ml_investment/config.json`.
        If 'sf1' - from folder specified at ``sf1_data_path``
        in `~/.ml_investment/secrets.json`.
    pretrained:
        use pretreined weights or not.
        Downloading directory path can be changed in
        `~/.ml_investment/config.json` ``models_path``
    '''

    if max_back_quarter is not None:
        global MAX_BACK_QUARTER
        MAX_BACK_QUARTER = max_back_quarter

    if min_back_quarter is not None:
        global MIN_BACK_QUARTER
        MIN_BACK_QUARTER = min_back_quarter

    _check_download_data()

    data = _create_data()
    feature = _create_feature()
    target = _create_target()
    model = _create_model()

    pipeline = Pipeline(feature=feature,
                        target=target,
                        model=model,
                        data=data,
                        out_name=OUT_NAME,
                        quantiles=QUANTILES)

    if pretrained:
        core_path = '{}/{}.pickle'.format(config['models_path'], OUT_NAME)
        pipeline.load_core(core_path)

    return pipeline


def main():
    '''
    Default model training. Resulted model weights directory path
    can be changed in `~/.ml_investment/config.json` ``models_path``
    '''

    pipeline = QuantileMarketcapDiffSF1(pretrained=False)
    base_df = pipeline.data['base'].load()
    tickers = base_df[
        (base_df['currency'] == CURRENCY) &
        (base_df['scalemarketcap'].apply(lambda x: x in SCALE_MARKETCAP)) &
        (base_df['scalerevenue'].apply(lambda x: x in SCALE_REVENUE))
    ]['ticker'].values

    filter_foo = bound_filter_foo_gen(min_bound=MIN_TARGET_BOUND,
                                      max_bound=MAX_TARGET_BOUND)

    result = pipeline.fit(tickers,
                          metric=median_abs_diff,
                          target_filter_foo=filter_foo)
    print(result)
    path = '{}/{}'.format(config['models_path'], OUT_NAME)
    pipeline.export_core(path)


def execute_example():
    global OUT_NAME
    OUT_NAME = 'quantile_marketcap_diff_sf1_'  # path to trained model .pickle

    pipeline = QuantileMarketcapDiffSF1()

    tickers = ['AAPL', 'TSLA']
    result = pipeline.execute(tickers)
    print(result)


if __name__ == '__main__':
    main()
    # execute_example()
