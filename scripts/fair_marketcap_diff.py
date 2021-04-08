import argparse
import lightgbm as lgbm
import catboost as ctb
from ml_investment.utils import load_json
from ml_investment.data import SF1Data
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, QuarterlyDiffFeatures
from ml_investment.targets import QuarterlyDiffTarget
from ml_investment.models import GroupedOOFModel, EnsembleModel
from ml_investment.metrics import median_absolute_relative_error
from ml_investment.pipelines import BasePipeline


OUT_NAME = 'fair_marketcap_diff'
CURRENCY = 'USD'
MAX_BACK_QUARTER = 10
BAGGING_FRACTION = 0.7
MODEL_CNT = 20
FOLD_CNT = 5
QUARTER_COUNTS = [2, 4, 10]
COMPARE_QUARTER_IDXS = [1, 4]
SCALE_MARKETCAP = ["4 - Mid", "5 - Large", "6 - Mega"]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
    
    data_loader = SF1Data(config['sf1_data_path'])
    tickers_df = data_loader.load_base_data(
        currency=CURRENCY,
        scalemarketcap=SCALE_MARKETCAP)
    ticker_list = tickers_df['ticker'].unique().tolist()

    fc1 = QuarterlyFeatures(
        columns=QUARTER_COLUMNS,
        quarter_counts=QUARTER_COUNTS,
        max_back_quarter=MAX_BACK_QUARTER)

    fc2 = BaseCompanyFeatures(cat_columns=CAT_COLUMNS)
        
    fc3 = QuarterlyDiffFeatures(
        columns=QUARTER_COLUMNS,
        compare_quarter_idxs=COMPARE_QUARTER_IDXS,
        max_back_quarter=MAX_BACK_QUARTER)
                            
    feature = FeatureMerger(fc1, fc2, on='ticker')
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])

    target = QuarterlyDiffTarget(col='marketcap')

    base_models = [lgbm.sklearn.LGBMRegressor(),
                   ctb.CatBoostRegressor(verbose=False)]
                   
    ensemble = EnsembleModel(base_models=base_models, 
                             bagging_fraction=BAGGING_FRACTION,
                             model_cnt=MODEL_CNT)

    model = GroupedOOFModel(ensemble,
                            group_column='ticker',
                            fold_cnt=FOLD_CNT)

    pipeline = BasePipeline(feature=feature, 
                            target=target, 
                            model=model, 
                            metric=median_absolute_relative_error,
                            out_name=OUT_NAME)
                            
    result = pipeline.fit(data_loader, ticker_list)
    print(result)
    pipeline.export_core('{}/{}'.format(config['models_data'], OUT_NAME)) 




