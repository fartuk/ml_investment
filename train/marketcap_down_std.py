import argparse
import lightgbm as lgbm
import catboost as ctb
from utils import load_json
from data import SF1Data
from features import QuarterlyFeatures, BaseCompanyFeatures, FeatureMerger, \
                     QuarterlyDiffFeatures, DailyAggQuarterFeatures
from targets import DailyAggTarget
from models import TimeSeriesOOFModel, AnsambleModel
from metrics import median_absolute_relative_error, down_std_norm
from pipelines import BasePipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
    pipeline_config = config['pipelines']['marketcap_down_std']

    data_loader = SF1Data(config['sf1_data_path'])
    tickers_df = data_loader.load_base_data(
        currency=pipeline_config['currency'],
        scalemarketcap=pipeline_config['scalemarketcap'])
    ticker_list = tickers_df['ticker'].unique().tolist()

    fc1 = QuarterlyFeatures(
        columns=pipeline_config['quarter_columns'],
        quarter_counts=pipeline_config['quarter_counts'],
        max_back_quarter=pipeline_config['max_back_quarter'])

    fc2 = BaseCompanyFeatures(
        cat_columns=pipeline_config['cat_columns'])
        
    fc3 = QuarterlyDiffFeatures(
        columns=pipeline_config['quarter_columns'],
        compare_quarter_idxs=pipeline_config['compare_quarter_idxs'],
        max_back_quarter=pipeline_config['max_back_quarter'])
    
    fc4 = DailyAggQuarterFeatures(
        columns=pipeline_config['daily_agg_columns'],
        agg_day_counts=pipeline_config['agg_day_counts'],
        max_back_quarter=pipeline_config['max_back_quarter'])


    feature = FeatureMerger(fc1, fc2, on='ticker')
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])
    feature = FeatureMerger(feature, fc4, on=['ticker', 'date'])

    target = DailyAggTarget(
        col='marketcap',
        horizon=pipeline_config['target_horizon'],
        foo=down_std_norm)

    base_models = [lgbm.sklearn.LGBMRegressor(),
                   ctb.CatBoostRegressor(verbose=False)]
                   
    ansamble = AnsambleModel(base_models=base_models, 
                             bagging_fraction=0.7, model_cnt=20)

    model = TimeSeriesOOFModel(ansamble,
                               time_column='date', fold_cnt=20)

    pipeline = BasePipeline(feature=feature, 
                            target=target, 
                            model=model, 
                            metric=median_absolute_relative_error)
                            
    result = pipeline.fit(data_loader, ticker_list)
    print(result)
    pipeline.export_core('models_data/marketcap_down_std') 
