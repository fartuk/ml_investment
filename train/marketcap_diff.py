import argparse
import lightgbm as lgbm
from utils import load_json
from data import SF1Data
from features import QuarterlyFeatures, BaseCompanyFeatures, FeatureMerger
from targets import QuarterlyDiffTarget
from models import GroupedOOFModel, AnsambleModel
from marketcap import MarketcapPipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
    pipeline_config = config['pipelines']['marketcap']

    data_loader = SF1Data(config['sf1_data_path'])
    tickers_df = data_loader.load_tickers(
        currency=pipeline_config['currency'],
        scalemarketcap=pipeline_config['scalemarketcap'])
    ticker_list = tickers_df['ticker'].unique().tolist()

    fc1 = QuarterlyDiffFeatures()

    fc2 = BaseCompanyFeatures(
        cat_columns=pipeline_config['cat_columns'])

    feature = FeatureMerger(fc1, fc2, on='ticker')
    target = QuarterlyDiffTarget(col='marketcap', quarter_shift=0)
    model = GroupedOOFModel(lgbm.sklearn.LGBMRegressor(), fold_cnt=5)
                    
    mc_pipeline = MarketcapPipeline(feature, target, model)
    mc_pipeline.fit(config, ticker_list)
    mc_pipeline.export_core()

