import argparse
import lightgbm as lgbm
import catboost as ctb
from ml_investment.utils import load_json
from ml_investment.data import SF1Data, QuandlCommoditiesData, ComboData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, DailyAggQuarterFeatures, \
                                   CommoditiesAggQuarterFeatures
from ml_investment.targets import QuarterlyTarget
from ml_investment.models import GroupedOOFModel, EnsembleModel, LogExpModel
from ml_investment.metrics import median_absolute_relative_error
from ml_investment.pipelines import BasePipeline


OUT_NAME = 'fair_marketcap'
CURRENCY = 'USD'
MAX_BACK_QUARTER = 10
BAGGING_FRACTION = 0.7
MODEL_CNT = 20
FOLD_CNT = 5
QUARTER_COUNTS = [2, 4, 10]
AGG_DAY_COUNTS = [100, 200, 400, 800]
COMMODITIES_AGG_DAY_LIMITS = [100, 200, 400, 800]
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
COMMODITIES_CODES = [
            'LBMA/GOLD',
            'LBMA/SILVER',
            'JOHNMATT/PALL',
            'ODA/PBARL_USD',
            'TFGRAIN/CORN', 
            'ODA/PRICENPQ_USD',  
            'CHRIS/CME_DA1',
            'ODA/PBEEF_USD',
            'ODA/PPOULT_USD', 
            'ODA/PPORK_USD',  
            'ODA/PWOOLC_USD',
            'CHRIS/CME_CL1',
            'ODA/POILBRE_USD',
            'CHRIS/CME_NG1', 
            'ODA/PCOFFOTM_USD',
            'ODA/PCOCO_USD',
            'ODA/PORANG_USD',
            'ODA/PBANSOP_USD',
            'ODA/POLVOIL_USD',
            'ODA/PLOGSK_USD',
            'ODA/PCOTTIND_USD'
                           ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
        
    dl1 = SF1Data(config['sf1_data_path'])
    dl2 = QuandlCommoditiesData(config['commodities_data_path'])
    data_loader = ComboData([dl1, dl2])
    
    tickers_df = data_loader.load_base_data(
        currency=CURRENCY,
        scalemarketcap=SCALE_MARKETCAP)
    ticker_list = tickers_df['ticker'].unique().tolist()

    fc1 = QuarterlyFeatures(
        columns=QUARTER_COLUMNS,
        quarter_counts=QUARTER_COUNTS,
        max_back_quarter=MAX_BACK_QUARTER)

    fc2 = BaseCompanyFeatures(cat_columns=CAT_COLUMNS)

    # Daily agss on marketcap and pe is possible here because it 
    # normalized and there are no leakage.
    fc3 = DailyAggQuarterFeatures(
        columns=DAILY_AGG_COLUMNS,
        agg_day_counts=AGG_DAY_COUNTS,
        max_back_quarter=MAX_BACK_QUARTER)
    
    fc4 = CommoditiesAggQuarterFeatures(
        commodities=COMMODITIES_CODES, 
        agg_day_limits=COMMODITIES_AGG_DAY_LIMITS, 
        max_back_quarter=MAX_BACK_QUARTER)
    
    feature = FeatureMerger(fc1, fc2, on='ticker')
    feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])
    feature = FeatureMerger(feature, fc4, on=['ticker', 'date'])

    target = QuarterlyTarget(col='marketcap', quarter_shift=0)

    base_models = [LogExpModel(lgbm.sklearn.LGBMRegressor()),
                   LogExpModel(ctb.CatBoostRegressor(verbose=False))]
                   
    ensemble = EnsembleModel(
        base_models=base_models, 
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
    
    
    
    
