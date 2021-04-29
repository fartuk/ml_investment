import argparse
import os
import lightgbm as lgbm
import catboost as ctb
from urllib.request import urlretrieve
from ml_investment.utils import load_config, load_tickers
from ml_investment.data import YahooData, DailyBarsData, ComboData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, DailyAggQuarterFeatures, \
                                   QuarterlyDiffFeatures
from ml_investment.targets import DailyAggTarget
from ml_investment.models import TimeSeriesOOFModel, EnsembleModel, LogExpModel
from ml_investment.metrics import median_absolute_relative_error, down_std_norm
from ml_investment.pipelines import BasePipeline
from ml_investment.download_scripts import download_yahoo, download_daily_bars


URL = 'https://github.com/fartuk/ml_investment/releases\
      /download/weights/marketcap_down_std_yahoo.pickle'
OUT_NAME = 'marketcap_down_std_yahoo'
TARGET_HORIZON = 90
MAX_BACK_QUARTER = 2
BAGGING_FRACTION = 0.7
MODEL_CNT = 20
FOLD_CNT = 5
QUARTER_COUNTS = [1, 2, 4]
COMPARE_QUARTER_IDXS = [1, 4]
CAT_COLUMNS = ["sector"]
QUARTER_COLUMNS = [
    'quarterlyTotalRevenue',
    'quarterlyNetIncome',
    'quarterlyFreeCashFlow',
    'quarterlyTotalAssets',
    'quarterlyNetDebt',
    'quarterlyGrossProfit',
    'quarterlyWorkingCapital',
    'quarterlyCashAndCashEquivalents',
    'quarterlyResearchAndDevelopment',
    'quarterlyCashDividendsPaid',
]


class MarketcapDownStdYahoo:
    def __init__(self, pretrained=True):
        self.config = load_config()

        self._check_download_data()
        self.data_loader = self._create_loader()
        self.pipeline = self._create_pipeline()   
        
        core_path = '{}/{}.pickle'.format(self.config['models_path'],
                                          OUT_NAME)

        if pretrained:
            if not os.path.exists(core_path):
                urlretrieve(URL, core_path)       
            self.pipeline.load_core(core_path)


    def _check_download_data(self):
        if not os.path.exists(self.config['yahoo_data_path']):
            print('Downloading Yahoo data')
            download_yahoo.main()

        if not os.path.exists(self.config['daily_bars_data_path']):
            print('Downloading daily bars data')
            download_daily_bars.main()
 

    def _create_loader(self):
        dl1 = YahooData(self.config['yahoo_data_path'])
        dl2 = DailyBarsData(self.config['daily_bars_data_path'])
        data_loader = ComboData([dl1, dl2])
        return data_loader 


    def _create_pipeline(self):
        fc1 = QuarterlyFeatures(columns=QUARTER_COLUMNS,
                                quarter_counts=QUARTER_COUNTS,
                                max_back_quarter=MAX_BACK_QUARTER)

        fc2 = BaseCompanyFeatures(cat_columns=CAT_COLUMNS)
            
        fc3 = QuarterlyDiffFeatures(columns=QUARTER_COLUMNS,
                                    compare_quarter_idxs=COMPARE_QUARTER_IDXS,
                                    max_back_quarter=MAX_BACK_QUARTER)
        
        feature = FeatureMerger(fc1, fc2, on='ticker')
        feature = FeatureMerger(feature, fc3, on=['ticker', 'date'])

        target = DailyAggTarget(col='Close',
                                horizon=TARGET_HORIZON,
                                foo=down_std_norm)

        base_models = [LogExpModel(lgbm.sklearn.LGBMRegressor()),
                       LogExpModel(ctb.CatBoostRegressor(verbose=False))]
                       
        ensemble = EnsembleModel(base_models=base_models, 
                                 bagging_fraction=BAGGING_FRACTION,
                                 model_cnt=MODEL_CNT)

        model = TimeSeriesOOFModel(base_model=ensemble,
                                   time_column='date',
                                   fold_cnt=FOLD_CNT)

        pipeline = BasePipeline(feature=feature, 
                                target=target, 
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=OUT_NAME)

        return pipeline


    def fit(self):
        ticker_list = load_tickers()['base_us_stocks']
        result = self.pipeline.fit(self.data_loader, ticker_list)
        print(result)


    def predict(self, tickers):
        return self.pipeline.execute(self.data_loader, tickers)





def main():
    model = MarketcapDownStdYahoo(pretrained=False)
    model.fit()
    path = '{}/{}'.format(model.config['models_path'], OUT_NAME)
    model.pipeline.export_core(path)    


if __name__ == '__main__':
   main() 
    
