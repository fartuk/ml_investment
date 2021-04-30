import argparse
import os
import lightgbm as lgbm
import catboost as ctb
from urllib.request import urlretrieve
from ml_investment.utils import load_config, load_tickers
from ml_investment.data import YahooData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures, \
                                   FeatureMerger, DailyAggQuarterFeatures, \
                                   CommoditiesAggQuarterFeatures
from ml_investment.targets import BaseInfoTarget
from ml_investment.models import GroupedOOFModel, EnsembleModel, LogExpModel
from ml_investment.metrics import median_absolute_relative_error
from ml_investment.pipelines import BasePipeline
from ml_investment.download_scripts import download_yahoo


URL = 'https://github.com/fartuk/ml_investment/releases\
      /download/weights/fair_marketcap_yahoo.pickle'
OUT_NAME = 'fair_marketcap_yahoo'
BAGGING_FRACTION = 0.7
MODEL_CNT = 20
FOLD_CNT = 5
QUARTER_COUNTS = [1, 2, 4]
CAT_COLUMNS = ['sector']
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



class FairMarketcapYahoo:
    '''
    Model is used to estimate fair company marketcap for `last` quarter. 
    Pipeline uses features from 
    :class:`~ml_investment.features.BaseCompanyFeatures`,
    :class:`~ml_investment.features.QuarterlyFeatures`
    and trained to predict real market capitalizations
    ( using :class:`~ml_investment.targets.QuarterlyTarget` ). 
    Since some companies are overvalued and some are undervalued, 
    the model makes an average "fair" prediction.
    :class:`~ml_investment.data.YahooData`
    is used for loading data.
    '''
    def __init__(self, pretrained=True):
        '''
        Parameters
        ----------
        pretrained:
            use pretreined weights or not. If so, `fair_marketcap_yahoo.pickle`
            will be downloaded. Downloading directory path can be changed in
            `~/.ml_investment/config.json` ``models_path``
        '''
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
         

    def _create_loader(self):
        data_loader = YahooData(self.config['yahoo_data_path'])
        return data_loader 


    def _create_pipeline(self):
        fc1 = QuarterlyFeatures(columns=QUARTER_COLUMNS,
                                quarter_counts=QUARTER_COUNTS,
                                max_back_quarter=1)
        
        fc2 = BaseCompanyFeatures(cat_columns=CAT_COLUMNS)

        feature = FeatureMerger(fc1, fc2, on='ticker')

        target = BaseInfoTarget(col='enterpriseValue')
        
        base_models = [LogExpModel(lgbm.sklearn.LGBMRegressor()),
                       LogExpModel(ctb.CatBoostRegressor(verbose=False))]
        
        ensemble = EnsembleModel(base_models=base_models, 
                                 bagging_fraction=BAGGING_FRACTION,
                                 model_cnt=MODEL_CNT)
            
        model = GroupedOOFModel(base_model=ensemble,
                                group_column='ticker',
                                fold_cnt=FOLD_CNT)

        pipeline = BasePipeline(feature=feature, 
                                target=target, 
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=OUT_NAME)
    
        return pipeline


    def fit(self):
        '''     
        Interface to fit pipeline model. Pre-downloaded appropriate
        data will be used.
        ''' 
        ticker_list = load_tickers()['base_us_stocks']
        result = self.pipeline.fit(self.data_loader, ticker_list)
        print(result)


    def predict(self, tickers):
        '''     
        Interface for model inference.
        
        Parameters
        ----------
        tickers:
            tickers of companies to make inference for
        ''' 
        return self.pipeline.execute(self.data_loader, tickers)





def main():
    '''
    Default model training. Resulted model weights directory path 
    can be changed in `~/.ml_investment/config.json` ``models_path``
    '''
    model = FairMarketcapYahoo(pretrained=False)
    model.fit()
    path = '{}/{}'.format(model.config['models_path'], OUT_NAME)
    model.pipeline.export_core(path)    


if __name__ == '__main__':
   main() 
    
