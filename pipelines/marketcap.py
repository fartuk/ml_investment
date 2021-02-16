import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from utils import load_json
from features import QuarterlyFeatures, BaseCompanyFeatures, FeatureMerger
from targets import QuarterlyTarget
from models import GroupedOOFModel



def load(folder_path):
    pipeline = MarketcapPipeline(None, None, None)
    pipeline.load_core(folder_path)
    
    return pipeline
    
    

class MarketcapPipeline:
    def __init__(self, feature, target, model):
        self.feature = feature
        self.target = target
        self.model = model


    def _eval(self, X, y, groups):    
        pred = self.model.predict(X=X, groups=groups)
        pred = np.exp(pred)                         
        print((np.abs(y - pred) / y).mean())


    def fit(self, config, tickers):
        X = self.feature.calculate(config['data_path'], tickers)
        y = self.target.calculate(config['data_path'], X[['ticker', 'date']])
        leave_mask = (y['y'] >= 2e9) * (y['y'].isnull() == False)
        y = y[leave_mask].reset_index(drop=True)
        X = X[leave_mask].reset_index(drop=True)
        
        self.model.fit(X=X.drop(['ticker', 'date'], axis=1),
                       y=np.log(y['y']), 
                       groups=X['ticker'])

        self._eval(X=X.drop(['ticker', 'date'], axis=1),
                   y=y['y'].values,
                   groups=X['ticker'])                          
                                  
                                  
    def execute(self, config, tickers):
        X = self.feature.calculate(config['data_path'], tickers)
        pred = self.model.predict(X=X.drop(['ticker', 'date'], axis=1), 
                                  groups=X['ticker'])
        result = X[['ticker', 'date']]
        result['y'] = np.exp(pred)

        return result


    def export_core(self, folder_path=None):
        if folder_path is None:
            now = time.strftime("%d.%m.%y_%H:%M", time.localtime(time.time()))
            folder_path = 'models_data/marketcap_pipeline_{}'.format(now)
        
        os.makedirs(folder_path, exist_ok=True)    
            
        with open('{}/feature.pickle'.format(folder_path), 'wb') as f:
            pickle.dump(self.feature, f)
            
        with open('{}/target.pickle'.format(folder_path), 'wb') as f:
            pickle.dump(self.target, f)

        with open('{}/model.pickle'.format(folder_path), 'wb') as f:
            pickle.dump(self.model, f)
            

    def load_core(self, folder_path):
        with open('{}/feature.pickle'.format(folder_path), 'rb') as f:
            self.feature = pickle.load(f)
        
        with open('{}/target.pickle'.format(folder_path), 'rb') as f:
            self.target = pickle.load(f)
            
        with open('{}/model.pickle'.format(folder_path), 'rb') as f:
            self.model = pickle.load(f)        

 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)

    tickers_df = pd.read_csv('{}/cf1/tickers.csv'.format(config['data_path']))
    tickers_df = tickers_df[tickers_df['currency']=='USD']
    tickers_df = tickers_df[tickers_df['scalemarketcap'].apply(lambda x:
                                 x in ['4 - Mid', '5 - Large', '6 - Mega'])]
    ticker_list = tickers_df['ticker'].unique().tolist()

    fc1 = QuarterlyFeatures(
        columns=config['pipelines']['marketcap']['quarter_columns'],
        quarter_counts=config['pipelines']['marketcap']['quarter_counts'],
        max_back_quarter=config['pipelines']['marketcap']['max_back_quarter'])

    fc2 = BaseCompanyFeatures(
        cat_columns=config['pipelines']['marketcap']['cat_columns'])

    feature = FeatureMerger(fc1, fc2, on='ticker')
    target = QuarterlyTarget(col='marketcap', quarter_shift=0)
    model = GroupedOOFModel(lgbm.sklearn.LGBMRegressor(), fold_cnt=5)
                    
    mc_pipeline = MarketcapPipeline(feature, target, model)
    mc_pipeline.fit(config, ticker_list)
    mc_pipeline.export_core()














