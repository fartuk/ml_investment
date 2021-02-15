import argparse
import numpy as np
import pandas as pd




class MarketcapPipeline:
    def __init__(self, config, feature, target, model):
        self.feature = feature
        self.target = target
        self.model = model


    def _eval(self, X, y, groups):    
        pred = self.model.predict(X=X, groups=groups)
        pred = np.exp(pred)                         
        print((np.abs(y - pred) / y).mean())


    def fit(self, tickers):
        X = self.feature.calculate(tickers)
        y = self.target.calculate(X[['ticker', 'date']])
        leave_mask = (y['y'] >= 2e9) * (y['y'].isnull() == False)
        y = y[leave_mask].reset_index(drop=True)
        X = X[leave_mask].reset_index(drop=True)
        
        self.model.fit(X=X.drop(['ticker', 'date'], axis=1),
                       y=np.log(y['y']), 
                       groups=X['ticker'])

        self._eval(X=X.drop(['ticker', 'date'], axis=1),
                   y=y['y'].values,
                   groups=X['ticker'])                          
                                  
                                  
    def execute(self, tickers):
        X = self.feature.calculate(tickers)
        pred = self.model.predict(X=X.drop(['ticker', 'date'], axis=1), 
                                  groups=X['ticker'])
        result = X[['ticker', 'date']]
        result['y'] = np.exp(pred)

        return result


    def export_core(self):
        None


    def load_core(self):
        None
        
        

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--txt_path', type=str)
    args = parser.parse_args()
    
    columns = ['revenue', 'netinc', 'ncf', 'assets', 'ebitda', 'debt', 'fcf', 'gp', 'workingcapital',
              'cashneq', 'rnd', 'sgna', 'ncfx', 'divyield', 'currentratio', 'netinccmn']

    cat_columns = ['sector', 'sicindustry']

    fc1 = QuarterlyFeatures(config=config, 
                            columns=columns,
                            quarter_counts=[2, 4, 10],
                            max_back_quarter=10)

    fc2 = BaseCompanyFeatures(config=config, cat_columns=cat_columns)

    feature = FeatureMerger(fc1, fc2, on='ticker')
    target = QuarterlyTarget(config=config, col='marketcap', quarter_shift=0)
    model = GroupedOOFModel(lgbm.sklearn.LGBMRegressor(), fold_cnt=5)
                    
    marketcap_pipeline = MarketcapPipeline(config, feature, target, model)
    marketcap_pipeline.fit(ticker_list)












