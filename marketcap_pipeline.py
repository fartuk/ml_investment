import argparse
from features import global_company_feature, quarterely_series_features


columns = ['revenue', 'netinc', 'ncf', 'assets', 'ebitda', 'debt', 'fcf', 'gp', 'workingcapital',
          'cashneq', 'rnd', 'sgna', 'ncfx', 'divyield', 'currentratio', 'netinccmn']

cat_columns = ['sector', 'sicindustry']


class FullFeatures:
    def __init__(self):
        None
    
    def calculate(self):
        None


class MarketcapPipeline:
    def __init__(self, config, base_model, max_back_quarter, feat_columns, config):
        self.fc1 = QuarterlyFeatures(config=config, 
                                     columns=columns,
                                     quarter_counts=[2, 4, 10],
                                     max_back_quarter=10)

        self.fc2 = BaseCompanyFeatures(config, cat_columns)
        self.target = QuarterlyTarget(config=config, col='marketcap', quarter_shift=0)
        self.model = GroupedOOFModel(base_model, fold_cnt=5)


    def _calc_feats(self, tickers): 
        X1 = fc1.calculate(tickers)
        X2 = fc2.calculate(tickers)
        X = pd.merge(X1, X2, on='ticker', how='left')

        return X        

    def _calc_target(info_df):
        return self.target.calculate(info_df)


    def fit(self, tickers):
        X = self._calc_feats(tickers)
        y = self._calc_target(X[['ticker', 'target']])
        self.model.fit(X=X.drop(['ticker', 'date'], axis=1),
                       y=np.log(y['y']), 
                       groups=X['ticker'])


    def execute(self, tickers):
        result = pd.DataFrame()
        result['ticker'] = tickers
        X = calc_feats(tickers)
        pred = self.model.predict(X=X.drop(['ticker', 'date'], axis=1), 
                                  groups=X['ticker'])
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
    
    pipeline = MarketcapPipeline()
    pipeline.fit()
    pipeline.dump()   













