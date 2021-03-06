import pytest
import os
import pandas as pd
import numpy as np
import lightgbm as lgbm
from data import SF1Data
from features import QuarterlyFeatures
from targets import QuarterlyTarget
from models import GroupedOOFModel
from pipelines import BasePipeline
from metrics import median_absolute_relative_error
from utils import load_json
config = load_json('config.json')




class TestBasePipeline:
    def _create_pipeline(self):
        columns = ['revenue', 'netinc', 'ncf', 'ebitda', 'debt', 'fcf']
        features = QuarterlyFeatures(columns=columns,
                                     quarter_counts=[2, 10],
                                     max_back_quarter=1)

        target = QuarterlyTarget(col='marketcap', quarter_shift=0)

        model = GroupedOOFModel(lgbm.sklearn.LGBMRegressor(),
                                group_column='ticker', fold_cnt=4)

        pipeline = BasePipeline(features, target, model, 
                                metric=median_absolute_relative_error)
                                
        return pipeline
                                
                                        
    def test_fit_execute(self):
        data_loader = SF1Data(config['sf1_data_path'])
        tickers_df = data_loader.load_base_data(
                                currency='USD',
                                scalemarketcap=['5 - Large'])
        tickers = tickers_df['ticker'].unique().tolist()
        pipeline = self._create_pipeline()
        res = pipeline.fit(data_loader, tickers[:800])
        
        assert type(res) == dict
        assert res['metric'] < 0.5
        
        df = pipeline.execute(data_loader, tickers[800:])
        assert type(df) == pd.DataFrame
        assert 'y' in df
        assert df['y'].mean() > 0
        X = pipeline.core['feature'].calculate(data_loader, tickers[800:])
        assert len(X) == len(df)
        

    def test_export_load(self, tmpdir):
        data_loader = SF1Data(config['sf1_data_path'])
        tickers_df = data_loader.load_base_data(
                                currency='USD',
                                scalemarketcap=['5 - Large'])
        tickers = tickers_df['ticker'].unique().tolist()
        pipeline = self._create_pipeline()
        res = pipeline.fit(data_loader, tickers[:100])
        df = pipeline.execute(data_loader, tickers[:100])
        pipeline.export_core('{}/pipeline'.format(str(tmpdir)))
        #assert str(tmpdir) == 'efef'
        pipeline = BasePipeline.load('{}/pipeline.pickle'.format(str(tmpdir)))
        df1 = pipeline.execute(data_loader, tickers[:100])
        
        np.testing.assert_array_equal(df['y'].values, df1['y'].values)









