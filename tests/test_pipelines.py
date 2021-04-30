import pytest
import os
import pandas as pd
import numpy as np
import lightgbm as lgbm
import catboost as ctb
from ml_investment.data import SF1Data
from ml_investment.features import QuarterlyFeatures
from ml_investment.targets import QuarterlyTarget
from ml_investment.models import GroupedOOFModel
from ml_investment.pipelines import BasePipeline, MergePipeline, QuarterlyLoadPipeline
from ml_investment.metrics import median_absolute_relative_error, mean_absolute_relative_error
from ml_investment.utils import load_config
from synthetic_data import GeneratedData

config = load_config()


loaders = [GeneratedData()]
if os.path.exists(config['sf1_data_path']):
    loaders.append(SF1Data(config['sf1_data_path']))

tickers = ['AAPL', 'TSLA', 'K', 'MAC', 'NVDA']


class TestBasePipeline:
    def _create_base_components(self):                                    
        columns = ['revenue', 'netinc', 'ncf', 'ebitda', 'debt', 'fcf']
        f1 = QuarterlyFeatures(columns=columns,
                               quarter_counts=[2, 10],
                               max_back_quarter=1)

        target = QuarterlyTarget(col='marketcap', quarter_shift=0)

        model = GroupedOOFModel(lgbm.sklearn.LGBMRegressor(),
                                group_column='ticker', fold_cnt=4)
        
        return f1, target, model
        
        
    @pytest.mark.parametrize('data_loader', loaders)
    def test_fit_execute_simple(self, data_loader):
        f1, target, model = self._create_base_components()
        pipeline = BasePipeline(feature=f1, 
                                target=target,
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=None)

        res = pipeline.fit(data_loader, tickers)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        df = pipeline.execute(data_loader, tickers)
        assert type(df) == pd.DataFrame
        assert df['y_0'].mean() > 0

        
    @pytest.mark.parametrize('data_loader', loaders)
    def test_fit_execute_multi_target(self, data_loader):
        f1, target, model = self._create_base_components()
        target1 = QuarterlyTarget(col='marketcap', quarter_shift=-1)

        pipeline = BasePipeline(feature=f1, 
                                target=[target, target1],
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=None)

        res = pipeline.fit(data_loader, tickers)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        df = pipeline.execute(data_loader, tickers)
        assert type(df) == pd.DataFrame
        assert df['y_0'].mean() > 0   
        assert df['y_1'].mean() > 0   
        assert (df['y_0'] == df['y_1']).min() == False
        
        pipeline = BasePipeline(feature=f1, 
                                target=[target, target],
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=None)

        res = pipeline.fit(data_loader, tickers)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        df = pipeline.execute(data_loader, tickers)
        assert (df['y_0'] == df['y_1']).min() == True
      
        
    @pytest.mark.parametrize('data_loader', loaders)
    def test_fit_execute_multi_target_model(self, data_loader):
        f1, target, model = self._create_base_components()
        target1 = QuarterlyTarget(col='marketcap', quarter_shift=-1)
        model1 = GroupedOOFModel(ctb.CatBoostRegressor(verbose=False),
                                 group_column='ticker', fold_cnt=4)
        pipeline = BasePipeline(feature=f1, 
                                target=[target, target],
                                model=[model, model1], 
                                metric=median_absolute_relative_error,
                                out_name=None)

        res = pipeline.fit(data_loader, tickers)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        df = pipeline.execute(data_loader, tickers)
        assert type(df) == pd.DataFrame
        assert df['y_0'].mean() > 0   
        assert df['y_1'].mean() > 0   
        assert (df['y_0'] == df['y_1']).min() == False
        
 
    @pytest.mark.parametrize('data_loader', loaders)
    def test_fit_execute_multi_target_metric(self, data_loader):
        f1, target, model = self._create_base_components()
        target1 = QuarterlyTarget(col='marketcap', quarter_shift=-1)
        pipeline = BasePipeline(feature=f1, 
                                target=[target, target1],
                                model=model, 
                                metric=[median_absolute_relative_error,
                                        mean_absolute_relative_error],
                                out_name=None)

        res = pipeline.fit(data_loader, tickers)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        assert res['metric_y_0'] < res['metric_y_1']
        
        
    @pytest.mark.parametrize('data_loader', loaders)
    def test_fit_execute_multi_names(self, data_loader):
        f1, target, model = self._create_base_components()
        pipeline = BasePipeline(feature=f1, 
                                target=[target, target],
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=['name1', 'name2'])

        res = pipeline.fit(data_loader, tickers)
        assert type(res) == dict
        assert res['metric_name1'] > 0
        assert res['metric_name2'] > 0
        df = pipeline.execute(data_loader, tickers)
        assert type(df) == pd.DataFrame
        assert df['name1'].mean() > 0   
        assert df['name2'].mean() > 0   
        assert (df['name1'] == df['name2']).min() == True
        
        
    @pytest.mark.parametrize('data_loader', loaders)        
    def test_export_load(self, data_loader, tmpdir):
        f1, target, model = self._create_base_components()
        pipeline = BasePipeline(feature=f1, 
                                target=target,
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name=None)
        res = pipeline.fit(data_loader, tickers)
        df = pipeline.execute(data_loader, tickers)
        pipeline.export_core('{}/pipeline'.format(str(tmpdir)))
        pipeline = BasePipeline.load('{}/pipeline.pickle'.format(str(tmpdir)))
        df1 = pipeline.execute(data_loader, tickers[:100])
        
        np.testing.assert_array_equal(df['y_0'].values, df1['y_0'].values)




class TestMergePipeline:       
    @pytest.mark.parametrize('data_loader', loaders)
    def test_fit_execute_simple(self, data_loader):
        columns = ['revenue', 'netinc', 'ncf', 'ebitda', 'debt', 'fcf']
        f1 = QuarterlyFeatures(columns=columns,
                               quarter_counts=[2, 10],
                               max_back_quarter=1)

        target1 = QuarterlyTarget(col='marketcap', quarter_shift=0)
        target2 = QuarterlyTarget(col='marketcap', quarter_shift=-1)

        model = lgbm.sklearn.LGBMRegressor()
    
        pipeline1 = BasePipeline(feature=f1, 
                                target=target1,
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name='p1')

        pipeline2 = BasePipeline(feature=f1, 
                                target=target2,
                                model=model, 
                                metric=median_absolute_relative_error,
                                out_name='p2')        

        pipeline3 = QuarterlyLoadPipeline(['ticker', 'date', 'marketcap'])

        merge1 = MergePipeline(
            pipeline_list=[pipeline1, pipeline2, pipeline3],
            execute_merge_on=['ticker', 'date'])

        merge1.fit(data_loader, tickers)
        df_m1 = merge1.execute(data_loader, tickers)


        pipeline1.fit(data_loader, tickers)
        pipeline2.fit(data_loader, tickers)
        
        merge2 = MergePipeline(
            pipeline_list=[pipeline1, pipeline2, pipeline3],
            execute_merge_on=['ticker', 'date'])
        
        df1 = pipeline1.execute(data_loader, tickers)
        df2 = pipeline2.execute(data_loader, tickers)
        df3 = pipeline3.execute(data_loader, tickers)
               

        df_m2 = merge1.execute(data_loader, tickers)

        assert type(df_m1) == pd.DataFrame
        assert type(df_m2) == pd.DataFrame
        assert len(df_m1) == len(df1)
        assert len(df_m2) == len(df1)
        np.testing.assert_array_equal(df_m1.columns, 
                                      ['ticker', 'date', 'p1', 'p2', 'marketcap'])

        np.testing.assert_array_equal(df_m2.columns, 
                                      ['ticker', 'date', 'p1', 'p2', 'marketcap'])

        np.testing.assert_array_equal(df1['p1'], df_m1['p1'])        
        np.testing.assert_array_equal(df2['p2'], df_m1['p2'])        
        
        np.testing.assert_array_equal(df_m1['p1'], df_m2['p1'])        
        np.testing.assert_array_equal(df_m1['p2'], df_m2['p2'])        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




