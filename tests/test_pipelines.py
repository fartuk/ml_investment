import pytest
import os
import pandas as pd
import numpy as np
import lightgbm as lgbm
import catboost as ctb
from ml_investment.features import QuarterlyFeatures
from ml_investment.targets import QuarterlyTarget
from ml_investment.models import GroupedOOFModel
from ml_investment.pipelines import Pipeline, MergePipeline, LoadingPipeline
from ml_investment.metrics import mean_absolute_relative_error, median_absolute_relative_error
from ml_investment.utils import load_config
from ml_investment.data_loaders.sf1 import SF1QuarterlyData, SF1BaseData,\
                                           SF1DailyData
from synthetic_data import GenQuarterlyData, GenBaseData, GenDailyData

config = load_config()


gen_data = {
    'quarterly': GenQuarterlyData(),  
    'base': GenBaseData(),        
    'daily': GenDailyData(),       
}

datas = [gen_data]
if os.path.exists(config['sf1_data_path']):
    sf1_data = {
        'quarterly': SF1QuarterlyData(config['sf1_data_path']),
        'base': SF1BaseData(config['sf1_data_path']),
        'daily': SF1DailyData(config['sf1_data_path']),
    }
    datas.append(sf1_data)
 


tickers = ['AAPL', 'TSLA', 'K', 'MAC', 'NVDA']


class TestBasePipeline:
    def _create_base_components(self):                                    
        columns = ['revenue', 'netinc', 'ncf', 'ebitda', 'debt', 'fcf']
        f1 = QuarterlyFeatures(data_key='quarterly',
                               columns=columns,
                               quarter_counts=[2, 10],
                               max_back_quarter=1)

        target = QuarterlyTarget(data_key='quarterly',
                                 col='marketcap',
                                 quarter_shift=0)

        model = GroupedOOFModel(lgbm.sklearn.LGBMRegressor(),
                                group_column='ticker', fold_cnt=4)
        
        return f1, target, model
        
        
    @pytest.mark.parametrize('data', datas)
    def test_fit_execute_simple(self, data):
        f1, target, model = self._create_base_components()
        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=target,
                            model=model, 
                            out_name=None)

        res = pipeline.fit(tickers, metric=median_absolute_relative_error)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        df = pipeline.execute(tickers)
        assert type(df) == pd.DataFrame
        assert df['y_0'].mean() > 0

        
    @pytest.mark.parametrize('data', datas)
    def test_fit_execute_multi_target(self, data):
        f1, target, model = self._create_base_components()
        target1 = QuarterlyTarget(data_key='quarterly', 
                                  col='marketcap',
                                  quarter_shift=-1)

        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=[target, target1],
                            model=model, 
                            out_name=None)

        res = pipeline.fit(tickers, metric=median_absolute_relative_error)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        df = pipeline.execute(tickers)
        assert type(df) == pd.DataFrame
        assert df['y_0'].mean() > 0   
        assert df['y_1'].mean() > 0   
        assert (df['y_0'] == df['y_1']).min() == False
        
        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=[target, target],
                            model=model, 
                            out_name=None)

        res = pipeline.fit(tickers, metric=median_absolute_relative_error)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        df = pipeline.execute(tickers)
        assert (df['y_0'] == df['y_1']).min() == True
      
        
    @pytest.mark.parametrize('data', datas)
    def test_fit_execute_multi_target_model(self, data):
        f1, target, model = self._create_base_components()
        target1 = QuarterlyTarget(data_key='quarterly',
                                  col='marketcap', 
                                  quarter_shift=-1)
        model1 = GroupedOOFModel(ctb.CatBoostRegressor(verbose=False),
                                 group_column='ticker', fold_cnt=4)
        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=[target, target],
                            model=[model, model1], 
                            out_name=None)

        res = pipeline.fit(tickers, metric=median_absolute_relative_error)
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        df = pipeline.execute(tickers)
        assert type(df) == pd.DataFrame
        assert df['y_0'].mean() > 0   
        assert df['y_1'].mean() > 0   
        assert (df['y_0'] == df['y_1']).min() == False
        
 
    @pytest.mark.parametrize('data', datas)
    def test_fit_execute_multi_target_metric(self, data):
        f1, target, model = self._create_base_components()
        target1 = QuarterlyTarget(data_key='quarterly',
                                  col='marketcap', 
                                  quarter_shift=-1)
        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=[target, target1],
                            model=model, 
                            out_name=None)

        res = pipeline.fit(tickers, metric=[median_absolute_relative_error,
                                            mean_absolute_relative_error])
        assert type(res) == dict
        assert res['metric_y_0'] > 0
        assert res['metric_y_1'] > 0
        assert res['metric_y_0'] < res['metric_y_1']
        
        
    @pytest.mark.parametrize('data', datas)
    def test_fit_execute_multi_names(self, data):
        f1, target, model = self._create_base_components()
        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=[target, target],
                            model=model, 
                            out_name=['name1', 'name2'])

        res = pipeline.fit(tickers, metric=median_absolute_relative_error)
        assert type(res) == dict
        assert res['metric_name1'] > 0
        assert res['metric_name2'] > 0
        df = pipeline.execute(tickers)
        assert type(df) == pd.DataFrame
        assert df['name1'].mean() > 0   
        assert df['name2'].mean() > 0   
        assert (df['name1'] == df['name2']).min() == True
        
        
    @pytest.mark.parametrize('data', datas)        
    def test_export_load(self, data, tmpdir):
        f1, target, model = self._create_base_components()
        pipeline = Pipeline(data=data,
                            feature=f1, 
                            target=target,
                            model=model, 
                            out_name=None)
        res = pipeline.fit(tickers, metric=median_absolute_relative_error)
        df = pipeline.execute(tickers)
        pipeline.export_core('{}/pipeline'.format(str(tmpdir)))
        pipeline.load_core('{}/pipeline.pickle'.format(str(tmpdir)))
        df1 = pipeline.execute(tickers)
        
        np.testing.assert_array_equal(df['y_0'].values, df1['y_0'].values)




class TestMergePipeline:       
    @pytest.mark.parametrize('data', datas)
    def test_fit_execute_simple(self, data):
        columns = ['revenue', 'netinc', 'ncf', 'ebitda', 'debt', 'fcf']
        f1 = QuarterlyFeatures(data_key='quarterly',
                               columns=columns,
                               quarter_counts=[2, 10],
                               max_back_quarter=1)

        target1 = QuarterlyTarget(data_key='quarterly',
                                  col='marketcap',
                                  quarter_shift=0)

        target2 = QuarterlyTarget(data_key='quarterly',
                                  col='marketcap',
                                  quarter_shift=-1)

        model = lgbm.sklearn.LGBMRegressor()

        pipeline1 = Pipeline(data=data,
                             feature=f1, 
                             target=target1,
                             model=model, 
                             out_name='p1')

        pipeline2 = Pipeline(data=data,
                             feature=f1, 
                             target=target2,
                             model=model, 
                             out_name='p2')        

        pipeline3 = LoadingPipeline(data['quarterly'], ['ticker', 'date', 'marketcap'])

        merge1 = MergePipeline(
            pipeline_list=[pipeline1, pipeline2, pipeline3],
            execute_merge_on=['ticker', 'date'])

        merge1.fit(tickers)
        df_m1 = merge1.execute(tickers)


        pipeline1.fit(tickers)
        pipeline2.fit(tickers)

        merge2 = MergePipeline(
            pipeline_list=[pipeline1, pipeline2, pipeline3],
            execute_merge_on=['ticker', 'date'])

        df1 = pipeline1.execute(tickers)
        df2 = pipeline2.execute(tickers)
        df3 = pipeline3.execute(tickers)


        df_m2 = merge1.execute(tickers)

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

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




