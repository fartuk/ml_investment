import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from copy import deepcopy
from functools import reduce
from typing import List, Dict
from .utils import copy_repeat, check_create_folder, nan_mask

import gc


class Pipeline:
    '''
    Class incapsulate feature and target calculation, 
    model training and validation during fit-phase 
    and feature calculation and model prediction during 
    execute-phase.
    Support multi-target with different models and metrics.
    '''
    def __init__(self, data: Dict, feature, target, model, out_name=None):
        '''     
        Parameters
        ----------
        data:
            dict having needed for features and targets fields.
            This field should contain classes implementing
            ``load(index) -> pd.DataFrame`` interfaces
        feature:
            feature calculator implements 
            ``calculate(data: Dict, index) -> pd.DataFrame`` interface
        target:
            target calculator implements 
            ``calculate(data: Dict, index) -> pd.DataFrame`` interface      
            OR ``List`` of such target calculators
        model:
            class implements ``fit(X, y)`` and ``predict(X)`` interfaces.
            Сopy of the model will be used for every single
            target if type of target is ``List``.           
            OR ``List`` of such classes(len of this list should
            be equal to len of target)
        out_name:
            str column name of result in ``pd.DataFrame`` after 
            :func:`~ml_investment.pipelines.Pipeline.execute`
            OR ``List[str]`` (len of this list should be equal to 
            len of target)
            OR ``None`` ( ``List['y_0', 'y_1'...]`` will be used in this case)
        '''
        self.core = {}
        self.data = data
        self.feature = feature    
        
        if type(target) == list and type(model) == list:
            assert len(target) == len(model)
            
        if type(target) == list and type(out_name) == list:
            assert len(target) == len(out_name)
            
            
        self.target = target if type(target) == list else [target]
        target_len = len(self.target)
        self.core['model'] = model if type(model) == list else \
                             copy_repeat(model, target_len)
        if out_name is None:
            self.out_name = ['y_{}'.format(k) for k in range(target_len)]
        if type(out_name) is str:
            self.out_name = [out_name]
        if type(out_name) == list:
            self.out_name = out_name
        

    def fit(self, index: List[str], metric=None, target_filter_foo=nan_mask):
        '''     
        Interface to fit pipeline model for tickers.
        Features and target will be based on data from data_loader
        
        Parameters
        ----------
        index:
             fit identification(i.e. list of tickers to fit model for)
        metric:
            function implements ``foo(gt, y) -> float`` interface.
            The same metric will be used for every single target 
            if type of target is ``List``.
            OR ``List`` of such functions(len of this list should be equal to 
            len of target)
        target_filter_foo:
            function for filtering samples according target values/
            Should implement ``foo(arr) -> np.array[bool]`` interface. 
            Len of resulted array should be equal to len of arr.
            OR ``List`` of such functions(len of this list should be equal to 
            len of target)
        ''' 
        if type(metric) == list:
            assert len(self.target) == len(metric)
        
        if type(target_filter_foo) == list:
            assert len(self.target) == len(target_filter_foo)
            
        metric = metric if type(metric) == list \
                             else [metric] * len(self.target)

        target_filter_foo = target_filter_foo if type(target_filter_foo) == list \
                             else [target_filter_foo] * len(self.target)

        metrics_result = {}
        X = self.feature.calculate(self.data, index)            
        for k, target in enumerate(self.target):
            y = target.calculate(self.data, 
                                 X.index.to_frame(index=False))
            #leave_mask = (y['y'].isnull() == False)
            leave_mask = target_filter_foo[k](y['y'].values)

            y_ = y[leave_mask]
            X_ = X[leave_mask]
            self.core['model'][k].fit(X_, y_['y'])
            
            if metric[0] is not None:
                pred = self.core['model'][k].predict(X_)
                metric_name = 'metric_{}'.format(self.out_name[k])
                metrics_result[metric_name] = metric[k](y_['y'].values, pred)
            
        return metrics_result


    def execute(self, index):
        '''     
        Interface for executing pipeline for tickers.
        Features will be based on data from data_loader
        
        Parameters
        ----------
        index:
             execute identification(i.e. list of tickers to predict model for)
                      
        Returns
        -------
        ``pd.DataFrame``
            result values in columns named as ``out_name`` param in
            :func:`~ml_investment.pipelines.Pipeline.__init__`
        '''   
        result = pd.DataFrame()
        X = self.feature.calculate(self.data, index)
        for k, target in enumerate(self.target):
            pred = self.core['model'][k].predict(X)
            result[self.out_name[k]] = pred
        result.index = X.index

        return result


    def export_core(self, path=None):
        '''     
        Interface for saving pipelines core
        
        Parameters
        ----------
        path:
            str with path to store pipeline core
            OR ``None`` (path will be generated automatically)
        '''   
        if path is None:
            now = time.strftime("%d.%m.%y_%H:%M", time.localtime(time.time()))
            path = 'models_data/pipeline_{}'.format(now)

        check_create_folder(path)
        with open('{}.pickle'.format(path), 'wb') as f:
            pickle.dump(self.core, f)


    def load_core(self, path):
        '''     
        Interface for loading pipeline core
        
        Parameters
        ----------
        path:
            str with path to load pipeline core from
        '''  
        with open(path, 'rb') as f:
            self.core = pickle.load(f)



class MergePipeline:
    '''
    Class combining list of pipelines to single pipilene.
    '''
    def __init__(self, pipeline_list:List, execute_merge_on):
        '''     
        Parameters
        ----------
        pipeline_list:
            list of classes implementing 
            ``fit(index)`` and 
            ``execute(index) -> pd.DataFrame()`` interfaces.
            Order is important: merging results during
            :func:`~ml_investment.pipelines.MergePipeline.execute`
            will be done from left to right.
        execute_merge_on:
            column names for merging pipelines results on.

        '''
        self.pipeline_list = pipeline_list
        self.execute_merge_on = execute_merge_on


    def fit(self, index):
        '''
        Interface for training all pipelines

        Parameters
        ----------
        index:
            identifiers for fit pipelines. I.e. list of companies tickers 
        '''
        for pipeline in self.pipeline_list:
            pipeline.fit(index)


    def _single_batch(self, batch):
        dfs = []
        for pipeline in self.pipeline_list:
            dfs.append(pipeline.execute(batch))
            
        batch_result = reduce(lambda l, r: pd.merge(
            l, r, on=self.execute_merge_on, how='left'), dfs)

        return batch_result


    def execute(self, index, batch_size=None) -> pd.DataFrame:
        '''     
        Interface for executing pipeline for tickers.
        Features will be based on data from data_loader
        
        Parameters
        ----------
        index:
            identifiers for executing pipelines. I.e. list of companies tickers 
        batch_size:
            size of batch for execute separation(may be usefull
            for lower memory usage).
            OR ``None`` (for full-size executing)

        Returns
        -------
        ``pd.DataFrame``
            combined pipelines execute result
        '''
        if batch_size is None:
            batch_size = len(index)
        batches = [index[k:k+batch_size] 
                    for k in range(0, len(index), batch_size)]
        result = []
        for batch in batches:
            result.append(self._single_batch(batch))
           
        result = pd.concat(result, axis=0)

        return result
            
            
class LoadingPipeline:
    '''
    Wrapper for data loaders for loading data
    in ``execute(index) -> pd.DataFrame``
    interface
    '''
    def __init__(self, data_loader, columns:List[str]):
        '''     
        Parameters
        ----------
        data_loader:
            class implements ``load(index) -> pd.DataFrame`` interface
        columns:
            column names for loading
        '''
        self.data_loader = data_loader
        self.columns = columns
       
    def fit(self, index):
        None

    def execute(self, index):
        '''     
        Interface for executing pipeline(lading data) for
        tickers.
        
        Parameters
        ----------
        index:
            inentification for loading data, i.e. list of tickers
                      
        Returns
        -------
        ``pd.DataFrame``
            resulted data
        '''  
        quarterly_data = self.data_loader.load(index)
        quarterly_df = quarterly_data[self.columns]
        return quarterly_df            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


