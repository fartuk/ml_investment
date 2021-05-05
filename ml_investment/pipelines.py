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
from .utils import copy_repeat, check_create_folder


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
        

    # @classmethod
    # def load(cls, path):
    #     pipeline = cls(None, None, None, None)
    #     pipeline.load_core(path)
    #     return pipeline
    #

    def fit(self, index: List[str], metric):
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

        ''' 
        if type(metric) == list:
            assert len(self.target) == len(metric)
            
        metric = metric if type(metric) == list \
                             else [metric] * len(self.target)
        metrics_result = {}
        X = self.feature.calculate(self.data, index)            
        for k, target in enumerate(self.target):
            y = target.calculate(self.data, 
                                 X.index.to_frame(index=False))
            leave_mask = (y['y'].isnull() == False)
            y_ = y[leave_mask.values]
            X_ = X[leave_mask.values]
            self.core['model'][k].fit(X_, y_['y'])
            
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


    def execute(self, index) -> pd.DataFrame:
        '''     
        Interface for executing pipeline for tickers.
        Features will be based on data from data_loader
        
        Parameters
        ----------
        index:
            identifiers for executing pipelines. I.e. list of companies tickers 

        Returns
        -------
        ``pd.DataFrame``
            combined pipelines execute result
        '''   
        dfs = []
        for pipeline in self.pipeline_list:
            curr_df = pipeline.execute(index)
            dfs.append(curr_df)
            
        result_df = reduce(lambda l, r: pd.merge(
            l, r, on=self.execute_merge_on, how='left'), dfs)

        return result_df
            
            
class QuarterlyLoadPipeline:
    '''
    Wrapper for data_loader for loading quarterly data
    in ``execute(data_loader, tickers:List[str]) -> pd.DataFrame``
    interface
    '''
    def __init__(self, columns:List[str]):
        '''     
        Parameters
        ----------
        columns:
            column names for loading
        '''
        self.columns = columns
       
    def fit(self, data_loader, tickers:List[str]):
        None

    def execute(self, data_loader, tickers:List[str]):
        '''     
        Interface for executing pipeline(lading data) for
        tickers using data_loader.
        
        Parameters
        ----------
        data_loader:
            class implements ``load_quarterly_data(tickers: List[str])``
            -> pd.DataFrame`` interface
            
        tickers:
            tickers of companies to load data for      
                      
        Returns
        -------
        ``pd.DataFrame``
            quarterly data
        '''  
        quarterly_data = data_loader.load_quarterly_data(tickers)
        quarterly_df = quarterly_data[self.columns]
        return quarterly_df            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


