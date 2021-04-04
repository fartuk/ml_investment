import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from copy import deepcopy
from functools import reduce
from typing import List
from .utils import copy_repeat


class BasePipeline:
    '''
    Class incapsulate feature and target calculation, 
    model training and validation during fit-phase 
    and feature calculation and model prediction during 
    execute-phase.
    Support multi-target with different models and metrics.
    '''
    def __init__(self, feature, target, model, metric, out_name=None):
        '''     
        Parameters
        ----------
        feature:
            feature calculator implements 
                calculate(data_loader, tickers: List[str]) -> 
                                                pd.DataFrame interface
        target:
            target calculator implements 
                calculate(data_loader, info_df: pd.DataFrame) ->
                                                pd.DataFrame interface      
            OR List of such target calculators
        model:
            class implements 
                fit(X, y)
                predict(X) interfaces
            Сopy of the model will be used for every single target if 
            type of target is List.           
            OR List of such classes(len of this list should be equal to 
            len of target)
        metric:
            function implements (gt, y) -> float interface
            The same metric will be used for every single target if 
            type of target is List.
            OR List of such functions(len of this list should be equal to 
            len of target)
        out_name:
            str column name of result in pd.DataFrame after self.execute(...)
            OR List[str](len of this list should be equal to 
            len of target)
            OR None(List['y_0', 'y_1'...] will be used in this case)
        '''
        self.core = {}
        self.core['feature'] = feature    
        
        if type(target) == list and type(model) == list:
            assert len(target) == len(model)
            
        if type(target) == list and type(metric) == list:
            assert len(target) == len(metric)
            
        if type(target) == list and type(out_name) == list:
            assert len(target) == len(out_name)
            
            
        self.core['target'] = target if type(target) == list else [target]
        target_len = len(self.core['target'])
        self.core['model'] = model if type(model) == list else \
                             copy_repeat(model, target_len)
        if out_name is None:
            self.core['out_name'] = ['y_{}'.format(k) for k in range(target_len)]
        if type(out_name) is str:
            self.core['out_name'] = [out_name]
        if type(out_name) == list:
            self.core['out_name'] = out_name
        
        self.metric = metric if type(metric) == list \
                             else [metric] * target_len

    @classmethod
    def load(cls, path):
        pipeline = cls(None, None, None, None)
        pipeline.load_core(path)
        return pipeline


    def fit(self, data_loader, tickers: List[str]):
        '''     
        Interface to fit pipeline model for tickers.
        Features and target will be based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements needed for feature.calculate()
            interfaces
        tickers:
            tickers of companies to fit model for
        ''' 
        metrics = {}
        X = self.core['feature'].calculate(data_loader, tickers)            
        for k, target in enumerate(self.core['target']):
            y = target.calculate(data_loader, 
                                 X.index.to_frame(index=False))
            leave_mask = (y['y'].isnull() == False)
            y_ = y[leave_mask]
            X_ = X[leave_mask]
            self.core['model'][k].fit(X_, y_['y'])
            
            pred = self.core['model'][k].predict(X_)
            metric_name = 'metric_{}'.format(self.core['out_name'][k])
            metrics[metric_name] = self.metric[k](y_['y'].values, pred)
            
        return metrics


    def execute(self, data_loader, tickers):
        '''     
        Interface for executing pipeline for tickers.
        Features will be based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements needed for feature.calculate()
            interfaces
        tickers:
            tickers of companies to fit model for       
                      
        Returns
        -------
            pd.DataFrame with result values in columns 
            named as self.core['out_name']
        '''   
        result = pd.DataFrame()
        X = self.core['feature'].calculate(data_loader, tickers)
        for k, target in enumerate(self.core['target']):
            pred = self.core['model'][k].predict(X)
            result[self.core['out_name'][k]] = pred
        result.index = X.index

        return result


    def export_core(self, path=None):
        '''     
        Interface for saving pipeline core
        
        Parameters
        ----------
        path:
            str with path to store pipeline core
            OR None(path will be generated automatically)
        '''   
        if path is None:
            now = time.strftime("%d.%m.%y_%H:%M", time.localtime(time.time()))
            path = 'models_data/pipeline_{}'.format(now)

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




class ExecuteMergePipeline:
    '''
    Class combining list of executive pipelines to 
    single pipilene.
    '''
    def __init__(self, pipeline_list:List, on):
        '''     
        Parameters
        ----------
        pipeline_list:
            list of classes implementing 
                execute(data_loader, tickers) -> 
                            pd.DataFrame interfaces
            Order is important: merging results during execute()
            will be done from left to right.
        on:
            column names for merging pipelines results on.

        '''
        self.pipeline_list = pipeline_list
        self.on = on
        
    def execute(self, data_loader, tickers:List[str]):
        '''     
        Interface for executing pipeline for tickers.
        Features will be based on data from data_loader
        
        Parameters
        ----------
        data_loader:
            class implements all interfaces needed for all 
            pipelines feature calculators
            
        tickers:
            tickers of companies to execute pipeline for       
                      
        Returns
        -------
            pd.DataFrame with resulted pipelines values
        '''   
        dfs = []
        for pipeline in self.pipeline_list:
            curr_df = pipeline.execute(data_loader, tickers)
            dfs.append(curr_df)
            
        result_df = reduce(lambda l, r: pd.merge(l, r, on=self.on, how='left'), dfs)

        return result_df
            
            
class QuarterlyLoadPipeline:
    '''
    Wrapper for data_loader for loading quarterly data
    in execute(data_loader, tickers:List[str]) -> pd.DataFrame
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
        
    def execute(self, data_loader, tickers:List[str]):
        '''     
        Interface for executing pipeline(lading data) for
        tickers using data_loader.
        
        Parameters
        ----------
        data_loader:
            class implements load_quarterly_data(tickers: List[str]) -> 
                                                 pd.DataFrame interface
            
        tickers:
            tickers of companies to load data for      
                      
        Returns
        -------
            pd.DataFrame with quarterly data
        '''  
        quarterly_data = data_loader.load_quarterly_data(tickers)
        quarterly_df = quarterly_data[self.columns]
        return quarterly_df            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


