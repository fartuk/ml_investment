import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from copy import deepcopy
from typing import List
from utils import load_json, copy_repeat
from data import SF1Data
from features import QuarterlyFeatures, BaseCompanyFeatures, FeatureMerger
from targets import QuarterlyTarget
from models import GroupedOOFModel



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







