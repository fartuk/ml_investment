import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgbm
from utils import load_json
from data import SF1Data
from features import QuarterlyFeatures, BaseCompanyFeatures, FeatureMerger
from targets import QuarterlyTarget
from models import GroupedOOFModel



class BasePipeline:
    def __init__(self, feature, target, model, metric):
        self.core = {}
        self.core['feature'] = feature 
        self.core['target'] = target
        self.core['model'] = model
        self.metric = metric

    @classmethod
    def load(cls, path):
        pipeline = cls(None, None, None, None)
        pipeline.load_core(path)
        return pipeline


    def fit(self, data_loader, tickers):
        X = self.core['feature'].calculate(data_loader, tickers)
        y = self.core['target'].calculate(data_loader, 
                                          X.index.to_frame(index=False))
        leave_mask = (y['y'].isnull() == False)
        y = y[leave_mask]
        X = X[leave_mask]

        self.core['model'].fit(X, y['y'])
        pred = self.core['model'].predict(X)
        print(self.metric(y['y'].values, pred))


    def execute(self, data_loader, tickers):
        X = self.core['feature'].calculate(data_loader, tickers)
        pred = self.core['model'].predict(X)
        result = pd.DataFrame()
        result['y'] = pred
        result.index = X.index

        return result


    def export_core(self, path=None):
        if path is None:
            now = time.strftime("%d.%m.%y_%H:%M", time.localtime(time.time()))
            path = 'models_data/pipeline_{}'.format(now)

        with open('{}.pickle'.format(path), 'wb') as f:
            pickle.dump(self.core, f)


    def load_core(self, path):
        with open(path, 'rb') as f:
            self.core = pickle.load(f)




class MiltiTargetPipeline:
    def __init__(self):
        None








