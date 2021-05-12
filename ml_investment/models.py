import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import List
from sklearn.model_selection import GroupKFold



class LogExpModel:
    '''
    Model wrapper to fit on log of target and exp produced prediction.
    May be usefull for some target distributions.
    '''
    def __init__(self, base_model):
        '''
        Parameters
        ----------
        base_model:
            class implements ``fit(X, y)``, 
            ``predict(X)``/``predict_proba(X)`` interfaces
        '''
        self.base_model = base_model
        
    def fit(self, X: pd.DataFrame, y):
        '''
        Interface for model training
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features
        y:
            target data
        ''' 
#         assert np.min(y) >= 0
        mask = (y > 0).values
        self.base_model.fit(X[mask], np.log(y[mask]))
   
    def predict(self, X):
        '''     
        Interface for prediction
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features
        ''' 
        return np.exp(self.base_model.predict(X))


class EnsembleModel:
    '''
    Class for training ansamble of base models. 
    '''
    def __init__(self, base_models: List, bagging_fraction: float=0.8, 
                 model_cnt: int=20):
        '''     
        Parameters
        ----------
        base_models:
            list of classes implements ``fit(X, y)``,
            ``predict(X)``/``predict_proba(X)`` interfaces 
        bagging_fraction:
            part of random data subsample for training models
        model_cnt:
            total number of models in resulted ansamble
        '''
        self.base_models = base_models
        self.bagging_fraction = bagging_fraction
        self.model_cnt = model_cnt
        self.models = []
        
     
    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        Interface for model training
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features
        y:
            target data
        ''' 
        for _ in tqdm(range(self.model_cnt)):
            idxs = np.random.randint(0, len(X), 
                                     int(len(X) * self.bagging_fraction))
            curr_model = deepcopy(np.random.choice(self.base_models))
            curr_model.fit(X.iloc[idxs], y.iloc[idxs])
            self.models.append(curr_model)
                
    
    def predict(self, X):
        '''     
        Interface for prediction
        
        Parameters
        ----------
        X:
            pd.DataFrame containing features
        ''' 
        preds = []
        for k in range(self.model_cnt):
            try:
                model_pred = self.models[k].predict_proba(X)[:, 1]
            except:
                model_pred = self.models[k].predict(X)
                
            preds.append(model_pred)
        
        return np.mean(preds, axis=0)         
                        


class GroupedOOFModel:
    '''
    Model wrapper incapsulate out of fold separation within data groups. 
    Each sample in group can not be in training and validation fold 
    at the same time.
    '''
    def __init__(self, base_model, group_column: str, fold_cnt: int=5):
        '''     
        Parameters
        ----------
        base_model:
            model implements ``fit(X, y)``,
            ``predict(X)``/``predict_proba(X)`` interfaces 
        group_column:
            name of column for grouping training data. 
            ``X`` in ``fit(X, y)`` and ``predict(X)`` 
            should contain this column. 
            Samples with one group value 
            will be placed only in one training fold.
        fold_cnt:
            number of folds for training
        '''
        self.fold_cnt = fold_cnt
        self.group_column = group_column
        self.base_models = []
        for k in range(self.fold_cnt):
            self.base_models.append(deepcopy(base_model))        
        self.group_df = None
        self.columns = None
       

    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''     
        Interface for model training
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features and  ``self.group_column``
        y:
            target data
        ''' 
        groups = X.reset_index()[self.group_column]
        df_arr = []
        kfold = GroupKFold(self.fold_cnt)
        for k, (itr, ite) in enumerate(kfold.split(X, y, groups)):
            self.base_models[k].fit(X.iloc[itr], y.iloc[itr])

            curr_group_df = pd.DataFrame()
            curr_group_df['group'] = np.unique(groups[ite])
            curr_group_df['fold_id'] = k
            df_arr.append(curr_group_df)

        self.group_df = pd.concat(df_arr, axis=0)
        self.columns = X.columns
        
        
    def predict(self, X: pd.DataFrame) -> np.array:
        '''     
        Interface for prediction
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features and  ``self.group_column``
        ''' 
        groups = X.reset_index()[self.group_column]
        predict_groups = pd.DataFrame()
        predict_groups['group'] = groups
        predict_groups = pd.merge(predict_groups, self.group_df,
                                  on='group', how='left')
        predict_groups.index = X.index
        # If group was not in train data -> put to 0th fold
        predict_groups = predict_groups.fillna(0)
        pred_df = []
        for fold_id in range(self.fold_cnt):
            X_curr = X[predict_groups['fold_id'] == fold_id]
            if len(X_curr) == 0:
                continue
            try:
                pred = self.base_models[fold_id].predict_proba(X_curr)[:, 1]
            except:
                pred = self.base_models[fold_id].predict(X_curr)

            curr_pred_df = pd.DataFrame()
            curr_pred_df['pred'] = pred
            curr_pred_df.index = X_curr.index
            pred_df.append(curr_pred_df)
        
        pred_df = pd.concat(pred_df, axis=0)
        pred_df = pred_df.loc[X.index]
        
        return pred_df['pred'].values


class TimeSeriesOOFModel:
    '''
    Model wrapper incapsulate out of fold time-series separation. 
    '''
    def __init__(self, base_model, time_column: str, fold_cnt: int=5):
        '''     
        Parameters
        ----------
        base_model:
            model implements ``fit(X, y)``,
            ``predict(X)``/``predict_proba(X)`` interfaces 
        time_column:
            name of column for separating training data. 
            ``X`` in ``fit(X, y)`` and ``predict(X)``
            should contain this column. 
            Samples from feature would not be used 
            for training and prediction past.
        fold_cnt:
            number of folds for training
        '''
        self.fold_cnt = fold_cnt
        self.time_column = time_column
        self.base_models = []
        for k in range(self.fold_cnt):
            self.base_models.append(deepcopy(base_model))
                    
        self.time_bounds = None
        self.is_fitted_fold = np.zeros(self.fold_cnt)
   
    def _create_time_bounds(self, times: List[np.datetime64]):
        max_time = max(times)
        min_time = min(times)
        delta = (max_time - min_time) // self.fold_cnt
        self.time_bounds = []
        for fold_id in range(1, self.fold_cnt):
            self.time_bounds.append(min_time + fold_id * delta)
        self.time_bounds.append(max_time)
        # Fictive boundary for fit() code simplification
        self.time_bounds.append(max_time + np.timedelta64(10000, 'D'))
                
   
    def fit(self, X: pd.DataFrame, y):
        '''     
        Interface for model training
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features and ``self.time_column``
        y:
            target data
        ''' 
        times = X.reset_index()[self.time_column].astype(np.datetime64).values
        self._create_time_bounds(times)
        for fold_id in range(self.fold_cnt):
            curr_mask = times <= self.time_bounds[fold_id]
            # check if there are enough samples
            if curr_mask.sum() > 5:
                self.base_models[fold_id].fit(X[curr_mask], y[curr_mask])
                self.is_fitted_fold[fold_id] = 1
          
    def predict(self, X: pd.DataFrame) -> np.array:
        '''     
        Interface for prediction
        
        Parameters
        ----------
        X:
            ``pd.DataFrame`` containing features and ``self.time_column``
        ''' 
        times = X.reset_index()[self.time_column].astype(np.datetime64).values
        pred_df = []
        X_curr = X[times <= self.time_bounds[0]]
        curr_pred_df = pd.DataFrame()
        curr_pred_df['pred'] = [np.nan] * len(X_curr)
        curr_pred_df.index = X_curr.index
        pred_df.append(curr_pred_df)                  
        for fold_id in range(self.fold_cnt):
            curr_mask = (times > self.time_bounds[fold_id]) * \
                        (times <= self.time_bounds[fold_id + 1])
            X_curr = X[curr_mask]
            if len(X_curr) == 0:
                continue

            if not self.is_fitted_fold[fold_id]:
                curr_pred_df = pd.DataFrame()
                curr_pred_df['pred'] = [np.nan] * len(X_curr)
                curr_pred_df.index = X_curr.index
                pred_df.append(curr_pred_df)
                continue

            try:   
                pred = self.base_models[fold_id].predict_proba(X_curr)[:, 1]         
            except:
                pred = self.base_models[fold_id].predict(X_curr)

            curr_pred_df = pd.DataFrame()
            curr_pred_df['pred'] = pred
            curr_pred_df.index = X_curr.index
            pred_df.append(curr_pred_df)
        
        
        pred_df = pd.concat(pred_df, axis=0)
        pred_df = pred_df.loc[X.index]                
                      
        return pred_df['pred'].values
        











