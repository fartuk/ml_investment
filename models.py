import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import GroupKFold



class LogExpModel:
    def __init__(self, base_model):
        self.base_model = base_model
        
    def fit(self, X, y):
        self.base_model.fit(X, np.log(y))
   
    def predict(self, X):
        return np.exp(self.base_model.predict(X))


class AnsambleModel:
    def __init__(self, base_models, bagging_fraction=0.8, model_cnt=20):
        self.base_models = base_models
        self.bagging_fraction = bagging_fraction
        self.model_cnt = model_cnt
        self.models = []
        
        
    def fit(self, X, y):
        X.index = range(len(X))
        y.index = range(len(y))
        for _ in tqdm(range(self.model_cnt)):
            idxs = np.random.randint(0, len(X), 
                                     int(len(X) * self.bagging_fraction))
            curr_model = deepcopy(np.random.choice(self.base_models))
            curr_model.fit(X.loc[idxs], y.loc[idxs])
            self.models.append(curr_model)
                
    
    def predict(self, X):
        preds = []
        for k in range(self.model_cnt):
            try:
                model_pred = self.base_models[k].predict_proba(X)[:, 0]
            except:
                model_pred = self.base_models[k].predict(X)
                
            preds.append(model_pred)
        
        return np.mean(preds, axis=0)         
                        


class GroupedOOFModel:
    def __init__(self, base_model, group_column, fold_cnt=5):
        self.fold_cnt = fold_cnt
        self.group_column = group_column
        self.base_models = []
        for k in range(self.fold_cnt):
            self.base_models.append(deepcopy(base_model))        
        self.group_df = None
        self.columns = None
       

    def fit(self, X, y):
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
        
        
    def predict(self, X):
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
            curr = X[predict_groups['fold_id'] == fold_id]
            if len(curr) == 0:
                continue
            try:
                pred = self.base_models[fold_id].predict_proba(curr)[:, 0]
            except:
                pred = self.base_models[fold_id].predict(curr)

            curr_pred_df = pd.DataFrame()
            curr_pred_df['pred'] = pred
            curr_pred_df.index = curr.index
            pred_df.append(curr_pred_df)
        
        pred_df = pd.concat(pred_df, axis=0)
        pred_df = pred_df.loc[X.index]
        
        return pred_df['pred'].values


class TimeSeriesOOFModel:
    def __init__(self, base_model, fold_cnt=5):
        self.fold_cnt = fold_cnt
        self.base_models = []
        for k in range(self.fold_cnt):
            self.base_models.append(deepcopy(base_model))
            
        
        self.time_bounds = None        
        None
        
   
    def fit(X, y, time):
        np.linspace(min(time), max(time))
        None
        
    
    def predict(X, time):
        None


class OneVsAllModel:
    None



































