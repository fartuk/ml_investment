import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GroupKFold


class GroupedOOFModel:
    def __init__(self, base_model, fold_cnt=5):
        self.fold_cnt = fold_cnt
        self.base_models = []
        for k in range(self.fold_cnt):
            self.base_models.append(deepcopy(base_model))        
        self.group_df = None
        

    def fit(self, X, y, groups):
        df_arr = []
        kfold = GroupKFold(self.fold_cnt)
        for k, (itr, ite) in enumerate(kfold.split(X, y, groups)):
            self.base_models[k].fit(X.loc[itr], y.loc[itr])

            curr_group_df = pd.DataFrame()
            curr_group_df['group'] = np.unique(groups[ite])
            curr_group_df['fold_id'] = k
            df_arr.append(curr_group_df)

        self.group_df = pd.concat(df_arr, axis=0)
        
        
    def predict(self, X, groups):
        predict_groups = pd.DataFrame()
        predict_groups['group'] = groups
        predict_groups = pd.merge(predict_groups, self.group_df, on='group', how='left')
        # If group was not in train data -> put to 0th fold
        predict_groups = predict_groups.fillna(0)
        pred_df = []
        for fold_id in range(self.fold_cnt):
            curr = X[predict_groups['fold_id'] == fold_id]
            if len(curr) == 0:
                continue
            try:
                pred = self.base_models[fold_id].predict_proba(curr)
            except:
                pred = self.base_models[fold_id].predict(curr)

            curr_pred_df = pd.DataFrame()
            curr_pred_df['idx'] = curr.index
            curr_pred_df['pred'] = pred
            pred_df.append(curr_pred_df)
        
        pred_df = pd.concat(pred_df, axis=0)
        pred_df = pred_df.sort_values('idx')
        
        return pred_df['pred'].values



class OneVsAllModel:
    None










