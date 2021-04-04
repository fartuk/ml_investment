import pytest
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ml_investment.models import LogExpModel, EnsembleModel, GroupedOOFModel, \
                   TimeSeriesOOFModel
from ml_investment.utils import load_json
config = load_json('config.json')



def gen_data(cnt):
    np.random.seed(0)
    y = np.random.normal(0, 1, cnt)
    x1 = y + np.random.normal(0, 0.1, cnt)
    x1 = np.expand_dims(x1, axis=1)
    x2 = y + np.random.normal(0, 0.2, cnt)
    x2 = np.expand_dims(x2, axis=1)
    X = np.concatenate([x1, x2], axis=1)
    y = np.exp(y)
    
    return pd.DataFrame(X), pd.DataFrame(y).rename({0: 'y'}, axis=1)
    
    
class TestLogExpModel:
    def test_fit_predict(self):
        X, y = gen_data(1000)
        model = LogExpModel(lgbm.sklearn.LGBMRegressor())
        model.fit(X[:600], y['y'][:600])
        pred = model.predict(X[600:])
        base_pred = model.base_model.predict(X[600:])
        np.testing.assert_array_almost_equal(base_pred, np.log(pred))    
        assert type(model.base_model) == lgbm.sklearn.LGBMRegressor

        base_model = LinearRegression()
        base_model.fit(X[:600], y['y'][:600])
        pred = base_model.predict(X[600:])
        base_score = mean_squared_error(y['y'][600:], pred)

        model = LogExpModel(LinearRegression())
        model.fit(X[:600], y['y'][:600])
        pred = model.predict(X[600:])
        logexp_score = mean_squared_error(y[600:], pred)

        assert logexp_score < base_score


class ConstModel:
    def __init__(self, const):
        self.const = const
        
    def fit(self, X, y):
        None
        
    def predict(self, X):
        return np.array([self.const] * len(X))


class TestEnsembleModel:
    def test_fit_predict(self):
        X, y = gen_data(1000)
        base_model = LinearRegression()
        base_model.fit(X[:600], y['y'][:600])
        pred = base_model.predict(X[600:])
        base_score = mean_squared_error(y['y'][600:], pred)

        model = EnsembleModel([LinearRegression(), 
                               lgbm.sklearn.LGBMRegressor()], 
                               bagging_fraction=0.8,
                               model_cnt=20)
                               
        model.fit(X[:600], y['y'][:600])
        pred = model.predict(X[600:])
        ans_score = mean_squared_error(y[600:], pred)
        assert len(model.models) == 20
        assert len(pred) == len(X[600:])
        assert ans_score < base_score


        model = EnsembleModel([ConstModel(-1), 
                               ConstModel(1)], 
                               bagging_fraction=0.8,
                               model_cnt=5000)
        model.fit(X[:600], y['y'][:600])
        pred = model.predict(X[600:])
        assert len(set(pred)) == 1
        assert np.abs(pred[0]) < 0.1

        model = EnsembleModel([ConstModel(1), 
                               ConstModel(1)], 
                               bagging_fraction=0.8,
                               model_cnt=5000)
        model.fit(X[:600], y['y'][:600])
        pred = model.predict(X[600:])
        assert len(set(pred)) == 1
        assert pred[0] == 1

        model = EnsembleModel([lgbm.sklearn.LGBMClassifier(max_depth=3), 
                               lgbm.sklearn.LGBMClassifier()], 
                               bagging_fraction=0.8,
                               model_cnt=20)
        model.fit(X[:600], np.log(y['y'])[:600] > 0)
        pred = model.predict(X[600:])
        assert (pred >= 0).min()
        assert (pred <= 1).min()


def gen_grouped_data(cnt):
    X = pd.DataFrame()
    y = pd.DataFrame()
    np.random.seed(0)
    X['ticker'] = np.random.randint(0, 20, cnt)
    X['date'] = np.random.uniform(0, 1, cnt)
    X['col'] = np.random.normal(0, 1, cnt)
    y['y'] = X['ticker'] 
    return X, y
    

class GroupTestModel:
    def __init__(self):
        self.known_targets = None
        
    def fit(self, X, y):
        self.known_targets = list(set(y))
    
    def predict(self, X):
        return np.random.choice(self.known_targets, len(X))
    

class TestGroupedOOFModel:
    def test_fit_predict(self):
        X_, y = gen_grouped_data(1000)    
        model = GroupedOOFModel(GroupTestModel(),
                                group_column='ticker', fold_cnt=5)

        for X in [X_, X_.set_index(['ticker', 'date'])]:
            model.fit(X, y['y'])
            pred = model.predict(X)
            assert len(X) == len(pred)
            assert len(model.group_df) == 20
            assert len(model.group_df['fold_id'].unique()) == 5
            info = X.copy()
            info['y'] = y['y']
            info['pred'] = pred
            info = info.reset_index()
            info = pd.merge(info.rename({'ticker':'group'}, axis=1), 
                            model.group_df, on='group', how='left')
            assert (info['y'] != info['pred']).min()
            folds_df = pd.merge(info.groupby('fold_id')['y'].unique(),
                                info.groupby('fold_id')['pred'].unique(),
                                on='fold_id', how='left')

            assert folds_df.apply(lambda x: len(set(x['y']) \
                                 .intersection(set(x['pred']))) == 0,
                                 axis=1).min()
            
            if 'ticker' in X.columns:
                X['ticker'] = 100500
                pred = model.predict(X)
                assert len(set(pred).intersection(
                                        set(folds_df.loc[0]['y']))) == 0


        X_, y = gen_grouped_data(1000)    
        model = GroupedOOFModel(lgbm.sklearn.LGBMClassifier(),
                                group_column='ticker', fold_cnt=5)
        model.fit(X, y['y'] > 5)
        pred = model.predict(X)
        assert (pred >= 0).min()
        assert (pred <= 1).min()        



def gen_ts_data(cnt):
    X = pd.DataFrame()
    y = pd.DataFrame()
    np.random.seed(0)
    X['ticker'] = np.random.randint(0, 20, cnt)
    X['date'] = [np.datetime64('2020-02-15') + np.timedelta64(k, 'D')
                 for k in range(cnt)]
    X['col'] = np.random.normal(0, 1, cnt)
    y['y'] = range(cnt) 
    return X, y
    

class TsTestModel:
    def __init__(self):
        self.max_target = None
        
    def fit(self, X, y):
        self.max_target = max(y)
    
    def predict(self, X):
        return np.random.randint(0, self.max_target, len(X))
        

class TestTimeSeriesOOFModel:
    def test_fit_predict(self):
        X, y = gen_ts_data(10000)
        model = TimeSeriesOOFModel(TsTestModel(),
                                   time_column='date', fold_cnt=20)
        model.fit(X, y['y'])
        pred = model.predict(X)
        info = X.copy()
        info['pred'] = pred
        assert (info['pred'][:len(X) // 20].isnull()).min()
        assert len(model.base_models) == 20
        assert len(model.time_bounds) == 20 + 1
        means = []
        maxs = []
        for fold_id in range(1, 20):
            fold_info = info[fold_id * (len(X) // 20):(fold_id + 1) * (len(X) // 20)]
            means.append(fold_info['pred'].mean())
            maxs.append(fold_info['pred'].max())

        means = np.array(means)
        maxs = np.array(maxs)

        assert (means[1:] - means[:-1]).min() > 0
        assert (maxs[1:] - maxs[:-1]).min() > 0
        
        X['date'] = np.datetime64('2050-01-01')
        pred = model.predict(X)
        assert np.abs(pred.mean() - 5000) < 20
        assert np.abs(pred.max() - 10000) < 20
        
        X, y = gen_ts_data(10000)
        X = X.set_index('date')
        model = TimeSeriesOOFModel(lgbm.sklearn.LGBMClassifier(),
                                   time_column='date', fold_cnt=20)
        model.fit(X, np.random.randint(0, 2, len(X)))
        pred = model.predict(X)
        assert (pred[len(X) // 20:] >= 0).min()
        assert (pred[len(X) // 20:] <= 1).min()        


























































