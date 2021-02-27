# clever_investment
Investment tools


## Pipelines
### Marketcap model
Model trying to estimate current fair company marketcap. 
Trained on real caps. Since some companies are overvalued and some are undervalued, the model makes an average "fair" prediction.
Quarterly-based series and general company features are used for the model.
![plot](./images/marketcap_prediction.png?raw=true "marketcap_prediction")

```python3
    fc1 = QuarterlyFeatures(
        columns=pipeline_config['quarter_columns'],
        quarter_counts=pipeline_config['quarter_counts'],
        max_back_quarter=pipeline_config['max_back_quarter'])

    fc2 = BaseCompanyFeatures(
        cat_columns=pipeline_config['cat_columns'])

    feature = FeatureMerger(fc1, fc2, on='ticker')
    target = QuarterlyTarget(col='marketcap', quarter_shift=0)

    base_models = [LogExpModel(lgbm.sklearn.LGBMRegressor()),
                   LogExpModel(ctb.CatBoostRegressor(verbose=False))]
                   
    ansamble = AnsambleModel(base_models=base_models, 
                             bagging_fraction=0.7, model_cnt=20)

    model = GroupedOOFModel(ansamble, group_column='ticker', fold_cnt=5)

    pipeline = BasePipeline(feature=feature, 
                            target=target, 
                            model=model, 
                            metric=median_absolute_relative_error)
                            
    pipeline.fit(config, ticker_list)
    pipeline.export_core('models_data/marketcap')
```

### Quarter marketcap difference model
Get last and current quarter results, calculate features and predict marketcap difference.

Model trying to estimate current fair company marketcap. 
Trained on real caps. Since some companies are overvalued and some are undervalued, the model makes an average "fair" prediction.
Quarterly-based series and general company features are used for the model.


## Features


## Data
Expected data from https://www.quandl.com/databases/SF1

    cf1
    ├── core_fundamental        # data from route 
    │   ├── AAPL.json
    │   ├── FB.json
    │   └── ...
    ├── daily                   # data from route 
    │   ├── AAPL.json
    │   ├── FB.json
    │   └── ...
    └── 


