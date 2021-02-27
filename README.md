
# clever_investment
Investment tools


## Pipelines
All models represented as pipelines composed of feature and target calculation, model training and validation.
Simple example of pipeline creation using QuarterlyFeatures and BaseCompanyFeatures:

```python3
    fc1 = QuarterlyFeatures(
        columns=["revenue", "netinc", "debt"],
        quarter_counts=[2, 4, 10],
        max_back_quarter=10)

    fc2 = BaseCompanyFeatures(
        cat_columns=["sector", "sicindustry"])

    feature = FeatureMerger(fc1, fc2, on='ticker')
    target = QuarterlyTarget(col='marketcap', quarter_shift=0)

    model = GroupedOOFModel(LogExpModel(lgbm.sklearn.LGBMRegressor()),
                            group_column='ticker', fold_cnt=5)

    pipeline = BasePipeline(feature=feature, 
                            target=target, 
                            model=model, 
                            metric=median_absolute_relative_error)
                            
    pipeline.fit(config, ticker_list)
    pipeline.export_core('models_data/marketcap')
```

### Marketcap model
Model trying to estimate current fair company marketcap. 
Trained on real caps. Since some companies are overvalued and some are undervalued, the model makes an average "fair" prediction.

To fit default pre-defined marketcap prediction pipeline run 
```properties
python3 train/marketcap.py --config_path config.json
```

![plot](./images/marketcap_prediction.png?raw=true "marketcap_prediction")




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

