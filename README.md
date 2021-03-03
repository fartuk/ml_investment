
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

### Marketcap evaluation
Model is used to estimate **current** fair company marketcap. 
Pipeline consist of calculating quarterly-based statistics of fundamental company indicators(revenue, netinc etc) and training to predict real market capitalizations. Since some companies are overvalued and some are undervalued, the model makes an average "fair" prediction.

To fit default pre-defined marketcap prediction pipeline run:
```properties
python3 train/marketcap.py --config_path config.json
```

![plot](./images/marketcap_prediction.png?raw=true "marketcap_prediction")
Lower predicted marketcap may indicates that company is overvalued according its fundamentdal base.



### Quarter marketcap difference
Model is used to evaluate quarter-to-quarter(q2q) company fundamental progress.
Pipeline consist of calculating q2q results progress(e.g. 30% revenue increase, decrease in debt by 15% etc) and prediction real q2q marketcap difference. So model prediction may be interpreted as "fair" marketcap change according this fundamental change.

To fit default pre-defined marketcap prediction pipeline run:
```properties
python3 train/marketcap_diff.py --config_path config.json
```

![plot](./images/marketcap_diff_prediction.png?raw=true "marketcap_prediction")
Similarly, a higher predicted capitalization may indicate that the company has fundamentally grown more than its value.



## Features


## Data
Expected data from https://www.quandl.com/databases/SF1

    sf1
    ├── core_fundamental        # data from route 
    │   ├── AAPL.json
    │   ├── FB.json
    │   └── ...
    ├── daily                   # data from route 
    │   ├── AAPL.json
    │   ├── FB.json
    │   └── ...
    └── 

