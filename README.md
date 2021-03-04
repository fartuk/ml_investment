
# clever_investment
Investment tools


## Pipelines
All investment tools represented as pipelines composed of feature and target calculation, model training and validation.
Simple example of pipeline creation:

```python3
    data_loader = SF1Data(config['sf1_data_path'])
    
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
                            
    pipeline.fit(data_loader, ['AAPL', 'TSLA', 'NVDA', 'K'])
    pipeline.export_core('models_data/marketcap')
```

### Marketcap evaluation
Model is used to estimate **current** fair company marketcap. 
Pipeline consist of calculating quarterly-based statistics of fundamental company indicators(revenue, netinc etc) and training to predict real market capitalizations. Since some companies are overvalued and some are undervalued, the model makes an average "fair" prediction.

To fit default pre-defined marketcap prediction pipeline run [train/marketcap.py](train/marketcap.py):
```properties
python3 train/marketcap.py --config_path config.json
```

![plot](./images/marketcap_prediction.png?raw=true "marketcap_prediction")
Lower predicted marketcap may indicates that company is overvalued according its fundamentdal base.



### Quarter marketcap difference
Model is used to evaluate quarter-to-quarter(q2q) company fundamental progress.
Pipeline consist of calculating q2q results progress(e.g. 30% revenue increase, decrease in debt by 15% etc) and prediction real q2q marketcap difference. So model prediction may be interpreted as "fair" marketcap change according this fundamental change.

To fit default pre-defined marketcap prediction pipeline run [train/marketcap_diff.py](train/marketcap_diff.py):
```properties
python3 train/marketcap_diff.py --config_path config.json
```

![plot](./images/marketcap_diff_prediction.png?raw=true "marketcap_prediction")
Similarly, a higher predicted capitalization may indicate that the company has fundamentally grown more than its value.



## Data
Most of feature calculators expect data_loader to have folowing structure:
```python3
class DataLoader:
    def load_base_data(self) -> pd.DataFrame:
        None
    def load_quartely_data(self, tickers: List[str]) -> pd.DataFrame:
        None
    def load_daily_data(self, tickers: List[str]) -> pd.DataFrame:
        None
```
There are pre-defined [SF1Data](data.py#L1) class implements this structure.
It is based on the data from https://www.quandl.com/databases/SF1

    sf1
    ├── core_fundamental 
    │   ├── AAPL.json
    │   ├── FB.json
    │   └── ...
    ├── daily
    │   ├── AAPL.json
    │   ├── FB.json
    │   └── ...
    └── tickers.csv

