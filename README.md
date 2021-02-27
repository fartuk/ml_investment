# clever_investment
Investment tools


## Pipelines
### Marketcap model
Model trying to estimate current fair company marketcap. 
Trained on real caps. Since some companies are overvalued and some are undervalued, the model makes an average "fair" prediction.
Quarterly-based series and general company features are used for the model.
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


