Ml\_investment
==============

Machine learning tools for investment tasks. The purpose of these tools
is to obtain deeper analytics about companies traded on the stock
exchange.

📔 Documentation
---------------

Visit [Read the
Docs](https://ml-investment.readthedocs.io/en/latest/index.html) to know
more about Ml\_investmrnt library.

🛠 Installation
--------------

**PyPI version**

``` {.sourceCode .bash}
$ pip install ml-investment
```

**Latest version from source**

``` {.sourceCode .bash}
$ pip install git+https://github.com/fartuk/ml_investment
```

or

``` {.sourceCode .bash}
$ git clone https://github.com/fartuk/ml_investment
$ cd ml_investment
$ pip install .
```

⏳ Quick Start
-------------

### Use application model

There are several pre-defined fitted models at
`ml_investment.applications`. It incapsulating data and weights
downloading, pipeline creation and model fitting. So you can just use it
without knowing internal structure.

``` {.sourceCode .python}
from ml_investment.applications.fair_marketcap_yahoo import FairMarketcapYahoo

fair_marketcap_yahoo = FairMarketcapYahoo()
fair_marketcap_yahoo.predict(['AAPL', 'FB', 'MSFT'])
```

  ticker                   date                                         fair\_marketcap\_yahoo
  ------------------------ -------------------------------------------- --------------------------------------------
  AAPL                     2020-12-31                                   5.173328e+11
  FB                       2020-12-31                                   8.442045e+11
  MSFT                     2020-12-31                                   4.501329e+11

### Create your own pipeline

**1. Download data**

You may download default datasets by `ml_investment.download_scripts`

``` {.sourceCode .python}
from ml_investment.download_scripts import download_yahoo
download_yahoo.main()
```

| &gt;&gt; 1365it \[03:32, 6.42it/s\]
| &gt;&gt; 1365it \[01:49, 12.51it/s\]

**2. Define and fit pipeline**

You may specify all steps of pipeline creation. Base pipeline consist of
the folowing steps:

-   Define features. Features is a number of values and characteristics
    that will be calculated for model trainig. Default feature
    calculators are located at `ml_investment.features`
-   Define targets. Target is a final goal of the pipeline, it should
    represent some desired useful property. Default target calculators
    are located at `ml_investment.targets`
-   Choose model. Model is machine learning algorithm, core of the
    pipeline. It also may incapsulate validateion and other stuff. You
    may use wrappers from `ml_investment.models`
-   Choose dataset. It should have all needed for features and targets
    data loading methods. There some pre-defined datasets at
    `ml_investment.data`

``` {.sourceCode .python}
from ml_investment.utils import load_config, load_tickers
from ml_investment.data import YahooData
from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures,\
                                   FeatureMerger
from ml_investment.target import BaseInfoTarget
from ml_investment.pipeline import BasePipeline

config = load_config()
data_loader = YahooData(config['yahoo_data_path'])

fc1 = QuarterlyFeatures(columns=['quarterlyNetIncome',
                                 'quarterlyFreeCashFlow',
                                 'quarterlyTotalAssets',
                                 'quarterlyNetDebt'],
                        quarter_counts=[2, 4, 10],
                        max_back_quarter=1)

fc2 = BaseCompanyFeatures(cat_columns=['sector'])

feature = FeatureMerger(fc1, fc2, on='ticker')

target = BaseInfoTarget(col='enterpriseValue')

base_model = LogExpModel(lgbm.sklearn.LGBMRegressor())
model = GroupedOOFModel(base_model=base_model,
                        group_column='ticker',
                        fold_cnt=4)

pipeline = BasePipeline(feature=feature,
                        target=target,
                        model=model,
                        metric=median_absolute_relative_error,
                        out_name='my_super_model')

tickers = load_tickers()['base_us_stocks']
pipeline.fit(data_loader, tickers)
```

&gt;&gt; {'metric\_my\_super\_model': 0.40599471294301914}

**3. Inference your pipeline**

Since `ml_investment.models.GroupedOOFModel` was used, there are no data
leakage and you may use pipeline on the same company tickers.

``` {.sourceCode .python}
pipeline.execute(data_loader, ['AAPL', 'FB', 'MSFT'])
```

  ticker             date                              my\_super\_model
  ------------------ --------------------------------- ------------------------
  AAPL               2020-12-31                        8.170051e+11
  FB                 2020-12-31                        3.898840e+11
  MSFT               2020-12-31                        3.540126e+11

📦 Applications
--------------

Collection of pre-trained models

-   FairMarketcapYahoo
    [docs](https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.fair_marketcap_yahoo)
-   FairMarketcapSF1
-   FairMarketcapDiffYahoo
-   FairMarketcapDiffSF1
-   MarketcapDownStdYahoo
-   MarketcapDownStdSF1

⭐ Contributing
--------------

### Run tests

``` {.sourceCode .bash}
$ cd /path/to/ml_investmant && pytest
```

### Run tests in Docker

``` {.sourceCode .bash}
$ docker build . -t tests
$ docker run tests
```
