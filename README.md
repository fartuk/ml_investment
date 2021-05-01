---
generator: 'Docutils 0.16: http://docutils.sourceforge.net/'
title: 'Ml\_investment'
---

<div id="ml-investment" class="document">

Machine learning tools for investment tasks. The purpose of these tools
is to obtain deeper analytics about companies traded on the stock
exchange.

<div id="table-of-content" class="contents topic">

Table of content

-   [📔 Documentation](#documentation){#id1 .reference .internal}
-   [🛠 Installation](#installation){#id2 .reference .internal}
-   [⏳ Quick Start](#quick-start){#id3 .reference .internal}
    -   [Use application model](#use-application-model){#id4 .reference
        .internal}
    -   [Create your own pipeline](#create-your-own-pipeline){#id5
        .reference .internal}
-   [📦 Applications](#applications){#id6 .reference .internal}
-   [⭐ Contributing](#contributing){#id7 .reference .internal}
    -   [Run tests](#run-tests){#id8 .reference .internal}
    -   [Run tests in Docker](#run-tests-in-docker){#id9 .reference
        .internal}

</div>

<div id="documentation" class="section">

📔 Documentation
===============

Visit [Read the
Docs](https://ml-investment.readthedocs.io/en/latest/index.html){.reference
.external} to know more about Ml\_investmrnt library.

</div>

<div id="installation" class="section">

🛠 Installation
==============

**PyPI version**

``` {.code .bash .literal-block}
$ pip install ml-investment
```

**Latest version from source**

``` {.code .bash .literal-block}
$ pip install git+https://github.com/fartuk/ml_investment
```

or

``` {.code .bash .literal-block}
$ git clone https://github.com/fartuk/ml_investment
$ cd ml_investment
$ pip install .
```

</div>

<div id="quick-start" class="section">

⏳ Quick Start
=============

<div id="use-application-model" class="section">

Use application model
---------------------

There are several pre-defined fitted models at
`ml_investment.applications`{.docutils .literal}. It incapsulating data
and weights downloading, pipeline creation and model fitting. So you can
just use it without knowing internal structure.

``` {.code .python .literal-block}
from ml_investment.applications.fair_marketcap_yahoo import FairMarketcapYahoo

fair_marketcap_yahoo = FairMarketcapYahoo()
fair_marketcap_yahoo.predict(['AAPL', 'FB', 'MSFT'])
```

  ticker                   date                                          fair\_marketcap\_yahoo
  ------------------------ --------------------------------------------- --------------------------------------------
  AAPL                     2020-12-31                                    5.173328e+11
  FB                       2020-12-31                                    8.442045e+11
  MSFT                     2020-12-31                                    4.501329e+11

</div>

<div id="create-your-own-pipeline" class="section">

Create your own pipeline
------------------------

**1. Download data**

You may download default datasets by
`ml_investment.download_scripts`{.docutils .literal}

``` {.code .python .literal-block}
from ml_investment.download_scripts import download_yahoo
download_yahoo.main()
```

<div class="line-block">

<div class="line">

&gt;&gt; 1365it \[03:32, 6.42it/s\]

</div>

<div class="line">

&gt;&gt; 1365it \[01:49, 12.51it/s\]

</div>

</div>

**2. Define and fit pipeline**

You may specify all steps of pipeline creation. Base pipeline consist of
the folowing steps:

-   Define features. Features is a number of values and characteristics
    that will be calculated for model trainig. Default feature
    calculators are located at `ml_investment.features`{.docutils
    .literal}
-   Define targets. Target is a final goal of the pipeline, it should
    represent some desired useful property. Default target calculators
    are located at `ml_investment.targets`{.docutils .literal}
-   Choose model. Model is machine learning algorithm, core of the
    pipeline. It also may incapsulate validateion and other stuff. You
    may use wrappers from `ml_investment.models`{.docutils .literal}
-   Choose dataset. It should have all needed for features and targets
    data loading methods. There some pre-defined datasets at
    `ml_investment.data`{.docutils .literal}

``` {.code .python .literal-block}
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

Since `ml_investment.models.GroupedOOFModel`{.docutils .literal} was
used, there are no data leakage and you may use pipeline on the same
company tickers.

``` {.code .python .literal-block}
pipeline.execute(data_loader, ['AAPL', 'FB', 'MSFT'])
```

  ticker             date                                my\_super\_model
  ------------------ ----------------------------------- -------------------------
  AAPL               2020-12-31                          8.170051e+11
  FB                 2020-12-31                          3.898840e+11
  MSFT               2020-12-31                          3.540126e+11

</div>

</div>

<div id="applications" class="section">

📦 Applications
==============

Collection of pre-trained models

-   FairMarketcapYahoo
    [docs](https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.fair_marketcap_yahoo){.reference
    .external}
-   FairMarketcapSF1
-   FairMarketcapDiffYahoo
-   FairMarketcapDiffSF1
-   MarketcapDownStdYahoo
-   MarketcapDownStdSF1

</div>

<div id="contributing" class="section">

⭐ Contributing
==============

<div id="run-tests" class="section">

Run tests
---------

``` {.code .bash .literal-block}
$ cd /path/to/ml_investmant && pytest
```

</div>

<div id="run-tests-in-docker" class="section">

Run tests in Docker
-------------------

``` {.code .bash .literal-block}
$ docker build . -t tests
$ docker run tests
```

</div>

</div>

</div>
