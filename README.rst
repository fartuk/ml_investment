Ml_investment
########################

Machine learning tools for investment tasks.
The purpose of these tools is to obtain deeper analytics
about companies using fundamental, global market and other data.
This repo is core of https://stocks.ml.

.. contents:: Table of content
   :depth: 2
   :backlinks: none



📔 Documentation
=================
Visit
`Read the Docs <https://ml-investment.readthedocs.io/en/latest/index.html>`__
to know more about Ml_investment library.

🛠 Installation
===============


**PyPI version**

.. code-block:: bash

    $ pip install ml-investment


**Latest version from source**

.. code-block:: bash

    $ pip install git+https://github.com/fartuk/ml_investment


**Configuration**

You may use config file `~/.ml_investment/config.json`
to change repo parameters i.e. downloading datasets pathes, models pathes etc.

Private information (i.e. api tokens for private datasets downloading)
should be located at `~/.ml_investment/secrets.json`

⏳ Quick Start
==============


Use application model
---------------------



There are several pre-defined fitted models at
``ml_investment.applications``.
It incapsulating data and weights downloading, pipeline creation
and model fitting. So you can just use it without knowing internal structure.

.. code-block:: python

    from ml_investment.applications.fair_marketcap_yahoo import FairMarketcapYahoo

    fair_marketcap_yahoo = FairMarketcapYahoo()
    fair_marketcap_yahoo.execute(['AAPL', 'FB', 'MSFT'])


+-------------+-------------------------+------------------------+
| ticker      | date                    | fair_marketcap_yahoo   |
+=============+=========================+========================+
| AAPL        | 2020-12-31              | 5.173328e+11           |
+-------------+-------------------------+------------------------+
| FB          | 2020-12-31              | 8.442045e+11           |
+-------------+-------------------------+------------------------+
| MSFT        | 2020-12-31              | 4.501329e+11           |
+-------------+-------------------------+------------------------+



Create your own pipeline
-------------------------


**1. Download data**

You may download default datasets by
``ml_investment.download_scripts``

.. code-block:: python

    from ml_investment.download_scripts import download_yahoo
    from ml_investment.utils import load_config

    # Config located at ~/.ml_investment/config.json
    config = load_config()

    download_yahoo.main(config['yahoo_data_path'])

>>> 1365it [03:32,  6.42it/s]
>>> 1365it [01:49,  12.51it/s]

**2. Create dict with dataloaders**

You may choose from default
``ml_investment.data_loaders``
or wrote your own. Each dataloader should have ``load(index)`` interface.

.. code-block:: python

    from ml_investment.data_loaders.yahoo import YahooQuarterlyData, YahooBaseData

    data = {}
    data['quarterly'] = YahooQuarterlyData(config['yahoo_data_path'])
    data['base'] = YahooBaseData(config['yahoo_data_path'])


**3. Define and fit pipeline**

You may specify all steps of pipeline creation.
Base pipeline consist of the folowing steps:

- Create data dict(it was done in previous step)
- Define features. Features is a number of values
  and characteristics that will be calculated for model trainig.
  Default feature calculators are located at
  ``ml_investment.features``
- Define targets. Target is a final goal of the pipeline, it should
  represent some desired useful property.
  Default target calculators are located at
  ``ml_investment.targets``
- Choose model. Model is machine learning algorithm, core of the pipeline.
  It also may incapsulate validation and other stuff.
  You may use wrappers from
  ``ml_investment.models``

.. code-block:: python

    import lightgbm as lgbm
    from ml_investment.utils import load_config, load_tickers
    from ml_investment.features import QuarterlyFeatures, BaseCompanyFeatures,\
                                       FeatureMerger
    from ml_investment.targets import BaseInfoTarget
    from ml_investment.models import LogExpModel, GroupedOOFModel
    from ml_investment.pipelines import Pipeline
    from ml_investment.metrics import median_absolute_relative_error

    fc1 = QuarterlyFeatures(data_key='quarterly',
                            columns=['netIncome',
                                     'cash',
                                     'totalAssets',
                                     'ebit'],
                            quarter_counts=[2, 4, 10],
                            max_back_quarter=1)

    fc2 = BaseCompanyFeatures(data_key='base', cat_columns=['sector'])

    feature = FeatureMerger(fc1, fc2, on='ticker')

    target = BaseInfoTarget(data_key='base', col='enterpriseValue')

    base_model = LogExpModel(lgbm.sklearn.LGBMRegressor())
    model = GroupedOOFModel(base_model=base_model,
                            group_column='ticker',
                            fold_cnt=4)

    pipeline = Pipeline(data=data,
                        feature=feature,
                        target=target,
                        model=model,
                        out_name='my_super_model')

    tickers = load_tickers()['base_us_stocks']
    pipeline.fit(tickers, metric=median_absolute_relative_error)

>>> {'metric_my_super_model': 0.40599471294301914}

**4. Inference your pipeline**

Since ``ml_investment.models.GroupedOOFModel`` was used,
there are no data leakage and you may use pipeline on the same company tickers.

.. code-block:: python

    pipeline.execute(['AAPL', 'FB', 'MSFT'])


+-------------+-------------------------+------------------+
| ticker      | date                    | my_super_model   |
+=============+=========================+==================+
| AAPL        | 2020-12-31              | 8.170051e+11     |
+-------------+-------------------------+------------------+
| FB          | 2020-12-31              | 3.898840e+11     |
+-------------+-------------------------+------------------+
| MSFT        | 2020-12-31              | 3.540126e+11     |
+-------------+-------------------------+------------------+

📦 Applications
================

Collection of pre-trained models

- FairMarketcapYahoo
  [`docs <https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.fair_marketcap_yahoo>`__]

- FairMarketcapSF1
  [`docs <https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.fair_marketcap_sf1>`__]
- FairMarketcapDiffYahoo
  [`docs <https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.fair_marketcap_diff_yahoo>`__]
- FairMarketcapDiffSF1
  [`docs <https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.fair_marketcap_diff_sf1>`__]
- MarketcapDownStdYahoo
  [`docs <https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.marketcap_down_std_yahoo>`__]
- MarketcapDownStdSF1
  [`docs <https://ml-investment.readthedocs.io/en/latest/applications.html#module-ml_investment.applications.marketcap_down_std_sf1>`__]


⭐ Contributing
=================

Run tests
----------

.. code-block:: bash

    $ cd /path/to/ml_investment && pytest


Run tests in Docker
--------------------

.. code-block:: bash

    $ docker build . -t tests
    $ docker run tests

