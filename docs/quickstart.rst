⏳ Quick Start
==============


Use application model
---------------------



There are several pre-defined fitted models at 
:mod:`~ml_investment.applications`. 
It incapsulating data and weights downloading, pipeline creation 
and model fitting. So you can just use it without knowing internal structure.

.. code-block:: python
    
    from ml_investment.applications.fair_marketcap_yahoo import FairMarketcapYahoo

    fair_marketcap_yahoo = FairMarketcapYahoo()
    fair_marketcap_yahoo.predict(['AAPL', 'FB', 'MSFT'])


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
######################

You may download default datasets by 
:mod:`~ml_investment.download_scripts`

.. code-block:: python

    from ml_investment.download_scripts import download_yahoo
    download_yahoo.main()

| >> 1365it [03:32,  6.42it/s]
| >> 1365it [01:49,  12.51it/s]


2. Define and fit pipeline 
###############################

You may specify all steps of pipeline creation.
There are several default data loaders in 
:mod:`~ml_investment.data`.
Features are in 
:mod:`~ml_investment.features`
Targets are in 
:mod:`~ml_investment.targets`
There are some model wrappers in 
:mod:`~ml_investment.targets`



.. code-block:: python

    from ml_investment.utils import load_config    
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

    pipeline.fit(data_loader, ['AAPL', 'TSLA', 'NVDA', 'K'])

>> {'metric_my_super_model': 0.40599471294301914}

**3. Inference your pipeline**

.. code-block:: python

    pipeline.execute(data_loader, ['AAPL', 'FB', 'MSFT'])


+-------------+-------------------------+------------------+
| ticker      | date                    | my_super_model   |
+=============+=========================+==================+
| AAPL        | 2020-12-31              | 8.170051e+11     |
+-------------+-------------------------+------------------+
| FB          | 2020-12-31              | 3.898840e+11     |
+-------------+-------------------------+------------------+
| MSFT        | 2020-12-31              | 3.540126e+11     |
+-------------+-------------------------+------------------+




