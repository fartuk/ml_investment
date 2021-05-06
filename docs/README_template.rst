Ml_investment
########################

Machine learning tools for investment tasks. 
The purpose of these tools is to obtain deeper analytics
about companies traded on the stock exchange.


.. contents:: Table of content
   :depth: 2
   :backlinks: none



📔 Documentation
=================
Visit  
`Read the Docs <https://ml-investment.readthedocs.io/en/latest/index.html>`__
to know more about Ml_investment library.


.. include:: install.rst
.. include:: quickstart.rst



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

    $ cd /path/to/ml_investmant && pytest


Run tests in Docker
--------------------

.. code-block:: bash

    $ docker build . -t tests
    $ docker run tests
