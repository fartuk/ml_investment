import pytest
import os
import pandas as pd
import numpy as np

from ml_investment.utils import load_config, load_secrets
from ml_investment.applications.fair_marketcap_sf1 import FairMarketcapSF1
from ml_investment.applications.fair_marketcap_sf1_v2 import FairMarketcapSF1V2
from ml_investment.applications.fair_marketcap_diff_sf1 import FairMarketcapDiffSF1
from ml_investment.applications.fair_marketcap_diff_sf1_v2 import FairMarketcapDiffSF1V2
from ml_investment.applications.marketcap_down_std_sf1 import MarketcapDownStdSF1



config = load_config()
secrets = load_secrets()



 

tickers = ['AAPL', 'TSLA', 'K', 'MAC', 'NVDA']


pipelines = [FairMarketcapSF1,
             FairMarketcapSF1V2,
             FairMarketcapDiffSF1,
             FairMarketcapSF1V2,
             MarketcapDownStdSF1]

class TestFitExecuteSimple:
    data_sources = []
    if os.path.exists(config['sf1_data_path']):
        data_sources.append('sf1')
    if secrets['mongodb_adminusername'] is not None:
        data_sources.append('mongo')


    @pytest.mark.parametrize('data_source', data_sources)
    @pytest.mark.parametrize('pipeline_class', pipelines)
    def test_fit_execute_simple(self, data_source, pipeline_class):
        pipeline = pipeline_class(data_source=data_source,
                                  pretrained=True,
                                  verbose=True)
        #pipeline.fit(tickers)
        result_df = pipeline.execute(tickers)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




