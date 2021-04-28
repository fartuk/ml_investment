import os
import json
from urllib.request import urlretrieve

_base_dir = os.path.expanduser('~')
_ml_investments_dir = os.path.join(_base_dir, '.ml_investments')
_config_path = os.path.join(_ml_investments_dir, 'config.json')
_secrets_path = os.path.join(_ml_investments_dir, 'secrets.json')
_tickers_path = os.path.join(_ml_investments_dir, 'tickers.json')

if not os.path.exists(_ml_investments_dir):
    try:
        os.makedirs(_ml_investments_dir)
    except OSError:
        pass
    
if not os.path.exists(_config_path):
    _config = {
        "sf1_data_path":os.path.join(_ml_investments_dir, 'data', 'sf1'),
        "yahoo_data_path":os.path.join(_ml_investments_dir, 'data', 'yahoo'),
        "commodities_data_path":os.path.join(_ml_investments_dir, 'data', 'commodities'),
        "daily_bars_data_path":os.path.join(_ml_investments_dir, 'data', 'daily_bars'),
        "models_path":os.path.join(_ml_investments_dir, 'models'),
        "out_path":os.path.join(_ml_investments_dir, 'data', 'out'),
        "quandl_api_url":"https://www.quandl.com/api/v3"
    }

    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        pass

if not os.path.exists(_secrets_path):
    _secrets = {
        "quandl_api_key":None,
        "tinkoff_token":None,
        "tinkoff_broker_account_id":None
    }

    try:
        with open(_secrets_path, 'w') as f:
            f.write(json.dumps(_secrets, indent=4))
    except IOError:
        pass



if not os.path.exists(_tickers_path):
    try:
        url = 'https://github.com/fartuk/ml_investment/releases/download/weights/tickers.json'
        urlretrieve(url, _tickers_path)
    except IOError:
        pass


   
