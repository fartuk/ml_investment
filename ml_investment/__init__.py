import os
import json
from urllib.request import urlretrieve

_base_dir = os.path.expanduser('~')
_ml_investments_dir = os.path.join(_base_dir, '.ml_investment')
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
    }

    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        pass

if not os.path.exists(_secrets_path):
    _secrets = {
        "quandl_api_key": os.getenv("QUANDL_API_KEY") or None,
        "tinkoff_token": os.getenv("TINKOFF_TOKEN") or None,
        "tinkoff_broker_account_id": os.getenv("TINKOFF_BROKER_ACCOUNT_ID") or None,
        "mongodb_adminusername": os.getenv("MONGODB_ADMINUSERNAME") or None,
        "mongodb_adminpassword": os.getenv("MONGODB_ADMINPASSWORD") or None,
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


_models_dir = os.path.join(_ml_investments_dir, 'models')
if not os.path.exists(_models_dir):
    try:
        os.makedirs(_models_dir)
    except OSError:
        pass
 
