import argparse
from tqdm import tqdm
from ml_investment.download import QuandlDownloader
from ml_investment.utils import load_config, save_json

quandl_commodities_codes = ['LBMA/GOLD',
                            'LBMA/SILVER',
                            'JOHNMATT/PLAT',
                            'JOHNMATT/PALL',
                            'ODA/PALUM_USD',
                            'ODA/PCOPP_USD',
                            'ODA/PNICK_USD',
                            'SHFE/RBV2013',
                            'ODA/PBARL_USD',
                            'TFGRAIN/CORN', 
                            'ODA/PRICENPQ_USD',  
                            'CHRIS/CME_DA1',
                            'ODA/PBEEF_USD',
                            'ODA/PPOULT_USD', 
                            'ODA/PPORK_USD',  
                            'ODA/PWOOLC_USD',
                            'CHRIS/CME_CL1',
                            'ODA/POILWTI_USD',
                            'ODA/POILBRE_USD',
                            'CHRIS/CME_NG1', 
                            'ODA/PCOALAU_USD',
                            'ODA/PCOFFOTM_USD',
                            'ODA/PCOCO_USD',
                            'ODA/PSUGAUSA_USD',
                            'ODA/PORANG_USD',
                            'ODA/PBANSOP_USD',
                            'ODA/POLVOIL_USD',
                            'ODA/PLOGSK_USD',
                            'ODA/PCOTTIND_USD'
                           ]

def main(data_path: str=None):
    '''
    Download commodities price history from 
    https://blog.quandl.com/api-for-commodity-data

    Note:
        To download this dataset you need to register at quandl 
        and paste token to `~/.ml_investment/secrets.json`

    Parameters
    ----------
    data_path:
        path to folder in which downloaded data will be stored.
        OR ``None`` (downloading path will be as ``commodities_data_path`` from 
        `~/.ml_investment/config.json`
    '''
    if data_path is None:
        config = load_config()
        data_path = config['commodities_data_path']

    downloader = QuandlDownloader(sleep_time=0.8)
    for code in tqdm(quandl_commodities_codes):
        downloader.single_download('datasets/{}'.format(code),
                                   '{}/{}.json'.format(data_path,
                                                  code.replace('/', '_')))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    args = parser.parse_args()
    main(args.data_path)
   
