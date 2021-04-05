import argparse
from tqdm import tqdm
from ml_investment.data import SF1Data
from ml_investment.download import QuandlDownloader
from ml_investment.utils import load_json, save_json

quandl_commodities_codes = ['LBMA/GOLD',
                            'LBMA/SILVER',
                            'JOHNMATT/PALL',
                            'ODA/PBARL_USD',
                            'TFGRAIN/CORN', 
                            'ODA/PRICENPQ_USD',  
                            'CHRIS/CME_DA1',
                            'ODA/PBEEF_USD',
                            'ODA/PPOULT_USD', 
                            'ODA/PPORK_USD',  
                            'ODA/PWOOLC_USD',
                            'CHRIS/CME_CL1',
                            'ODA/POILBRE_USD',
                            'CHRIS/CME_NG1', 
                            'ODA/PCOFFOTM_USD',
                            'ODA/PCOCO_USD',
                            'ODA/PORANG_USD',
                            'ODA/PBANSOP_USD',
                            'ODA/POLVOIL_USD',
                            'ODA/PLOGSK_USD',
                            'ODA/PCOTTIND_USD'
                           ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str)
    arg('--secrets_path', type=str)
    args = parser.parse_args()
    
    config = load_json(args.config_path)
    secrets = load_json(args.secrets_path)  

    downloader = QuandlDownloader(config, secrets, sleep_time=0.8)

    for code in tqdm(quandl_commodities_codes):
        downloader.single_download('datasets/{}'.format(code),
                                   '{}/{}.json'.format(config['commodities_data_path'],
                                                  code.replace('/', '_')))
    
