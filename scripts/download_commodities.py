import argparse
from tqdm import tqdm
from ml_investment.data import SF1Data
from ml_investment.download import QuandlDownloader
from ml_investment.utils import load_json, save_json

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

def main(config_path, secrets_path):
    config = load_json(config_path)
    secrets = load_json(secrets_path)  

    downloader = QuandlDownloader(config, secrets, sleep_time=0.8)

    for code in tqdm(quandl_commodities_codes):
        downloader.single_download('datasets/{}'.format(code),
                                   '{}/{}.json'.format(config['commodities_data_path'],
                                                  code.replace('/', '_')))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config_path', type=str, default="config.json")
    arg('--secrets_path', type=str, default="secrets.json")
    args = parser.parse_args()
    
    main(args.config_path, args.secrets_path)
   
