import argparse
from tqdm import tqdm
from ml_investment.data import SF1Data
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

def main():
    config = load_config()
    downloader = QuandlDownloader(sleep_time=0.8)
    for code in tqdm(quandl_commodities_codes):
        downloader.single_download('datasets/{}'.format(code),
                                   '{}/{}.json'.format(config['commodities_data_path'],
                                                  code.replace('/', '_')))
 

if __name__ == '__main__':
    main()
   
