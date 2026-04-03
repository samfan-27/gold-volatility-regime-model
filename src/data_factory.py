import pandas as pd
import numpy as np

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataFactory')

class DataFactory():
    def __init__(self, raw_dict='../data/raw', processed_dict='../data/processed'):
        self.raw_dict = Path(raw_dict)
        self.processed_dict = Path(processed_dict)
        
    def _clean_price_series(self, file_name, date_col, price_col, rename_to):
        file_path = self.raw_dict / file_name
        logging.info(f'Loading {file_name}...')
        
        df = pd.read_csv(file_path, thousands=',')
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        df = df[[price_col]].copy()
        df[price_col] = df[price_col].astype(float)

        df = df.rename(columns={price_col: rename_to})
        df = df.sort_index()
        return df
    
    def build_master_dataset(self):
        logger.info('Starting master dataset compilation...')
        
        gold_df = self._clean_price_series(
            'Gold Futures Historical Data.csv',
            date_col='Date',
            price_col='Price',
            rename_to='Gold_Close'
        )
        
        gvz_df = self._clean_price_series(
            'CBOE Gold Volatitity Historical Data.csv',
            date_col='Date',
            price_col='Price',
            rename_to='GVZ_Close'
        )
        
        rates_df = self._clean_price_series(
            'TB3MS.csv',
            date_col='observation_date',
            price_col='TB3MS',
            rename_to='RiskFreeRate'
        )
        
        rates_df['RiskFreeRate'] = rates_df['RiskFreeRate'] / 100.0
        
        logger.info('Aligning indices and handling frequency mismatches...')
        master_df = gold_df.join(gvz_df, how='outer')
        master_df = master_df.join(rates_df, how='outer')
        
        master_df['RiskFreeRate'] = master_df['RiskFreeRate'].ffill()
        master_df = master_df.dropna(subset=['Gold_Close', 'GVZ_Close'])
        
        master_df['Log_Return'] = np.log(master_df['Gold_Close'] / master_df['Gold_Close'].shift(1))
        master_df = master_df.dropna()
        
        output_file = self.processed_dict / 'master_dataset.parquet'
        master_df.to_parquet(output_file)
        logger.info(f'Master dataset saved to {output_file} with shape {master_df.shape}')
        
        return master_df
    
if __name__ == '__main__':
    factory = DataFactory(raw_dict='data/raw', processed_dict='data/processed')
    df = factory.build_master_dataset()
    
    print('\nSample Data:')
    print(df.head())
        