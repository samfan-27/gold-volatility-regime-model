import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Black76Pricer')

class Black76Pricer:
    """
    Prices European options on futures contracts using the Black '76 model
    Utilizes GARCH forecasted volatility to generate theoretical model prices
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _calculate_d1_d2(self, F: pd.Series, K: pd.Series, T: float, sigma: pd.Series) -> tuple:
        sigma = np.maximum(sigma, 1e-8)
        d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price_daily_straddles(self, days_to_expiry: int = 30) -> pd.DataFrame:
        """
        theo ATM Call, Put, and Straddle prices
        """
        logger.info(f'Pricing theoretical ATM straddles (T = {days_to_expiry} calendar days)...')
        
        F = self.df['Gold_Close']
        K = self.df['Gold_Close']
        T = days_to_expiry / 365.0
        r = self.df['RiskFreeRate']
        sigma = self.df['GARCH_Vol_Annualized']

        d1, d2 = self._calculate_d1_d2(F, K, T, sigma)
        discount_factor = np.exp(-r * T)

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        self.df['Theo_Call_Price'] = discount_factor * (F * N_d1 - K * N_d2)
        self.df['Theo_Put_Price'] = discount_factor * (K * N_neg_d2 - F * N_neg_d1)
        
        self.df['Theo_Straddle_Price'] = self.df['Theo_Call_Price'] + self.df['Theo_Put_Price']

        final_df = self.df.dropna(subset=['Theo_Straddle_Price']).copy()
        logger.info('Straddle pricing complete.')
        return final_df

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    regime_data_path = base_dir / 'data' / 'processed' / 'regime_dataset.parquet'
    
    df = pd.read_parquet(regime_data_path)
    pricer = Black76Pricer(df)
    priced_df = pricer.price_daily_straddles(days_to_expiry=30)
        
    print('\nSample Output (Theoretical Pricing):')
    columns_to_show = ['Gold_Close', 'GARCH_Vol_Annualized', 'Theo_Call_Price', 'Theo_Put_Price', 'Theo_Straddle_Price']
    print(priced_df[columns_to_show].tail())
        
    output_file = base_dir / 'data' / 'processed' / 'priced_dataset.parquet'
    priced_df.to_parquet(output_file)
    logger.info(f'Priced dataset saved to {output_file}')
