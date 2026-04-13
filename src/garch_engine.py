import pandas as pd
import numpy as np
from arch import arch_model
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning, module='arch')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GarchEngine')

class GarchVolatilityModel:
    def __init__(self, df: pd.DataFrame, lookback_window: int = 252, regime_window: int = 60, high_vol_percentile: float = 0.80):
        self.df = df.copy()
        self.lookback_window = lookback_window
        self.regime_window = regime_window
        self.high_vol_percentile = high_vol_percentile

    def _fit_and_forecast(self, returns: pd.Series) -> float:
        scaled_returns = returns * 100.0

        am = arch_model(
            scaled_returns, 
            vol='Garch', 
            p=1, 
            q=1, 
            mean='Constant', 
            dist='Normal', 
            rescale=False
        )

        res = am.fit(update_freq=0, disp='off')
        forecasts = res.forecast(horizon=1, reindex=False)
        var_forecast_scaled = forecasts.variance.iloc[-1, 0]
        ann_vol_decimal = np.sqrt(var_forecast_scaled * 252) / 100.0

        return ann_vol_decimal

    def generate_rolling_forecasts(self):
        logger.info(f'Starting rolling GARCH(1,1) forecasts. Lookback: {self.lookback_window} days.')

        returns = self.df['Log_Return'].dropna()
        n_obs = len(returns)
        forecasts = np.full(n_obs, np.nan)

        for i in range(self.lookback_window, n_obs):
            train_returns = returns.iloc[i - self.lookback_window : i]
            
            forecasts[i] = self._fit_and_forecast(train_returns)
            
            if i % 250 == 0:
                logger.info(f'Progress: {i}/{n_obs} trading days processed...')

        self.df['GARCH_Vol_Annualized'] = np.nan
        self.df.loc[returns.index, 'GARCH_Vol_Annualized'] = forecasts
        logger.info('Rolling forecasts completed.')

    def classify_regimes(self):
        logger.info('Classifying volatility regimes...')
        rolling_rank = self.df['GARCH_Vol_Annualized'].rolling(window=self.regime_window).rank(pct=True)
        self.df['Regime'] = np.where(rolling_rank >= self.high_vol_percentile, 1, 0)

        self.df.loc[self.df['GARCH_Vol_Annualized'].isna(), 'Regime'] = np.nan

        logger.info('Regime classification completed.')

    def run_pipeline(self) -> pd.DataFrame:
        self.generate_rolling_forecasts()
        self.classify_regimes()

        final_df = self.df.dropna(subset=['GARCH_Vol_Annualized']).copy()
        logger.info(f'Pipeline finished. Output shape: {final_df.shape}')

        return final_df

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    processed_data_path = base_dir / 'data' / 'processed' / 'master_dataset.parquet'
    
    df = pd.read_parquet(processed_data_path)
    garch_model = GarchVolatilityModel(
        df=df, 
        lookback_window=252,
        regime_window=60,
        high_vol_percentile=0.80
    )
    regime_df = garch_model.run_pipeline()
        
    print('\nSample Output (End of Dataset):')
    print(regime_df[['Gold_Close', 'Log_Return', 'GARCH_Vol_Annualized', 'Regime']].tail())
        
    output_file = base_dir / 'data' / 'processed' / 'regime_dataset.parquet'
    regime_df.to_parquet(output_file)
    logger.info(f'Regime dataset saved to {output_file}')
