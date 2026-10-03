import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path="data/historical_prices.csv", trading_days=252):
        self.file_path = file_path
        self.trading_days = trading_days
        self.data = None
        self.assets = None
        self.simple_returns = None
        self.log_returns = None
        
    def load_and_validate(self):
        """Loads the price data and validates it."""
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.file_path}")
            
        # Ensure 'Date' column exists
        if 'Date' not in self.data.columns:
            raise ValueError("Dataset must contain a 'Date' column.")
            
        # Convert Date to datetime and check for duplicates
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        if self.data['Date'].duplicated().any():
            raise ValueError("Duplicate dates found in the dataset.")
            
        # Sort by date
        self.data = self.data.sort_values(by='Date').reset_index(drop=True)
        
        # Identify asset columns (all except Date)
        self.assets = [col for col in self.data.columns if col != 'Date']
        
        if not self.assets:
            raise ValueError("No asset columns found in the dataset.")
            
        # Check for missing values
        if self.data.isnull().values.any():
            self.data = self.data.dropna()
            
        # Ensure all prices are numeric and strictly positive
        for asset in self.assets:
            if not pd.api.types.is_numeric_dtype(self.data[asset]):
                raise TypeError(f"Asset column {asset} contains non-numeric values.")
            if (self.data[asset] <= 0).any():
                raise ValueError(f"Asset column {asset} contains zero or negative prices.")
                
        return self.data
        
    def calculate_returns(self):
        """Calculates daily simple and log returns."""
        if self.data is None:
            self.load_and_validate()
            
        # Calculate simple returns: (P_t / P_{t-1}) - 1
        price_data = self.data[self.assets]
        self.simple_returns = price_data.pct_change().dropna()
        
        # Calculate log returns: ln(P_t / P_{t-1})
        self.log_returns = np.log(price_data / price_data.shift(1)).dropna()
        
        return self.simple_returns, self.log_returns
        
    def estimate_parameters(self):
        """
        Estimates expected returns, volatilities, covariance, and correlation.
        Returns are annualized using self.trading_days.
        Uses log returns for parameter estimation as standard in GBM.
        """
        if self.log_returns is None:
            self.calculate_returns()
            
        # Daily parameters
        mean_daily_return = self.log_returns.mean()
        daily_volatility = self.log_returns.std()
        
        # Annualized parameters
        mean_annual_return = mean_daily_return * self.trading_days
        annual_volatility = daily_volatility * np.sqrt(self.trading_days)
        
        # Covariance and correlation (daily)
        cov_matrix_daily = self.log_returns.cov()
        corr_matrix = self.log_returns.corr()
        
        latest_prices = self.data[self.assets].iloc[-1].to_dict()
        
        return {
            'assets': self.assets,
            'latest_prices': latest_prices,
            'mean_daily_return': mean_daily_return,
            'daily_volatility': daily_volatility,
            'mean_annual_return': mean_annual_return,
            'annual_volatility': annual_volatility,
            'cov_matrix_daily': cov_matrix_daily,
            'corr_matrix': corr_matrix
        }
