import os
import numpy as np
import pandas as pd

def generate_synthetic_data(seed=42, num_days=756, output_path="data/historical_prices.csv"):
    """
    Generate synthetic daily price data for 5 assets with correlated returns.
    """
    np.random.seed(seed)
    
    # Define assets
    assets = ["Asset_A", "Asset_B", "Asset_C", "Asset_D", "Asset_E"]
    num_assets = len(assets)
    
    # Expected annual returns and volatilities
    # Asset A: Low risk, low return (e.g. Bonds)
    # Asset B: Medium risk, medium return (e.g. Value Stock)
    # Asset C: High risk, high return (e.g. Growth Stock)
    # Asset D: Medium-High risk, medium return
    # Asset E: Low risk, uncorrelated (e.g. Gold)
    mu_annual = np.array([0.04, 0.08, 0.12, 0.10, 0.05])
    vol_annual = np.array([0.05, 0.15, 0.25, 0.20, 0.12])
    
    # Convert to daily (assume 252 trading days)
    dt = 1/252
    mu_daily = mu_annual * dt
    vol_daily = vol_annual * np.sqrt(dt)
    
    # Define correlation matrix
    # A is weakly correlated with B, C, D
    # B, C, D are highly correlated
    # E is negatively or weakly correlated with the others
    corr_matrix = np.array([
        [ 1.0,  0.3,  0.2,  0.2,  0.1],
        [ 0.3,  1.0,  0.7,  0.6, -0.1],
        [ 0.2,  0.7,  1.0,  0.8, -0.2],
        [ 0.2,  0.6,  0.8,  1.0, -0.1],
        [ 0.1, -0.1, -0.2, -0.1,  1.0]
    ])
    
    # Calculate covariance matrix for daily returns
    # Cov(i,j) = corr(i,j) * vol(i) * vol(j)
    D = np.diag(vol_daily)
    cov_matrix = D @ corr_matrix @ D
    
    # Generate correlated daily log returns
    # We use multivariate normal: mean = mu_daily - 0.5 * vol_daily^2 (for log returns)
    # But let's generate simple random shocks first using Cholesky to be explicit about the process
    # Or just use multivariate normal which uses Cholesky under the hood.
    # The prompt says: "First define a reasonable correlation matrix and use it to generate correlated daily return shocks."
    
    # Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate independent standard normal shocks
    independent_shocks = np.random.normal(0, 1, size=(num_days, num_assets))
    
    # Transform to correlated shocks
    correlated_shocks = independent_shocks @ L.T
    
    # Calculate log returns
    # drift = mu_daily - 0.5 * vol_daily^2
    drift = mu_daily - 0.5 * (vol_daily ** 2)
    
    # LogReturn = drift + vol_daily * correlated_shocks
    log_returns = drift + vol_daily * correlated_shocks
    
    # Initial prices
    initial_prices = np.array([100.0, 150.0, 80.0, 120.0, 50.0])
    
    # Calculate price paths
    # Price_t = Price_0 * exp(cumsum(log_returns))
    cumulative_returns = np.cumsum(log_returns, axis=0)
    prices = initial_prices * np.exp(cumulative_returns)
    
    # Prepend initial prices to make the series start at day 0
    prices = np.vstack([initial_prices, prices])
    
    # Generate dates (skipping weekends to simulate trading days)
    start_date = pd.to_datetime('2020-01-01')
    dates = pd.bdate_range(start=start_date, periods=num_days + 1)
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=assets)
    df['Date'] = dates
    
    # Reorder columns to put Date first
    df = df[['Date'] + assets]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_days} days of data for {num_assets} assets at {output_path}")
    print("Correlation matrix used:")
    print(corr_matrix)

if __name__ == "__main__":
    generate_synthetic_data()
