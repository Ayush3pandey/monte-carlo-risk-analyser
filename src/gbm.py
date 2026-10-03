import numpy as np

def simulate_gbm(initial_prices, mu, sigma, cholesky_matrix, num_simulations, horizon_days, seed=None):
    """
    Simulate Geometric Brownian Motion (GBM) for multiple correlated assets.
    
    Parameters:
    - initial_prices: numpy array of initial prices for each asset.
    - mu: numpy array of expected daily returns for each asset.
    - sigma: numpy array of daily volatilities for each asset.
    - cholesky_matrix: Cholesky decomposition of the historical correlation matrix (L).
    - num_simulations: Number of Monte Carlo simulations to run.
    - horizon_days: Forecast horizon in days.
    - seed: Random seed for reproducibility.
    
    Returns:
    - simulated_prices: numpy array of shape (horizon_days + 1, num_simulations, num_assets)
      containing the simulated price paths.
    """
    if seed is not None:
        np.random.seed(seed)
        
    num_assets = len(initial_prices)
    
    # Initialize array to store price paths
    # Dimensions: (time_steps, simulations, assets)
    simulated_prices = np.zeros((horizon_days + 1, num_simulations, num_assets))
    simulated_prices[0, :, :] = initial_prices
    
    # Pre-calculate drift component: (mu - 0.5 * sigma^2) * dt
    # Since mu and sigma are already daily, dt = 1
    dt = 1
    drift = (mu - 0.5 * sigma**2) * dt
    
    # Generate independent standard normal shocks for all steps and simulations
    # Dimensions: (horizon_days, num_simulations, num_assets)
    independent_shocks = np.random.normal(0, 1, size=(horizon_days, num_simulations, num_assets))
    
    for t in range(1, horizon_days + 1):
        # Extract shocks for current time step
        z = independent_shocks[t - 1] # Shape: (num_simulations, num_assets)
        
        # Transform independent shocks to correlated shocks
        # z is (num_simulations, num_assets)
        # cholesky_matrix (L) is (num_assets, num_assets)
        # We need to compute z @ L.T
        correlated_shocks = z @ cholesky_matrix.T
        
        # Calculate log returns for this time step
        # Shape: (num_simulations, num_assets)
        log_returns = drift + sigma * np.sqrt(dt) * correlated_shocks
        
        # Calculate new prices
        simulated_prices[t, :, :] = simulated_prices[t - 1, :, :] * np.exp(log_returns)
        
    return simulated_prices
