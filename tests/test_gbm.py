import numpy as np
from src.gbm import simulate_gbm

def test_simulate_gbm_dimensions():
    initial_prices = np.array([100.0, 150.0])
    mu = np.array([0.001, 0.002])
    sigma = np.array([0.01, 0.02])
    cholesky = np.eye(2)
    horizon_days = 5
    num_simulations = 10
    
    simulated_prices = simulate_gbm(
        initial_prices, mu, sigma, cholesky, num_simulations, horizon_days, seed=42
    )
    
    # Check shape: (horizon_days + 1, num_simulations, num_assets)
    assert simulated_prices.shape == (6, 10, 2)
    
    # Check that t=0 prices match initial prices
    np.testing.assert_array_equal(simulated_prices[0, 0, :], initial_prices)
    
def test_simulate_gbm_positive_prices():
    initial_prices = np.array([100.0, 150.0])
    mu = np.array([0.001, 0.002])
    sigma = np.array([0.01, 0.02])
    cholesky = np.eye(2)
    horizon_days = 50
    num_simulations = 100
    
    simulated_prices = simulate_gbm(
        initial_prices, mu, sigma, cholesky, num_simulations, horizon_days, seed=42
    )
    
    # Prices must always be strictly positive for GBM
    assert (simulated_prices > 0).all()
