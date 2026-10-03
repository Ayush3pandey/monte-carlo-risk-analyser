import numpy as np

def apply_stress_scenario(mu, sigma, vol_multiplier=1.0, drift_adjustment=0.0):
    """
    Applies stress scenario shocks to market parameters.
    
    Parameters:
    - mu: numpy array of expected daily returns.
    - sigma: numpy array of daily volatilities.
    - vol_multiplier: multiplier to apply to volatility (e.g. 1.25 for +25%).
    - drift_adjustment: absolute adjustment to expected return (e.g. -0.01 for -1%).
    
    Returns:
    - stressed_mu: numpy array of stressed expected returns.
    - stressed_sigma: numpy array of stressed volatilities.
    """
    if vol_multiplier < 0:
        raise ValueError("Volatility multiplier cannot be negative.")
        
    stressed_sigma = sigma * vol_multiplier
    stressed_mu = mu + drift_adjustment
    
    return stressed_mu, stressed_sigma
