import numpy as np

def calculate_var(losses, confidence_level=0.95):
    """
    Calculates Value-at-Risk (VaR) from a distribution of simulated losses.
    
    Parameters:
    - losses: numpy array of simulated portfolio losses (positive values = losses).
    - confidence_level: VaR confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - var: The Value-at-Risk at the given confidence level.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be strictly between 0 and 1.")
        
    # VaR is the percentile of the loss distribution
    # For 95% confidence, we want the 95th percentile of losses
    percentile = confidence_level * 100
    var = np.percentile(losses, percentile)
    return var

def calculate_cvar(losses, confidence_level=0.95):
    """
    Calculates Conditional Value-at-Risk (CVaR) / Expected Shortfall (ES).
    
    Parameters:
    - losses: numpy array of simulated portfolio losses (positive values = losses).
    - confidence_level: VaR confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - cvar: The Expected Shortfall beyond the VaR threshold.
    """
    # First find the VaR threshold
    var_threshold = calculate_var(losses, confidence_level)
    
    # CVaR is the expected loss given that the loss exceeds VaR
    tail_losses = losses[losses >= var_threshold]
    
    # If there are no losses beyond VaR (e.g., very few simulations or discrete distribution),
    # fallback to the threshold itself to avoid NaN.
    if len(tail_losses) == 0:
        return var_threshold
        
    cvar = np.mean(tail_losses)
    return cvar
