import pytest
import numpy as np
from src.risk_metrics import calculate_var, calculate_cvar

def test_calculate_var():
    # Array of losses from 1 to 100
    losses = np.arange(1, 101)
    
    # 95% VaR is calculated using linear interpolation: index = 0.95*(100-1) = 94.05 -> value 95.05
    var_95 = calculate_var(losses, 0.95)
    assert np.isclose(var_95, 95.05)
    
    # 99% VaR: index = 0.99*(99) = 98.01 -> value 99.01
    var_99 = calculate_var(losses, 0.99)
    assert np.isclose(var_99, 99.01)

def test_calculate_cvar():
    # Array of losses from 1 to 100
    losses = np.arange(1, 101)
    
    # 95% CVaR is mean of losses >= 95.05
    # Losses >= 95.05 are [96, 97, 98, 99, 100]
    # Mean is 98.0
    cvar_95 = calculate_cvar(losses, 0.95)
    assert np.isclose(cvar_95, 98.0)
    
    # 99% CVaR is mean of losses >= 99.01
    # Losses >= 99.01 are [100]
    # Mean is 100.0
    cvar_99 = calculate_cvar(losses, 0.99)
    assert np.isclose(cvar_99, 100.0)

def test_var_invalid_confidence():
    losses = np.arange(1, 101)
    with pytest.raises(ValueError):
        calculate_var(losses, 1.05)
    with pytest.raises(ValueError):
        calculate_var(losses, -0.05)
