import pytest
import numpy as np
from src.stress_testing import apply_stress_scenario

def test_apply_stress_scenario():
    mu = np.array([0.01, 0.02])
    sigma = np.array([0.1, 0.2])
    
    # Apply shocks
    stressed_mu, stressed_sigma = apply_stress_scenario(mu, sigma, vol_multiplier=1.5, drift_adjustment=-0.01)
    
    np.testing.assert_array_almost_equal(stressed_mu, np.array([0.00, 0.01]))
    np.testing.assert_array_almost_equal(stressed_sigma, np.array([0.15, 0.3]))
    
def test_apply_stress_scenario_invalid():
    mu = np.array([0.01])
    sigma = np.array([0.1])
    
    with pytest.raises(ValueError):
        apply_stress_scenario(mu, sigma, vol_multiplier=-0.5)
