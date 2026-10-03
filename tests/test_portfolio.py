import pytest
import numpy as np
from src.portfolio import Portfolio

def test_portfolio_initialization():
    assets = ['A', 'B']
    weights = [0.6, 0.4]
    investment = 10000
    
    portfolio = Portfolio(assets, weights, investment)
    assert portfolio.num_assets == 2
    np.testing.assert_array_equal(portfolio.weights, np.array([0.6, 0.4]))
    assert portfolio.initial_investment == 10000
    
def test_portfolio_validation_errors():
    assets = ['A', 'B']
    
    # Invalid weights sum
    with pytest.raises(ValueError):
        Portfolio(assets, [0.5, 0.6], 10000)
        
    # Invalid number of weights
    with pytest.raises(ValueError):
        Portfolio(assets, [1.0], 10000)
        
    # Invalid initial investment
    with pytest.raises(ValueError):
        Portfolio(assets, [0.5, 0.5], 0)

def test_portfolio_position_values():
    assets = ['A', 'B']
    weights = [0.6, 0.4]
    investment = 10000
    
    portfolio = Portfolio(assets, weights, investment)
    pos_values = portfolio.calculate_position_values()
    
    np.testing.assert_array_almost_equal(pos_values, np.array([6000.0, 4000.0]))

def test_portfolio_simulated_values():
    assets = ['A', 'B']
    weights = [0.5, 0.5]
    investment = 10000
    current_prices = np.array([100.0, 200.0]) # Shares: 50 of A, 25 of B
    
    portfolio = Portfolio(assets, weights, investment)
    
    # Mock simulated prices
    # Shape: (horizon_days + 1, num_simulations, num_assets)
    # Let's make it 2 days, 2 simulations, 2 assets
    simulated_prices = np.array([
        [[100.0, 200.0], [100.0, 200.0]], # t=0
        [[110.0, 190.0], [90.0, 210.0]]   # t=1
    ])
    
    final_values, losses = portfolio.calculate_simulated_values(simulated_prices, current_prices)
    
    # Sim 1 final value: 50*110 + 25*190 = 5500 + 4750 = 10250
    # Sim 2 final value: 50*90 + 25*210 = 4500 + 5250 = 9750
    np.testing.assert_array_almost_equal(final_values, np.array([10250.0, 9750.0]))
    
    # Losses = Initial - Final
    # Sim 1 loss: 10000 - 10250 = -250 (profit)
    # Sim 2 loss: 10000 - 9750 = +250 (loss)
    np.testing.assert_array_almost_equal(losses, np.array([-250.0, 250.0]))
