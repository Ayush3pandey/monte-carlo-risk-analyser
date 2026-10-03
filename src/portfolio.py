import numpy as np

class Portfolio:
    def __init__(self, assets, weights, initial_investment):
        self.assets = assets
        self.weights = np.array(weights)
        self.initial_investment = initial_investment
        self.num_assets = len(assets)
        self.validate()
        
    def validate(self):
        """Validates portfolio parameters."""
        if len(self.weights) != self.num_assets:
            raise ValueError(f"Expected {self.num_assets} weights, got {len(self.weights)}.")
            
        if not np.isclose(np.sum(self.weights), 1.0, rtol=1e-5):
            raise ValueError(f"Portfolio weights must sum to 1, currently sum to {np.sum(self.weights)}.")
            
        if self.initial_investment <= 0:
            raise ValueError("Initial investment must be strictly positive.")
            
    def calculate_position_values(self):
        """Calculates the initial monetary value of each position."""
        return self.weights * self.initial_investment
        
    def calculate_simulated_values(self, simulated_prices, current_prices):
        """
        Calculates simulated portfolio values and losses.
        
        Parameters:
        - simulated_prices: numpy array of shape (horizon_days + 1, num_simulations, num_assets)
        - current_prices: numpy array of current prices for the assets (shape: num_assets)
        
        Returns:
        - final_portfolio_values: numpy array of shape (num_simulations,)
        - portfolio_losses: numpy array of shape (num_simulations,) (Loss = Initial - Final)
        """
        # Number of shares bought initially
        position_values = self.calculate_position_values()
        shares = position_values / current_prices
        
        # Calculate final portfolio value across all simulations
        # Final prices are at index horizon_days (the last element in the time dimension)
        final_prices = simulated_prices[-1, :, :] # Shape: (num_simulations, num_assets)
        
        final_portfolio_values = np.sum(final_prices * shares, axis=1) # Shape: (num_simulations,)
        
        # Calculate P&L: Final Value - Initial Value
        pnl = final_portfolio_values - self.initial_investment
        
        # Loss convention: Loss = -P&L
        # Positive values represent losses, negative values represent profits
        losses = -pnl
        
        return final_portfolio_values, losses
