# Monte Carlo Portfolio Risk Analyzer

## 1. Project Overview
The Monte Carlo Portfolio Risk Analyzer is an educational risk management tool that simulates the future behavior of a multi-asset portfolio to estimate its risk exposure. It leverages Geometric Brownian Motion (GBM) to project correlated asset price paths and calculates essential risk metrics: Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) or Expected Shortfall (ES). 

## 2. Problem Statement
Financial risk management requires understanding the likelihood and magnitude of potential future losses. Traditional risk measures like historical simulation rely heavily on the past repeating exactly. A model-based approach like Monte Carlo allows us to generate thousands of statistically plausible future scenarios based on current estimates of market parameters (drift, volatility, and correlation) to better quantify tail risks.

## 3. Why Monte Carlo Simulation is Useful for Risk
Monte Carlo simulation allows risk managers to model complex portfolios with interacting assets where analytical solutions are difficult or impossible to formulate. By sampling thousands of possible future states of the world, we can empirically build a distribution of portfolio values and directly measure the probability of severe losses.

## 4. Financial Returns
We define returns as the percentage change in asset prices. 
*   **Simple Return:** $R_t = \frac{P_t}{P_{t-1}} - 1$
*   **Log Return:** $r_t = \ln(\frac{P_t}{P_{t-1}})$
Log returns are commonly used in financial modeling because they are time-additive, making them mathematically convenient for simulating continuous-time processes.

## 5. Volatility
Volatility ($\sigma$) represents the degree of variation in an asset's trading price over time. It is estimated as the standard deviation of daily log returns and annualized by multiplying by the square root of time (e.g., $\sqrt{252}$ for daily data).

## 6. Drift
Drift ($\mu$) is the expected rate of return of an asset over time. In a risk-neutral or real-world simulation, it is the constant directional bias of the asset's price path.

## 7. Geometric Brownian Motion (GBM)
GBM is a standard model for asset prices that ensures prices remain strictly positive. It assumes that log returns are normally distributed.
The discrete-time simulation formula is:
$$S(t+dt) = S(t) \cdot \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma \sqrt{dt} Z\right)$$
Where:
- $S(t)$ is the current price
- $\mu$ is the drift
- $\sigma$ is the volatility
- $dt$ is the time step
- $Z$ is a standard normal random variable

## 8. Monte Carlo Simulation
The Monte Carlo engine generates random values for $Z$ across thousands of iterations (scenarios) for a specified forecast horizon. This results in a wide array of possible future price paths for each asset.

## 9. Portfolio Construction
A portfolio is a weighted combination of multiple assets. 
Initial position value = Initial Investment $\times$ Weight
The future portfolio value is simply the sum of the future position values across all assets.

## 10. Correlation & 11. Covariance
Assets do not move in isolation. 
*   **Covariance:** A measure of the joint variability of two assets.
*   **Correlation:** A normalized measure of covariance ranging from -1 to 1.
If two assets tend to fall together (positive correlation), ignoring this relationship can severely underestimate portfolio tail risk.

## 12. Cholesky Decomposition
To simulate correlated asset prices, we start with independent standard normal variables and transform them. The Cholesky decomposition of the correlation matrix ($\Sigma = L L^T$) gives us a lower triangular matrix $L$. Multiplying independent random shocks by $L$ yields correlated shocks that preserve the historical relationships between assets.

## 13. Portfolio P&L
For each simulated scenario, we calculate the Profit & Loss (P&L):
$P\&L = \text{Future Portfolio Value} - \text{Initial Portfolio Value}$
For risk calculations, we define Loss as negative P&L. Thus, positive loss values indicate portfolio losses.

## 14. Value-at-Risk (VaR)
VaR is a threshold value such that the probability that the loss exceeds this threshold is a specified level (e.g., 5% for 95% VaR). It is computed empirically as the 95th or 99th percentile of the simulated loss distribution.

## 15. CVaR / Expected Shortfall
While VaR tells us where the tail begins, CVaR tells us what happens *inside* the tail. CVaR is the expected (average) loss given that the loss has exceeded the VaR threshold.

## 16. Stress Testing
We perform scenario-based stress testing by manually shocking the market parameters. For instance, multiplying historical volatility by 1.25x or shifting the expected drift downward. This reveals how VaR and CVaR respond to extreme, non-historical market conditions.

## 17. System Architecture
- `app.py`: Streamlit dashboard and user interface.
- `src/data_loader.py`: Data ingestion, validation, and parameter estimation.
- `src/gbm.py`: Core mathematical simulation using numpy.
- `src/portfolio.py`: Portfolio construction and valuation logic.
- `src/risk_metrics.py`: Calculation of VaR and CVaR.
- `src/stress_testing.py`: Scenario modification logic.
- `tests/`: Pytest unit tests for all mathematical modules.

## 18. Data Methodology
**IMPORTANT**: The market dataset used in this project is synthetically generated for educational and portfolio demonstration purposes. No real market data is used. The data is generated using a reproducible random seed to exhibit meaningful volatilities, expected returns, and cross-asset correlations over a 3-year simulated period.

## 19. Installation
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## 20. Running the Dashboard
1. Ensure the synthetic data is generated:
   ```bash
   python src/data_generator.py
   ```
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 21. Running Tests
Run all unit tests using pytest:
```bash
pytest tests/
```

## 22. Example Workflow
1. Historical market data is loaded and daily returns are calculated.
2. The drift ($\mu$), volatility ($\sigma$), and correlation matrices are estimated.
3. The user inputs their desired portfolio weights and simulation parameters in the dashboard.
4. Independent random shocks are generated and transformed into correlated shocks via Cholesky decomposition.
5. The GBM model generates 10,000 future asset price paths.
6. The future portfolio values and losses are calculated.
7. VaR and CVaR are computed from the loss distribution and visualized.
8. The user adjusts stress-testing sliders to see how elevated volatility affects the risk metrics.

## 23. Limitations
*   **Continuous Price Evolution:** GBM assumes prices evolve continuously without jumps. In reality, markets can experience sudden gaps.
*   **Normal Returns Assumption:** The model assumes log returns are normally distributed (no heavy tails). Real financial returns exhibit "fat tails" (extreme events happen more frequently than a normal distribution predicts).
*   **Constant Parameters:** Historical volatility and correlation are assumed constant throughout the simulation, but in reality, correlations tend to spike toward 1.0 during market crashes.
*   **Demonstration Only:** Synthetic data is used. This project is not a production risk system and must not be used for actual trading or regulatory reporting.

## 24. Future Improvements
*   Implement jump-diffusion models (e.g., Merton model) to account for heavy tails.
*   Integrate actual historical API data (e.g., Yahoo Finance).
*   Add interest rate and dividend yield parameters to the drift calculation.
