import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import DataLoader
from src.portfolio import Portfolio
from src.gbm import simulate_gbm
from src.risk_metrics import calculate_var, calculate_cvar
from src.stress_testing import apply_stress_scenario

st.set_page_config(page_title="Monte Carlo Portfolio Risk Analyzer", layout="wide")

# --- Title and Description ---
st.title("Monte Carlo Portfolio Risk Analyzer")
st.subheader("Portfolio Simulation, VaR, Expected Shortfall & Stress Testing")
st.markdown("""
This is an educational risk analysis tool using synthetic data to demonstrate Monte Carlo simulation for portfolio risk.
**Note**: It must NOT claim to be suitable for real trading, investment decisions, or production risk management.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
initial_investment = st.sidebar.number_input("Initial Portfolio Value ($)", value=100000.0, step=10000.0)
num_simulations = st.sidebar.number_input("Number of Simulations", value=10000, step=1000)
forecast_horizon = st.sidebar.number_input("Forecast Horizon (Days)", value=1, min_value=1, step=1)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
confidence_level_1 = 0.95
confidence_level_2 = 0.99

st.sidebar.header("Stress Testing")
vol_multiplier = st.sidebar.slider("Stress Volatility Multiplier", min_value=1.0, max_value=3.0, value=1.25, step=0.05)
drift_adjustment = st.sidebar.slider("Stress Drift Adjustment (Daily)", min_value=-0.05, max_value=0.05, value=0.0, step=0.005)

st.sidebar.header("Display Options")
num_paths_display = st.sidebar.selectbox("Simulated Paths to Display", options=[10, 50, 100], index=1)

# --- Load Data and Estimate Parameters ---
@st.cache_data
def load_data():
    loader = DataLoader("data/historical_prices.csv")
    loader.load_and_validate()
    params = loader.estimate_parameters()
    return params

try:
    params = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have generated the synthetic data using `python src/data_generator.py`.")
    st.stop()
    
assets = params['assets']
latest_prices = params['latest_prices']
mu = params['mean_daily_return'].values
sigma = params['daily_volatility'].values
corr_matrix = params['corr_matrix'].values
current_prices_arr = np.array([latest_prices[a] for a in assets])

# Cholesky decomposition
try:
    L = np.linalg.cholesky(corr_matrix)
except np.linalg.LinAlgError:
    st.error("Correlation matrix is not positive definite. Cholesky decomposition failed.")
    st.stop()

# --- Portfolio Weights Controls ---
st.sidebar.header("Portfolio Allocation (%)")
weights_input = []
remaining_weight = 100.0
for i, asset in enumerate(assets):
    w = st.sidebar.number_input(f"{asset} Weight (%)", min_value=0.0, max_value=100.0, value=100.0/len(assets), step=1.0)
    weights_input.append(w)

weights_arr = np.array(weights_input) / 100.0

try:
    portfolio = Portfolio(assets, weights_arr, initial_investment)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

# --- Base Simulation ---
simulated_prices = simulate_gbm(
    current_prices_arr, mu, sigma, L, 
    num_simulations, forecast_horizon, seed=int(random_seed)
)

final_values, losses = portfolio.calculate_simulated_values(simulated_prices, current_prices_arr)

# Risk Metrics - Base
var_95 = calculate_var(losses, 0.95)
cvar_95 = calculate_cvar(losses, 0.95)
var_99 = calculate_var(losses, 0.99)
cvar_99 = calculate_cvar(losses, 0.99)

expected_final_value = np.mean(final_values)
expected_return_pct = (expected_final_value / initial_investment - 1) * 100
portfolio_volatility = np.std(final_values)

# --- Stressed Simulation ---
stressed_mu, stressed_sigma = apply_stress_scenario(mu, sigma, vol_multiplier, drift_adjustment)
simulated_prices_stress = simulate_gbm(
    current_prices_arr, stressed_mu, stressed_sigma, L, 
    num_simulations, forecast_horizon, seed=int(random_seed)
)
final_values_stress, losses_stress = portfolio.calculate_simulated_values(simulated_prices_stress, current_prices_arr)

# Risk Metrics - Stressed
stress_var_99 = calculate_var(losses_stress, 0.99)
stress_cvar_99 = calculate_cvar(losses_stress, 0.99)

# --- KPIs ---
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Initial Portfolio Value", f"${initial_investment:,.2f}")
    st.metric("Expected Final Value", f"${expected_final_value:,.2f}")
with col2:
    st.metric("Expected Return", f"{expected_return_pct:.2f}%")
    st.metric("Portfolio Volatility ($)", f"${portfolio_volatility:,.2f}")
with col3:
    st.metric("95% VaR", f"${var_95:,.2f}")
    st.metric("95% CVaR", f"${cvar_95:,.2f}")
with col4:
    st.metric("99% VaR", f"${var_99:,.2f}")
    st.metric("99% CVaR", f"${cvar_99:,.2f}")

st.markdown("---")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Stress 99% VaR", f"${stress_var_99:,.2f}", f"${stress_var_99 - var_99:,.2f}", delta_color="inverse")
with col_s2:
    st.metric("Stress 99% CVaR", f"${stress_cvar_99:,.2f}", f"${stress_cvar_99 - cvar_99:,.2f}", delta_color="inverse")

# --- Visualizations ---
st.header("Visualizations")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Portfolio Allocation")
    alloc_df = pd.DataFrame({
        'Asset': assets,
        'Weight': weights_arr,
        'Value': portfolio.calculate_position_values(),
        'Current Price': current_prices_arr
    })
    fig_pie = px.pie(alloc_df, values='Value', names='Asset', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.dataframe(alloc_df.style.format({'Weight': '{:.2%}', 'Value': '${:,.2f}', 'Current Price': '${:,.2f}'}))

with col_chart2:
    st.subheader("Correlation Heatmap")
    fig_corr = px.imshow(
        corr_matrix, 
        x=assets, y=assets, 
        color_continuous_scale='RdBu_r', 
        zmin=-1, zmax=1,
        text_auto=".2f"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("*High positive correlation means assets tend to move together. Negative correlation means they tend to move in opposite directions.*")

st.markdown("---")

# Loss Distribution Histogram
st.subheader("Simulated Portfolio Loss Distribution")
st.markdown("Positive values represent **losses**. The tail on the right represents the most extreme losses.")
fig_hist = px.histogram(
    x=losses, nbins=100, 
    labels={'x': 'Simulated Portfolio Loss ($)', 'y': 'Frequency'},
    opacity=0.75, color_discrete_sequence=['#ef553b']
)

# Add VaR/CVaR lines
fig_hist.add_vline(x=var_95, line_dash="dash", line_color="blue", annotation_text="95% VaR")
fig_hist.add_vline(x=cvar_95, line_dash="dot", line_color="darkblue", annotation_text="95% CVaR")
fig_hist.add_vline(x=var_99, line_dash="dash", line_color="black", annotation_text="99% VaR")
fig_hist.add_vline(x=cvar_99, line_dash="dot", line_color="black", annotation_text="99% CVaR")

st.plotly_chart(fig_hist, use_container_width=True)

# Simulated Price Paths
st.subheader(f"Sample of {num_paths_display} Simulated Asset Paths")
# Create a dropdown to select which asset to view
selected_asset_idx = st.selectbox("Select Asset to view paths", range(len(assets)), format_func=lambda x: assets[x])

fig_paths = go.Figure()
# Shape of simulated_prices: (horizon_days + 1, num_simulations, num_assets)
x_days = np.arange(forecast_horizon + 1)
for i in range(num_paths_display):
    fig_paths.add_trace(go.Scatter(
        x=x_days, 
        y=simulated_prices[:, i, selected_asset_idx],
        mode='lines',
        line=dict(width=1),
        opacity=0.5,
        showlegend=False
    ))

fig_paths.update_layout(
    xaxis_title="Days",
    yaxis_title="Price ($)",
    hovermode=False
)
st.plotly_chart(fig_paths, use_container_width=True)

# --- Risk Summary ---
st.header("Risk Summary")
st.markdown(f"""
### Portfolio Risk Summary
- **95% VaR:** ${var_95:,.2f}
- **95% CVaR:** ${cvar_95:,.2f}
- **99% VaR:** ${var_99:,.2f}
- **99% CVaR:** ${cvar_99:,.2f}

**Stress Scenario ({vol_multiplier}x Volatility, {drift_adjustment} Drift):**
- **Stress VaR (99%):** ${stress_var_99:,.2f}
- **Stress CVaR (99%):** ${stress_cvar_99:,.2f}

**Explanation:**
Under the current simulation assumptions, the 99% VaR is higher than the 95% VaR because it represents a more extreme loss percentile. The CVaR represents the average of the losses that exceed the VaR threshold, providing information about the tail risk that VaR does not capture.

*Disclaimer: This is a scenario analysis, not a prediction. The model relies on Geometric Brownian Motion which assumes continuous price evolution and may not fully capture jumps or heavy tails present in real financial markets. Historical parameters might not represent future behavior.*
""")
