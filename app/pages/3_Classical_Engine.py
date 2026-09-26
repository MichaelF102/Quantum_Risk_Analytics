"""
Page 3: Classical Monte Carlo Risk Simulation Engine
"""
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    sys.modules["app"].__path__ = [app_dir]

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from app.state import initialize_app_theme, get_shared_sidebar_inputs
    from app.components.charts import plot_loss_distribution_with_tail, get_terminal_layout
    from app.components.metrics import render_terminal_kpi
    from app.components.explanations import explain_monte_carlo
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.charts import plot_loss_distribution_with_tail, get_terminal_layout
    from components.metrics import render_terminal_kpi
    from components.explanations import explain_monte_carlo

from src.classical.monte_carlo import MonteCarloEngine

initialize_app_theme()
inputs = get_shared_sidebar_inputs()
portfolio = inputs["portfolio"]
conf_level = inputs["conf_level"]
time_horizon = inputs["time_horizon"]
port_val = inputs["portfolio_val"]
mc_sims = inputs["mc_sims"]
seed = inputs["seed"]

st.markdown("""
<div style='border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;'>
    <h2 style='margin:0; font-family:monospace; color:#f0f6fc;'>// 03_CLASSICAL_MONTE_CARLO_ENGINE</h2>
    <div style='font-family:monospace; font-size:12px; color:#58a6ff;'>High-Performance Vectorized Geometric Brownian Motion & Multivariate Normal Simulation</div>
</div>
""", unsafe_allow_html=True)

# Simulation Configuration Controls
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    sim_method = st.selectbox("Simulation Model", ["GBM (Geometric Brownian Motion)", "MVN (Multivariate Normal)"])
with col_ctrl2:
    cov_override = st.selectbox("Covariance Estimation", ["Sample Covariance", "Ledoit-Wolf Shrinkage"])
with col_ctrl3:
    run_button = st.button("🚀 Execute Monte Carlo Simulation", use_container_width=True)

method_code = "gbm" if "GBM" in sim_method else "mvn"

# Initialize and run Monte Carlo Engine
mc_engine = MonteCarloEngine(
    initial_portfolio_value=port_val,
    weights=portfolio.weights,
    expected_returns=portfolio.returns.mean().values * 252,
    covariance_matrix=portfolio.covariance_matrix,
    asset_prices=portfolio.prices.iloc[-1].values,
    seed=seed
)

sim_res = mc_engine.run_simulation(
    n_simulations=mc_sims,
    time_horizon_days=time_horizon,
    confidence_levels=[0.90, 0.95, 0.99, 0.995],
    method=method_code
)

alpha_key = f"{int(conf_level*100)}%" if conf_level in [0.90, 0.95, 0.99] else "99.5%"
current_risk = sim_res["risk_metrics"].get(alpha_key, sim_res["risk_metrics"]["95%"])

# Metrics Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_terminal_kpi("Simulated Paths", f"{sim_res['n_simulations']:,}", badge="NUMPY VECTORIZED")
with c2:
    render_terminal_kpi("Execution Wall Time", f"{sim_res['runtime_seconds']*1000:.1f} ms", delta="Deterministic Seed = 42")
with c3:
    render_terminal_kpi(f"Classical VaR ({alpha_key})", f"-${current_risk['var_cash']:,.0f}", delta=f"{current_risk['var_pct']*100:.2f}% loss", delta_color="red")
with c4:
    render_terminal_kpi(f"Classical CVaR ({alpha_key})", f"-${current_risk['cvar_cash']:,.0f}", delta=f"{current_risk['cvar_pct']*100:.2f}% tail", delta_color="red")

st.markdown("<br>", unsafe_allow_html=True)

# Risk Metric Summary Table with 95% Confidence Intervals
st.markdown("### // ESTIMATED RISK METRICS WITH 95% ASYMPTOTIC CONFIDENCE INTERVALS")
table_data = []
for alpha_k, data in sim_res["risk_metrics"].items():
    ci_low, ci_high = data["var_ci_95"]
    table_data.append({
        "Confidence Level": f"{float(data['confidence_level'])*100:.1f}%",
        "VaR (%)": f"{data['var_pct']*100:.3f}%",
        "VaR ($)": f"-${data['var_cash']:,.0f}",
        "95% CI (VaR %)": f"[{ci_low*100:.3f}%, {ci_high*100:.3f}%]",
        "Std. Error (SE)": f"{data['var_se']*100:.4f}%",
        "CVaR (%)": f"{data['cvar_pct']*100:.3f}%",
        "CVaR ($)": f"-${data['cvar_cash']:,.0f}"
    })
st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations: Simulated Portfolio Terminal Values and Loss Histogram
v_col1, v_col2 = st.columns([6, 6])

with v_col1:
    # Terminal Portfolio Value Distribution
    fig_val = go.Figure()
    fig_val.add_trace(go.Histogram(
        x=sim_res["simulated_values"],
        nbinsx=70,
        marker=dict(color="#1f6feb", opacity=0.8),
        name="Terminal Value ($)"
    ))
    fig_val.add_vline(x=port_val, line_dash="dash", line_color="#8b949e", annotation_text=f"Initial: ${port_val:,.0f}")
    layout_val = get_terminal_layout(f"Simulated Terminal Portfolio Values ({time_horizon}D Horizon)", height=400)
    layout_val["xaxis"]["title"] = "Portfolio Value ($)"
    layout_val["yaxis"]["title"] = "Frequency"
    fig_val.update_layout(**layout_val)
    st.plotly_chart(fig_val, use_container_width=True)

with v_col2:
    # Loss Distribution with VaR and CVaR markers
    fig_loss = plot_loss_distribution_with_tail(
        losses=sim_res["losses_pct"],
        var_threshold=current_risk["var_pct"],
        cvar_threshold=current_risk["cvar_pct"],
        confidence_level=conf_level,
        title=f"Classical Monte Carlo Loss Distribution (α={conf_level})"
    )
    st.plotly_chart(fig_loss, use_container_width=True)

explain_monte_carlo()
