"""
Page 2: Comprehensive Risk Analysis - Historical, Parametric, and Monte Carlo
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

try:
    from app.state import initialize_app_theme, get_shared_sidebar_inputs
    from app.components.charts import plot_loss_distribution_with_tail
    from app.components.metrics import render_terminal_kpi
    from app.components.explanations import explain_var, explain_cvar
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.charts import plot_loss_distribution_with_tail
    from components.metrics import render_terminal_kpi
    from components.explanations import explain_var, explain_cvar

from src.classical.historical_simulation import run_historical_simulation
from src.classical.parametric_var import compute_parametric_normal_risk, compute_parametric_student_t_risk
from src.classical.monte_carlo import MonteCarloEngine

initialize_app_theme()
inputs = get_shared_sidebar_inputs()
portfolio = inputs["portfolio"]
conf_level = inputs["conf_level"]
time_horizon = inputs["time_horizon"]
port_val = inputs["portfolio_val"]

st.markdown("""
<div style='border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;'>
    <h2 style='margin:0; font-family:monospace; color:#f0f6fc;'>// 02_RISK_ANALYSIS</h2>
    <div style='font-family:monospace; font-size:12px; color:#58a6ff;'>Multi-Method Tail Risk Evaluation: Historical, Parametric, and Empirical Monte Carlo</div>
</div>
""", unsafe_allow_html=True)

# 1. Historical Simulation
hist_res = run_historical_simulation(
    portfolio.portfolio_returns,
    confidence_levels=[0.90, 0.95, 0.99],
    time_horizon_days=time_horizon,
    portfolio_value=port_val
)

# 2. Parametric Risk Models
daily_mu = float(portfolio.portfolio_returns.mean())
daily_vol = float(portfolio.portfolio_returns.std())
norm_risk = compute_parametric_normal_risk(daily_mu, daily_vol, confidence_level=conf_level, time_horizon_days=time_horizon, portfolio_value=port_val)
student_risk = compute_parametric_student_t_risk(portfolio.portfolio_returns.values, confidence_level=conf_level, time_horizon_days=time_horizon, portfolio_value=port_val)

# 3. Monte Carlo Simulation
mc_engine = MonteCarloEngine(
    initial_portfolio_value=port_val,
    weights=portfolio.weights,
    expected_returns=portfolio.returns.mean().values * 252,
    covariance_matrix=portfolio.covariance_matrix,
    asset_prices=portfolio.prices.iloc[-1].values,
    seed=inputs["seed"]
)
mc_res = mc_engine.run_simulation(
    n_simulations=inputs["mc_sims"],
    time_horizon_days=time_horizon,
    confidence_levels=[0.90, 0.95, 0.99]
)

# KPI Cards for chosen confidence level
alpha_str = f"{int(conf_level * 100)}%"
hist_data = hist_res["results"].get(alpha_str, hist_res["results"]["95%"])
mc_data = mc_res["risk_metrics"].get(alpha_str, mc_res["risk_metrics"]["95%"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_terminal_kpi(f"Historical VaR ({alpha_str})", f"-${hist_data['var_cash']:,.0f}", delta=f"{hist_data['var_pct']*100:.2f}%", delta_color="red")
with c2:
    render_terminal_kpi(f"Gaussian Parametric VaR", f"-${norm_risk['var_cash']:,.0f}", delta=f"{norm_risk['var_pct']*100:.2f}%", delta_color="red")
with c3:
    render_terminal_kpi(f"Student-t VaR (df={student_risk['degrees_of_freedom']:.1f})", f"-${student_risk['var_cash']:,.0f}", delta=f"{student_risk['var_pct']*100:.2f}%", delta_color="red")
with c4:
    render_terminal_kpi(f"Monte Carlo VaR ({inputs['mc_sims']:,})", f"-${mc_data['var_cash']:,.0f}", delta=f"{mc_data['var_pct']*100:.2f}%", delta_color="red")

st.markdown("<br>", unsafe_allow_html=True)

# Methodology Comparison Table
st.markdown("### // CROSS-METHODOLOGY RISK COMPARISON MATRIX")
summary_rows = [
    {
        "Methodology": "Historical Simulation (Empirical)",
        f"VaR 90% ($)": f"-${hist_res['results']['90%']['var_cash']:,.0f}",
        f"VaR 95% ($)": f"-${hist_res['results']['95%']['var_cash']:,.0f}",
        f"VaR 99% ($)": f"-${hist_res['results']['99%']['var_cash']:,.0f}",
        f"CVaR 95% ($)": f"-${hist_res['results']['95%']['cvar_cash']:,.0f}",
        "Assumption": "No distribution assumed (past repeats)"
    },
    {
        "Methodology": "Parametric Gaussian (Normal)",
        f"VaR 90% ($)": f"-${compute_parametric_normal_risk(daily_mu, daily_vol, 0.90, time_horizon, port_val)['var_cash']:,.0f}",
        f"VaR 95% ($)": f"-${compute_parametric_normal_risk(daily_mu, daily_vol, 0.95, time_horizon, port_val)['var_cash']:,.0f}",
        f"VaR 99% ($)": f"-${compute_parametric_normal_risk(daily_mu, daily_vol, 0.99, time_horizon, port_val)['var_cash']:,.0f}",
        f"CVaR 95% ($)": f"-${compute_parametric_normal_risk(daily_mu, daily_vol, 0.95, time_horizon, port_val)['cvar_cash']:,.0f}",
        "Assumption": "Strictly Gaussian i.i.d. returns"
    },
    {
        "Methodology": "Parametric Student-t (Fat Tails)",
        f"VaR 90% ($)": f"-${compute_parametric_student_t_risk(portfolio.portfolio_returns.values, 0.90, time_horizon, port_val)['var_cash']:,.0f}",
        f"VaR 95% ($)": f"-${compute_parametric_student_t_risk(portfolio.portfolio_returns.values, 0.95, time_horizon, port_val)['var_cash']:,.0f}",
        f"VaR 99% ($)": f"-${compute_parametric_student_t_risk(portfolio.portfolio_returns.values, 0.99, time_horizon, port_val)['var_cash']:,.0f}",
        f"CVaR 95% ($)": f"-${compute_parametric_student_t_risk(portfolio.portfolio_returns.values, 0.95, time_horizon, port_val)['cvar_cash']:,.0f}",
        "Assumption": "Fitted leptokurtic Student-t distribution"
    },
    {
        "Methodology": "Classical Monte Carlo (Vectorized GBM)",
        f"VaR 90% ($)": f"-${mc_res['risk_metrics']['90%']['var_cash']:,.0f}",
        f"VaR 95% ($)": f"-${mc_res['risk_metrics']['95%']['var_cash']:,.0f}",
        f"VaR 99% ($)": f"-${mc_res['risk_metrics']['99%']['var_cash']:,.0f}",
        f"CVaR 95% ($)": f"-${mc_res['risk_metrics']['95%']['cvar_cash']:,.0f}",
        "Assumption": "Multivariate correlated diffusion"
    }
]
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Loss Distribution with Tail Cutoffs
st.markdown("### // INTERACTIVE LOSS DISTRIBUTION & CVaR TAIL SHADING")
fig_loss = plot_loss_distribution_with_tail(
    losses=mc_res["losses_pct"],
    var_threshold=mc_data["var_pct"],
    cvar_threshold=mc_data["cvar_pct"],
    confidence_level=conf_level,
    title=f"Simulated Monte Carlo Loss Distribution ({inputs['mc_sims']:,} Paths, {time_horizon}D Horizon)"
)
st.plotly_chart(fig_loss, use_container_width=True)

# Tail Observations
st.markdown("### // WORST-CASE TAIL LOSS SCENARIOS (SIMULATED OBSERVATIONS)")
tail_mask = mc_res["losses_pct"] >= mc_data["var_pct"]
tail_losses = mc_res["losses_pct"][tail_mask]
tail_losses_sorted = np.sort(tail_losses)[::-1][:10]

tail_df = pd.DataFrame({
    "Rank": list(range(1, len(tail_losses_sorted) + 1)),
    "Simulated Loss (%)": [f"{l*100:.2f}%" for l in tail_losses_sorted],
    "Simulated Loss ($)": [f"-${l*port_val:,.0f}" for l in tail_losses_sorted],
    "Tail Classification": ["Severe Tail Event (> VaR)" for _ in tail_losses_sorted]
})
st.dataframe(tail_df, hide_index=True, use_container_width=True)

explain_var()
explain_cvar()
