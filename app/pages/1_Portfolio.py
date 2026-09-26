"""
Page 1: Portfolio Construction & Optimization Analysis
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
import plotly.figure_factory as ff
import plotly.graph_objects as go

try:
    from app.state import initialize_app_theme, get_shared_sidebar_inputs
    from app.components.charts import (
        plot_allocation_donut,
        plot_cumulative_returns,
        plot_rolling_volatility,
        plot_correlation_heatmap,
        get_terminal_layout
    )
    from app.components.metrics import render_terminal_kpi
    from app.components.explanations import explain_portfolio_return
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.charts import (
        plot_allocation_donut,
        plot_cumulative_returns,
        plot_rolling_volatility,
        plot_correlation_heatmap,
        get_terminal_layout
    )
    from components.metrics import render_terminal_kpi
    from components.explanations import explain_portfolio_return

from src.data.preprocessing import calculate_rolling_volatility
from src.portfolio.portfolio import Portfolio

initialize_app_theme()
inputs = get_shared_sidebar_inputs()
prices_df = inputs["prices_df"]
portfolio = inputs["portfolio"]
assets = portfolio.assets

st.markdown("""
<div style='border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;'>
    <h2 style='margin:0; font-family:monospace; color:#f0f6fc;'>// 01_PORTFOLIO_CONSTRUCTION</h2>
    <div style='font-family:monospace; font-size:12px; color:#58a6ff;'>Asset Allocation, Covariance Structure, and Historical Performance Analysis</div>
</div>
""", unsafe_allow_html=True)

# Weight Adjustment Section
with st.expander("⚙️ Adjust Portfolio Holdings & Weights", expanded=True):
    st.markdown("<div style='font-family:monospace; font-size:12px; color:#8b949e; margin-bottom:10px;'>MODIFY ASSET ALLOCATION WEIGHTS (MUST SUM TO 100%):</div>", unsafe_allow_html=True)
    cols = st.columns(min(len(assets), 4))
    new_weights = []
    
    current_weights = portfolio.weights
    for i, asset in enumerate(assets):
        col_idx = i % 4
        with cols[col_idx]:
            val = st.number_input(
                f"{asset} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(current_weights[i] * 100.0),
                step=5.0,
                key=f"w_input_{asset}"
            )
            new_weights.append(val / 100.0)
            
    total_w = sum(new_weights)
    if abs(total_w - 1.0) > 1e-4:
        st.warning(f"⚠️ Total allocation is currently {total_w * 100:.1f}%. It must equal 100.0% to rebalance.")
    else:
        st.success(f"✓ Total allocation is exactly 100.0%. Portfolio weights are valid.")
        if st.button("Apply New Allocation Weights"):
            portfolio.update_weights(np.array(new_weights))
            st.session_state["portfolio"] = portfolio
            st.rerun()

# KPI Metrics Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_terminal_kpi("Portfolio Value", f"${portfolio.initial_value:,.0f}")
with c2:
    render_terminal_kpi("Expected Return (Ann.)", f"{portfolio.ann_return * 100:.2f}%", delta="Historical Compound CAGR")
with c3:
    render_terminal_kpi("Volatility (Ann. σ)", f"{portfolio.ann_vol * 100:.2f}%", delta=f"{portfolio.portfolio_returns.std() * 100:.2f}% daily")
with c4:
    render_terminal_kpi("Sharpe Ratio", f"{portfolio.sharpe_ratio:.2f}", delta="Rf = 2.00%")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations Tab Set
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Asset Allocation",
    "2. Cumulative Performance",
    "3. Rolling Volatility",
    "4. Correlation Heatmap",
    "5. Return Distribution"
])

with tab1:
    col_a, col_b = st.columns([6, 6])
    with col_a:
        st.plotly_chart(plot_allocation_donut(portfolio.get_asset_allocation_dict()), use_container_width=True)
    with col_b:
        st.markdown("<div style='font-family:monospace; font-size:12px; color:#58a6ff;'>// HOLDINGS BREAKDOWN</div>", unsafe_allow_html=True)
        alloc_df = pd.DataFrame({
            "Asset": assets,
            "Weight (%)": [f"{w*100:.2f}%" for w in portfolio.weights],
            "Nominal Value ($)": [f"${w * portfolio.initial_value:,.2f}" for w in portfolio.weights],
            "Historical Ann. Vol (%)": [f"{v*100:.2f}%" for v in portfolio.asset_vols.values]
        })
        st.dataframe(alloc_df, hide_index=True, use_container_width=True)

with tab2:
    st.plotly_chart(plot_cumulative_returns(portfolio.cumulative_returns), use_container_width=True)

with tab3:
    rolling_vols = calculate_rolling_volatility(portfolio.returns, window=21, annualize=True)
    st.plotly_chart(plot_rolling_volatility(rolling_vols), use_container_width=True)

with tab4:
    st.plotly_chart(plot_correlation_heatmap(portfolio.correlation_matrix), use_container_width=True)

with tab5:
    # Portfolio Return Distribution Histogram
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=portfolio.portfolio_returns.values * 100.0,
        nbinsx=60,
        marker=dict(color="#58a6ff", opacity=0.75),
        name="Daily Return (%)"
    ))
    layout = get_terminal_layout("Historical Daily Return Distribution (%)", height=380)
    layout["xaxis"]["title"] = "Daily Return (%)"
    layout["yaxis"]["title"] = "Frequency"
    fig_ret.update_layout(**layout)
    st.plotly_chart(fig_ret, use_container_width=True)

# Portfolio Equation & Mathematical Meaning
st.markdown("---")
st.markdown("### // MATHEMATICAL FORMULATION")
st.latex(r"R_p = w^T R = \sum_{i=1}^N w_i R_i")
st.markdown(r"""
* **$R_p$**: Aggregate portfolio return at time $t$.
* **$w$**: Allocation weight column vector $[w_1, w_2, \dots, w_N]^T$ satisfying the budget constraint $\sum_{i=1}^N w_i = 1$.
* **$R$**: Vector of stochastic asset returns at time $t$.
* **Covariance Structure**: Portfolio risk satisfies $\sigma_p^2 = w^T \Sigma w$, where cross-asset correlations $\rho_{ij}$ provide the mathematical foundation for diversification benefits.
""")

explain_portfolio_return()
