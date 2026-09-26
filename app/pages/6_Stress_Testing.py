"""
Page 6: Stress Testing & Advanced Volatility Extensions (GARCH / EGARCH)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.state import initialize_app_theme, get_shared_sidebar_inputs
from app.components.charts import plot_stress_test_comparison, get_terminal_layout
from app.components.metrics import render_terminal_kpi
from src.classical.monte_carlo import MonteCarloEngine
from src.models.garch import fit_garch
from src.models.egarch import fit_egarch
from src.classical.var import compute_var_from_losses
from src.classical.cvar import compute_cvar_from_losses

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
    <h2 style='margin:0; font-family:monospace; color:#f0f6fc;'>// 06_STRESS_TESTING_AND_VOLATILITY</h2>
    <div style='font-family:monospace; font-size:12px; color:#58a6ff;'>Macro Stress Scenarios, Correlation Breakdown, and GARCH/EGARCH Extensions</div>
</div>
""", unsafe_allow_html=True)

stress_tab, garch_tab = st.tabs(["1. Macro Stress Scenarios", "2. GARCH / EGARCH Volatility Extension"])

with stress_tab:
    st.markdown("### // MACRO STRESS SIMULATION")
    
    # Baseline simulation
    mc_engine = MonteCarloEngine(
        initial_portfolio_value=port_val,
        weights=portfolio.weights,
        expected_returns=portfolio.returns.mean().values * 252,
        covariance_matrix=portfolio.covariance_matrix,
        asset_prices=portfolio.prices.iloc[-1].values,
        seed=seed
    )
    base_res = mc_engine.run_simulation(n_simulations=mc_sims, time_horizon_days=time_horizon)
    base_losses = base_res["losses_pct"]
    base_var = compute_var_from_losses(base_losses, confidence_level=conf_level)
    base_cvar = compute_cvar_from_losses(base_losses, confidence_level=conf_level)
    
    scenario_choice = st.selectbox(
        "Select Stress Testing Scenario",
        [
            "Market Crash (-15% Systematic Shock)",
            "Volatility Spike (+80% Volatility Surge)",
            "Correlation Breakdown (ρ → 0.90 Contagion)",
            "Single Asset Catastrophe (-40% Shock to Largest Holding)"
        ]
    )
    
    n_assets = portfolio.num_assets
    stressed_cov = portfolio.covariance_matrix.copy()
    stressed_mu = portfolio.returns.mean().values * 252
    
    if "Market Crash" in scenario_choice:
        stressed_mu = stressed_mu - 0.30  # Severe negative drift
    elif "Volatility Spike" in scenario_choice:
        stressed_cov = stressed_cov * (1.8 ** 2)  # 80% vol surge
    elif "Correlation Breakdown" in scenario_choice:
        vols = np.sqrt(np.diag(stressed_cov))
        contagion_corr = np.full((n_assets, n_assets), 0.90)
        np.fill_diagonal(contagion_corr, 1.0)
        stressed_cov = np.outer(vols, vols) * contagion_corr
    elif "Single Asset" in scenario_choice:
        largest_asset_idx = int(np.argmax(portfolio.weights))
        stressed_mu[largest_asset_idx] -= 0.60
        stressed_cov[largest_asset_idx, largest_asset_idx] *= 4.0
        
    stressed_engine = MonteCarloEngine(
        initial_portfolio_value=port_val,
        weights=portfolio.weights,
        expected_returns=stressed_mu,
        covariance_matrix=stressed_cov,
        asset_prices=portfolio.prices.iloc[-1].values,
        seed=seed
    )
    stressed_res = stressed_engine.run_simulation(n_simulations=mc_sims, time_horizon_days=time_horizon)
    stressed_losses = stressed_res["losses_pct"]
    stressed_var = compute_var_from_losses(stressed_losses, confidence_level=conf_level)
    stressed_cvar = compute_cvar_from_losses(stressed_losses, confidence_level=conf_level)
    
    # KPI Impact
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_terminal_kpi(f"Baseline VaR ({int(conf_level*100)}%)", f"{base_var*100:.2f}%", sublabel=f"-${base_var*port_val:,.0f}")
    with s2:
        var_delta_pct = (stressed_var - base_var) * 100.0
        render_terminal_kpi(f"Stressed VaR ({int(conf_level*100)}%)", f"{stressed_var*100:.2f}%", delta=f"+{var_delta_pct:.2f}% loss surge", delta_color="red")
    with s3:
        render_terminal_kpi(f"Baseline CVaR ({int(conf_level*100)}%)", f"{base_cvar*100:.2f}%", sublabel=f"-${base_cvar*port_val:,.0f}")
    with s4:
        cvar_delta_pct = (stressed_cvar - base_cvar) * 100.0
        render_terminal_kpi(f"Stressed CVaR ({int(conf_level*100)}%)", f"{stressed_cvar*100:.2f}%", delta=f"+{cvar_delta_pct:.2f}% tail surge", delta_color="red")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(
        plot_stress_test_comparison(
            base_losses,
            stressed_losses,
            scenario_choice.split("(")[0].strip(),
            base_var,
            stressed_var
        ),
        use_container_width=True
    )

with garch_tab:
    st.markdown("### // DYNAMIC VOLATILITY MODELING: GARCH & EGARCH")
    st.markdown("""
    Standard RiskMetrics and simple Monte Carlo assume constant historical volatility.
    In reality, financial returns display **volatility clustering** and **asymmetric leverage effects** (bad news induces higher volatility than good news).
    """)
    
    col_g1, col_g2 = st.columns([6, 6])
    with col_g1:
        with st.spinner("Fitting GARCH(1,1)..."):
            garch_res = fit_garch(portfolio.portfolio_returns)
            
        st.markdown("<div style='font-family:monospace; font-size:12px; color:#58a6ff;'>// GARCH(1,1) ESTIMATION</div>", unsafe_allow_html=True)
        if garch_res["success"]:
            st.markdown(f"""
            * **Model**: `GARCH(1,1)`
            * **Alpha (Shock reaction $\\alpha$)**: `{garch_res['alpha']:.4f}`
            * **Beta (Persistence $\\beta$)**: `{garch_res['beta']:.4f}`
            * **Total Persistence ($\\alpha + \\beta$)**: `{garch_res['persistence']:.4f}`
            * **Forecast Ann. Volatility**: `{garch_res['annualized_forecast_vol']*100:.2f}%`
            """)
        else:
            st.warning(f"GARCH fitting note: {garch_res.get('error')}")
            
    with col_g2:
        with st.spinner("Fitting EGARCH(1,1)..."):
            egarch_res = fit_egarch(portfolio.portfolio_returns)
            
        st.markdown("<div style='font-family:monospace; font-size:12px; color:#58a6ff;'>// EGARCH(1,1) ASYMMETRIC LEVERAGE</div>", unsafe_allow_html=True)
        if egarch_res["success"]:
            st.markdown(f"""
            * **Model**: `EGARCH(1,1)`
            * **Leverage Parameter ($\\gamma$)**: `{egarch_res['gamma']:.4f}` *(<0 implies negative returns spike volatility)*
            * **Persistence ($\\beta$)**: `{egarch_res['beta']:.4f}`
            * **Forecast Ann. Volatility**: `{egarch_res['annualized_forecast_vol']*100:.2f}%`
            """)
        else:
            st.warning(f"EGARCH fitting note: {egarch_res.get('error')}")
            
    # Volatility comparison table
    hist_ann_vol = portfolio.ann_vol
    garch_ann_vol = garch_res.get("annualized_forecast_vol", hist_ann_vol)
    egarch_ann_vol = egarch_res.get("annualized_forecast_vol", hist_ann_vol)
    
    vol_comp_df = pd.DataFrame([
        {"Volatility Metric": "Sample Historical Volatility (1Y)", "Annualized σ": f"{hist_ann_vol*100:.2f}%", "VaR 95% Impact": f"-${compute_var_from_losses(base_losses, 0.95)*port_val:,.0f}"},
        {"Volatility Metric": "GARCH(1,1) Conditional Forecast", "Annualized σ": f"{garch_ann_vol*100:.2f}%", "VaR 95% Impact": f"-${compute_var_from_losses(base_losses * (garch_ann_vol/hist_ann_vol), 0.95)*port_val:,.0f}"},
        {"Volatility Metric": "EGARCH(1,1) Asymmetric Leverage Forecast", "Annualized σ": f"{egarch_ann_vol*100:.2f}%", "VaR 95% Impact": f"-${compute_var_from_losses(base_losses * (egarch_ann_vol/hist_ann_vol), 0.95)*port_val:,.0f}"}
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(vol_comp_df, hide_index=True, use_container_width=True)
