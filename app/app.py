"""
Quantum Risk Analytics Engine - Home / Research Overview Dashboard
"""
import sys
import os

# Ensure root directory and app directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app_dir = os.path.abspath(os.path.dirname(__file__))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# If Streamlit registered app.py as module 'app', grant it package __path__ so app.state imports resolve
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    sys.modules["app"].__path__ = [app_dir]

import importlib
for _mod_name in [
    "app.components.metrics", "components.metrics",
    "app.state", "state",
    "app.components.charts", "components.charts",
    "app.components.tables", "components.tables",
    "app.components.explanations", "components.explanations"
]:
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass

import streamlit as st
import numpy as np
import pandas as pd

try:
    from app.state import initialize_app_theme, get_shared_sidebar_inputs
    from app.components.metrics import render_terminal_kpi, render_status_pill
    from app.components.charts import plot_cumulative_returns, plot_allocation_donut
    from app.components.explanations import explain_var, explain_cvar, explain_amplitude_estimation
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.metrics import render_terminal_kpi, render_status_pill
    from components.charts import plot_cumulative_returns, plot_allocation_donut
    from components.explanations import explain_var, explain_cvar, explain_amplitude_estimation

from src.classical.var import compute_var_from_returns
from src.classical.cvar import compute_cvar_from_returns

st.set_page_config(
    page_title="Quantum Risk Analytics Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    initialize_app_theme()
    inputs = get_shared_sidebar_inputs()
    portfolio = inputs["portfolio"]
    conf_level = inputs["conf_level"]
    port_val = inputs["portfolio_val"]

    # Title Banner
    st.html("""
<div style='padding: 10px 0px 20px 0px; border-bottom: 1px solid #30363d; margin-bottom: 25px;'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <h1 style='margin:0; font-size: 28px; font-weight:800; color:#f0f6fc; font-family:monospace;'>
                ⚡ QUANTUM RISK ANALYTICS ENGINE
            </h1>
            <div style='font-family:monospace; font-size:14px; color:#58a6ff; margin-top:5px;'>
                Hybrid Quantum-Classical Portfolio Risk Analysis & Tail Risk Benchmarking
            </div>
        </div>
        <div style='text-align:right;'>
            <span style='background:#238636; color:#ffffff; font-family:monospace; font-size:11px; font-weight:600; padding:4px 8px; border-radius:3px;'>ONLINE</span>
            <div style='font-family:monospace; font-size:11px; color:#8b949e; margin-top:4px;'>QISKIT 2.5 • AER SIMULATOR</div>
        </div>
    </div>
</div>
""")

    # Risk calculations for 95% default
    var_pct, var_cash = compute_var_from_returns(portfolio.portfolio_returns.values, confidence_level=conf_level, portfolio_value=port_val)
    cvar_pct, cvar_cash = compute_cvar_from_returns(portfolio.portfolio_returns.values, confidence_level=conf_level, portfolio_value=port_val)

    # Top KPI Row (6 cards)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_terminal_kpi("Portfolio Value", f"${port_val:,.0f}", badge="USD")
    with c2:
        ret_color = "green" if portfolio.ann_return >= 0 else "red"
        render_terminal_kpi("Ann. Return", f"{portfolio.ann_return * 100:.2f}%", delta=f"{portfolio.portfolio_returns.mean()*252*100:+.2f}% p.a.", delta_color=ret_color)
    with c3:
        render_terminal_kpi("Ann. Volatility", f"{portfolio.ann_vol * 100:.2f}%", sublabel="Historical 1Y σ")
    with c4:
        sharpe_color = "green" if portfolio.sharpe_ratio >= 1.0 else ("normal" if portfolio.sharpe_ratio >= 0 else "red")
        render_terminal_kpi("Sharpe Ratio", f"{portfolio.sharpe_ratio:.2f}", delta="Rf = 2.0%", delta_color=sharpe_color)
    with c5:
        render_terminal_kpi(f"VaR {int(conf_level*100)}% (10D)", f"-${var_cash:,.0f}", delta=f"{var_pct*100:.2f}% loss", delta_color="red")
    with c6:
        render_terminal_kpi(f"CVaR {int(conf_level*100)}% (10D)", f"-${cvar_cash:,.0f}", delta=f"{cvar_pct*100:.2f}% tail", delta_color="red")

    st.html("<br>")

    # Main Grid: Methodology & System Architecture
    col_left, col_right = st.columns([7, 5])

    with col_left:
        st.markdown("### CORE RESEARCH OBJECTIVE")
        st.markdown("""
        > **Research Inquiry:** *Can quantum amplitude-estimation techniques provide accurate estimates of portfolio tail risk,
        particularly VaR and CVaR, and how do they compare with classical Monte Carlo methods in terms of accuracy, convergence,
        and computational requirements?*
        
        This platform implements a scientifically rigorous, reproducible testbed comparing **Classical Monte Carlo** with **Quantum Amplitude Estimation (QAE)**.
        It separates **experimentally simulated executions** from **theoretical asymptotic advantages**, providing an objective empirical assessment of quantum speedups in quantitative finance.
        """)
        
        st.markdown("### HIGH-LEVEL METHODOLOGY PIPELINE")
        st.markdown("""
        ```
        ┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
        │  1. Classical Engine    │     │   2. Quantum Engine    │     │   3. Benchmark Lab     │
        │  • Historical Returns   │ ──► │  • State Encoding      │ ──► │  • Head-to-Head Error  │
        │  • Multivariate GBM     │     │  • Tail Loss Oracle    │     │  • O(1/√N) vs O(1/M)   │
        │  • Monte Carlo Sim      │     │  • QAE (MLQAE / Canon) │     │  • Scalability & Depth │
        └─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
        ```
        """)
        
        st.plotly_chart(plot_cumulative_returns(portfolio.cumulative_returns), width="stretch")

    with col_right:
        st.markdown("### ACTIVE PORTFOLIO COMPOSITION")
        st.plotly_chart(plot_allocation_donut(portfolio.get_asset_allocation_dict()), width="stretch")
        
        st.markdown("### SCIENTIFIC DISCLOSURE & STANDARDS")
        st.info("""
        * **Simulated Execution**: All quantum calculations are executed on Qiskit Aer simulators or statevector backends without fabricated noise profiles.
        * **Theoretical Scaling**: While Quantum Amplitude Estimation provides asymptotic quadratic speedup $\\mathcal{O}(1/M)$ in oracle queries, current classical vectorized matrix operations remain orders of magnitude faster in wall-clock time for realistic portfolio sizes.
        """)

    # Math explanations
    explain_var()
    explain_cvar()
    explain_amplitude_estimation()


if __name__ == "__main__":
    main()
