"""
Page 2: Risk Analysis - Multi-Method Tail Risk Evaluation
Redesigned into an institutional quantitative-risk analysis research terminal.
"""
import sys
import os
from datetime import datetime

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
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

try:
    st.set_page_config(
        page_title="Quantum Risk Terminal - Risk Analysis",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from app.state import initialize_app_theme
    from app.components.explanations import explain_var, explain_cvar
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme
    from components.explanations import explain_var, explain_cvar

from src.data.loader import load_demo_data, load_user_csv
from src.portfolio.portfolio import Portfolio
from src.classical.historical_simulation import run_historical_simulation
from src.classical.parametric_var import compute_parametric_normal_risk, compute_parametric_student_t_risk
from src.classical.monte_carlo import MonteCarloEngine
from src.utils.reproducibility import set_seed

initialize_app_theme()

# ==============================================================================
# CSS STYLING: INSTITUTIONAL QUANTITATIVE TERMINAL
# ==============================================================================
st.html("""
<style>
    /* Dark Terminal Layout */
    .stApp {
        background-color: #080D14 !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Container resolution */
    .main .block-container {
        max-width: 98% !important;
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Hide default Streamlit sidebar page nav to use custom terminal sidebar */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Sidebar terminal styling */
    [data-testid="stSidebar"] {
        background-color: #0B111A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Top application bar */
    .top-app-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0 16px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }
    .top-app-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-app-logo {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.4);
    }
    .top-app-title {
        font-size: 15px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .top-app-subtitle {
        font-size: 10px;
        font-weight: 600;
        color: #64748B;
        letter-spacing: 0.8px;
    }
    .top-app-search {
        display: flex;
        align-items: center;
        background: #0E1621;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 14px;
        width: 380px;
        gap: 8px;
        color: #64748B;
        font-size: 12px;
    }
    .top-app-kbd {
        margin-left: auto;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 10px;
        font-family: monospace;
        color: #94A3B8;
    }
    .top-app-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .top-app-user {
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 13px;
        color: #F8FAFC;
        font-weight: 500;
    }
    .top-app-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #2563EB;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Terminal Card */
    .terminal-card {
        background-color: #0E1621;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #0E1621;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(59, 130, 246, 0.35);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .kpi-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .kpi-title {
        font-size: 11px;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 11px;
        margin-top: 3px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
    }

    /* Context Strip */
    .context-strip {
        background-color: #0E1621;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 18px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .context-item {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .context-label {
        font-size: 11px;
        color: #94A3B8;
        font-weight: 500;
    }
    .context-value {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 13px;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Clean Terminal Tables */
    .terminal-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .terminal-table th {
        background-color: #111B27;
        color: #60A5FA;
        font-weight: 600;
        font-size: 11px;
        text-align: right;
        padding: 9px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        letter-spacing: 0.3px;
    }
    .terminal-table th:first-child {
        text-align: left;
    }
    .terminal-table th:last-child {
        text-align: left;
    }
    .terminal-table td {
        padding: 9px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        color: #F8FAFC;
        text-align: right;
    }
    .terminal-table td:first-child {
        text-align: left;
        color: #F8FAFC;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .terminal-table td:last-child {
        text-align: left;
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        font-size: 11px;
    }
    .terminal-table tr:hover td {
        background-color: rgba(255, 255, 255, 0.02);
    }

    /* Metric pill boxes */
    .metric-pill {
        border-radius: 6px;
        padding: 4px 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 70px;
    }

    /* Sidebar Navigation Links */
    .terminal-nav-link {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 6px;
        color: #94A3B8;
        text-decoration: none;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.15s ease;
        margin-bottom: 2px;
    }
    .terminal-nav-link:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #F8FAFC;
    }
    .terminal-nav-active {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 0 14px rgba(29, 78, 216, 0.45);
    }

    /* Streamlit input overrides */
    div[data-baseweb="select"] {
        font-size: 12px !important;
        background-color: #111A26 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
    }
    .stSelectbox label {
        font-size: 11px !important;
        color: #94A3B8 !important;
        margin-bottom: 2px !important;
    }
</style>
""")

# ==============================================================================
# STATE & DATA INITIALIZATION
# ==============================================================================
if "global_data_mode" not in st.session_state:
    st.session_state["global_data_mode"] = "Bundled Demo Data"
if "portfolio_val" not in st.session_state:
    st.session_state["portfolio_val"] = 1_000_000.0
if "conf_level" not in st.session_state:
    st.session_state["conf_level"] = 0.95
if "time_horizon" not in st.session_state:
    st.session_state["time_horizon"] = 10
if "mc_sims" not in st.session_state:
    st.session_state["mc_sims"] = 20_000
if "seed" not in st.session_state:
    st.session_state["seed"] = 42

if "prices_df" not in st.session_state:
    st.session_state["prices_df"] = load_demo_data()

prices_df = st.session_state["prices_df"]
if "portfolio" not in st.session_state or st.session_state.get("_last_prices_shape") != prices_df.shape:
    st.session_state["portfolio"] = Portfolio(prices_df, initial_value=st.session_state["portfolio_val"])
    st.session_state["_last_prices_shape"] = prices_df.shape

portfolio = st.session_state["portfolio"]

# ==============================================================================
# LEFT SIDEBAR: QUANTITATIVE TERMINAL NAVIGATION
# ==============================================================================
with st.sidebar:
    st.html("""
    <div style='display:flex; align-items:center; gap:10px; padding: 6px 0 14px 0;'>
        <div style='width:28px; height:28px; border-radius:6px; background:linear-gradient(135deg, #2563EB, #7C3AED); display:flex; align-items:center; justify-content:center; box-shadow: 0 0 10px rgba(37,99,235,0.4);'>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M12 3v3m0 12v3M3 12h3m12 0h3"></path>
                <path d="M5.636 5.636l2.122 2.122m8.484 8.484l2.122 2.122M5.636 18.364l2.122-2.122m8.484-8.484l2.122-2.122"></path>
            </svg>
        </div>
        <div>
            <div style='font-size:14px; font-weight:700; color:#F8FAFC; letter-spacing:-0.2px;'>Quantum Risk Terminal</div>
            <div style='font-size:9px; font-weight:600; color:#64748B; letter-spacing:0.8px;'>RISK • SIMULATION • PORTFOLIO</div>
        </div>
    </div>
    """)

    # Main Navigation Links
    st.html("""
    <div style='margin-bottom: 8px;'>
        <a href='/' target='_self' class='terminal-nav-link'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="9"></rect><rect x="14" y="3" width="7" height="5"></rect><rect x="14" y="12" width="7" height="9"></rect><rect x="3" y="16" width="7" height="5"></rect></svg>
            Dashboard
        </a>
        <a href='/Portfolio' target='_self' class='terminal-nav-link'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>
            Portfolio
        </a>
        
        <div style='display:flex; justify-content:space-between; align-items:center; padding: 7px 12px; margin-top: 4px; font-size: 13px; font-weight: 600; color: #F8FAFC;'>
            <span style='display:flex; align-items:center; gap:9px;'>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                Risk Analysis
            </span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
        </div>
        
        <div style='padding-left: 14px; margin-bottom: 2px;'>
            <a href='/Risk_Analysis' target='_self' class='terminal-nav-link terminal-nav-active'>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <circle cx="12" cy="12" r="4" fill="#FFFFFF"></circle>
                </svg>
                Risk Analysis
            </a>
            <a href='/Classical_Engine' target='_self' class='terminal-nav-link'>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                Classical Engine
            </a>
            <a href='/Quantum_Engine' target='_self' class='terminal-nav-link'>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"></circle><path d="M12 2a10 10 0 0 0-10 10c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"></path></svg>
                Quantum Engine
            </a>
        </div>
        
        <a href='/Benchmark_Lab' target='_self' class='terminal-nav-link'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 18h12"></path><path d="M10 22h4"></path><path d="M14 2v4"></path><path d="M10 2v4"></path><path d="M9 6h6"></path><path d="M12 6v12"></path></svg>
            Benchmark Lab
        </a>
        <a href='/Stress_Testing' target='_self' class='terminal-nav-link'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            Stress Testing
        </a>
        <a href='/Research_Report' target='_self' class='terminal-nav-link'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            Research Report
        </a>
    </div>
    """)

    st.html("<div style='border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0 10px 0;'></div>")
    st.html("<div style='font-size:10px; font-weight:700; color:#64748B; letter-spacing:0.8px; margin-bottom:8px;'>DATA & SETTINGS</div>")

    # Data Source Section
    with st.expander("⛁ Data Source", expanded=False):
        data_mode = st.radio(
            "Source Mode",
            ["Bundled Demo Data", "Upload Custom CSV"],
            key="p2_data_mode",
            label_visibility="collapsed"
        )
        if data_mode == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="p2_csv_uploader")
            if uploaded_file is not None:
                try:
                    loaded_df = load_user_csv(uploaded_file)
                    st.session_state["prices_df"] = loaded_df
                    st.session_state["portfolio"] = Portfolio(loaded_df, initial_value=st.session_state["portfolio_val"])
                    st.success(f"✓ Loaded {len(loaded_df.columns)} assets")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            if st.session_state.get("global_data_mode") != "Bundled Demo Data":
                st.session_state["prices_df"] = load_demo_data()
                st.session_state["portfolio"] = Portfolio(st.session_state["prices_df"], initial_value=st.session_state["portfolio_val"])

    # Model Parameters Section
    with st.expander("⚙ Model Parameters", expanded=False):
        new_port_val = st.number_input(
            "Initial Portfolio Value (₹)",
            min_value=10_000.0,
            max_value=100_000_000.0,
            value=float(st.session_state["portfolio_val"]),
            step=50_000.0,
            key="p2_port_val"
        )
        if new_port_val != st.session_state["portfolio_val"]:
            st.session_state["portfolio_val"] = new_port_val
            portfolio.initial_value = new_port_val

        new_conf = st.selectbox(
            "Confidence Level (α)",
            [0.90, 0.95, 0.99, 0.995],
            index=[0.90, 0.95, 0.99, 0.995].index(st.session_state["conf_level"]) if st.session_state["conf_level"] in [0.90, 0.95, 0.99, 0.995] else 1,
            format_func=lambda x: f"{int(x*100) if x in [0.90, 0.95, 0.99] else 99.5}%",
            key="p2_conf_level"
        )
        st.session_state["conf_level"] = new_conf

        new_horizon = st.slider("Time Horizon (Days)", min_value=1, max_value=60, value=int(st.session_state["time_horizon"]), key="p2_time_horizon")
        st.session_state["time_horizon"] = new_horizon

    # Simulation Settings Section
    with st.expander("🎛 Simulation Settings", expanded=False):
        new_sims = st.selectbox("Monte Carlo Paths", [1_000, 5_000, 10_000, 20_000, 50_000, 100_000], index=3, key="p2_mc_sims")
        st.session_state["mc_sims"] = new_sims
        new_seed = st.number_input("Deterministic Seed", min_value=1, max_value=999999, value=int(st.session_state["seed"]), step=1, key="p2_seed")
        st.session_state["seed"] = new_seed

    st.html("""
    <div style='background: #0E1621; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 14px; margin-top: 20px;'>
        <div style='font-size: 12px; font-weight: 700; color: #F8FAFC; margin-bottom: 2px;'>QUANTUM RISK ENGINE</div>
        <div style='font-size: 10px; color: #64748B; font-family: monospace;'>v2.4 • RESEARCH TERMINAL</div>
    </div>
    """)

# Active session parameters
port_val = float(st.session_state["portfolio_val"])
conf_level = float(st.session_state["conf_level"])
time_horizon = int(st.session_state["time_horizon"])
mc_sims = int(st.session_state["mc_sims"])
seed = int(st.session_state["seed"])
set_seed(seed)

# ==============================================================================
# COMPUTATIONS: RISK METHODOLOGIES (CACHED WHERE POSSIBLE)
# ==============================================================================
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

norm_90 = compute_parametric_normal_risk(daily_mu, daily_vol, confidence_level=0.90, time_horizon_days=time_horizon, portfolio_value=port_val)
norm_95 = compute_parametric_normal_risk(daily_mu, daily_vol, confidence_level=0.95, time_horizon_days=time_horizon, portfolio_value=port_val)
norm_99 = compute_parametric_normal_risk(daily_mu, daily_vol, confidence_level=0.99, time_horizon_days=time_horizon, portfolio_value=port_val)

student_90 = compute_parametric_student_t_risk(portfolio.portfolio_returns.values, confidence_level=0.90, time_horizon_days=time_horizon, portfolio_value=port_val)
student_95 = compute_parametric_student_t_risk(portfolio.portfolio_returns.values, confidence_level=0.95, time_horizon_days=time_horizon, portfolio_value=port_val)
student_99 = compute_parametric_student_t_risk(portfolio.portfolio_returns.values, confidence_level=0.99, time_horizon_days=time_horizon, portfolio_value=port_val)

# 3. Monte Carlo Simulation Engine
mc_cache_key = f"mc_res_{port_val}_{time_horizon}_{mc_sims}_{seed}_{prices_df.shape}"
if mc_cache_key not in st.session_state:
    mc_engine = MonteCarloEngine(
        initial_portfolio_value=port_val,
        weights=portfolio.weights,
        expected_returns=portfolio.returns.mean().values * 252.0,
        covariance_matrix=portfolio.covariance_matrix,
        asset_prices=portfolio.prices.iloc[-1].values,
        seed=seed
    )
    st.session_state[mc_cache_key] = mc_engine.run_simulation(
        n_simulations=mc_sims,
        time_horizon_days=time_horizon,
        confidence_levels=[0.90, 0.95, 0.99],
        method="gbm"
    )

mc_res = st.session_state[mc_cache_key]

# Reference metrics at chosen confidence level
alpha_key = f"{int(conf_level * 100)}%" if conf_level in [0.90, 0.95, 0.99] else "95%"
hist_m = hist_res["results"].get(alpha_key, hist_res["results"]["95%"])
norm_m = norm_95 if conf_level == 0.95 else (norm_90 if conf_level == 0.90 else norm_99)
student_m = student_95 if conf_level == 0.95 else (student_90 if conf_level == 0.90 else student_99)
mc_m = mc_res["risk_metrics"].get(alpha_key, mc_res["risk_metrics"]["95%"])

# ==============================================================================
# 1. TOP APPLICATION BAR
# ==============================================================================
st.html("""
<div class='top-app-bar'>
    <div class='top-app-left'>
        <div class='top-app-logo'>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 3v18h18"></path>
                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path>
            </svg>
        </div>
        <div>
            <div class='top-app-title'>Quantum Risk Terminal</div>
            <div class='top-app-subtitle'>RISK • SIMULATION • PORTFOLIO</div>
        </div>
    </div>
    <div class='top-app-search'>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        <span>Search assets, simulations, or reports...</span>
        <span class='top-app-kbd'>⌘K</span>
    </div>
    <div class='top-app-right'>
        <div style='display:flex; align-items:center; gap:6px; background:#0E1621; border:1px solid rgba(255,255,255,0.08); border-radius:20px; padding:3px 8px; cursor:pointer;'>
            <span style='font-size:12px;'>☀️</span>
            <span style='font-size:12px; opacity:0.4;'>🌙</span>
        </div>
        <div class='top-app-user'>
            <div class='top-app-avatar'>MF</div>
            <span>Michael Fernandes</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
        </div>
    </div>
</div>
""")

# ==============================================================================
# 2. BREADCRUMB & HEADER BAR
# ==============================================================================
now_str = datetime.now().strftime("%b %d, %Y %I:%M %p")

h_col_left, h_col_right = st.columns([7.8, 4.2])
with h_col_left:
    st.html("""
    <div style='font-size:12px; color:#94A3B8; font-weight:500; margin-bottom:4px;'>
        Risk Analysis &nbsp;&gt;&nbsp; <span style='color:#F8FAFC;'>Risk Analysis</span>
    </div>
    <h1 style='margin:0; font-size:22px; font-weight:700; color:#F8FAFC; letter-spacing:-0.4px;'>
        02. RISK ANALYSIS
    </h1>
    <div style='font-size:12px; color:#94A3B8; margin-top:2px; margin-bottom:12px;'>
        Multi-Method Tail Risk Evaluation: Historical, Parametric, and Empirical Monte Carlo
    </div>
    """)

with h_col_right:
    st.html(f"""
    <div style='display:flex; justify-content:flex-end; align-items:center; gap:12px; margin-bottom:8px;'>
        <span style='font-size:11px; color:#94A3B8; font-family:monospace;'>Last updated: {now_str}</span>
    </div>
    """)
    act_c1, act_c2 = st.columns([1, 1])
    with act_c1:
        if st.button("🔄 Refresh", key="btn_refresh_p2", width="stretch"):
            for k in list(st.session_state.keys()):
                if k.startswith("mc_res_"):
                    del st.session_state[k]
            st.rerun()
    with act_c2:
        st.button("📥 Export Report", key="btn_export_p2", width="stretch")

# ==============================================================================
# 3. RISK OVERVIEW KPI SECTION (4 TILES)
# ==============================================================================
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4, gap="small")

with kpi_c1:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(59, 130, 246, 0.15); color:#3B82F6;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Historical VaR ({alpha_key})</div>
            <div class='kpi-value'>-₹{hist_m['var_cash']:,.0f}</div>
            <div class='kpi-sub'>
                <span style='color:#EF4444; font-weight:600;'>{hist_m['var_pct']*100:.2f}%</span>
                <span style='color:#94A3B8;'>Historical / Empirical</span>
            </div>
        </div>
    </div>
    """)

with kpi_c2:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(99, 102, 241, 0.15); color:#6366F1;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Gaussian Parametric VaR</div>
            <div class='kpi-value'>-₹{norm_m['var_cash']:,.0f}</div>
            <div class='kpi-sub'>
                <span style='color:#EF4444; font-weight:600;'>{norm_m['var_pct']*100:.2f}%</span>
                <span style='color:#94A3B8;'>Normal Distribution</span>
            </div>
        </div>
    </div>
    """)

with kpi_c3:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(225, 29, 72, 0.15); color:#E11D48;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Student-t VaR (df={student_m['degrees_of_freedom']:.1f})</div>
            <div class='kpi-value'>-₹{student_m['var_cash']:,.0f}</div>
            <div class='kpi-sub'>
                <span style='color:#EF4444; font-weight:600;'>{student_m['var_pct']*100:.2f}%</span>
                <span style='color:#94A3B8;'>df = {student_m['degrees_of_freedom']:.1f}</span>
            </div>
        </div>
    </div>
    """)

with kpi_c4:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(245, 158, 11, 0.15); color:#F59E0B;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                <line x1="12" y1="22.08" x2="12" y2="12"></line>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Monte Carlo VaR ({mc_sims:,})</div>
            <div class='kpi-value'>-₹{mc_m['var_cash']:,.0f}</div>
            <div class='kpi-sub'>
                <span style='color:#EF4444; font-weight:600;'>{mc_m['var_pct']*100:.2f}%</span>
                <span style='color:#94A3B8;'>Vectorized GBM</span>
            </div>
        </div>
    </div>
    """)

# ==============================================================================
# 4. CONTEXTUAL RISK STRIP
# ==============================================================================
st.html(f"""
<div class='context-strip'>
    <div class='context-item'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2DD4BF" stroke-width="2.2"><path d="M3 3v18h18"></path><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path></svg>
        <div>
            <div class='context-label'>Portfolio Value</div>
            <div class='context-value'>₹{port_val:,.0f}</div>
        </div>
    </div>
    <div class='context-item'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2.2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
        <div>
            <div class='context-label'>Confidence Level</div>
            <div class='context-value'>{int(conf_level*100) if conf_level in [0.9, 0.95, 0.99] else conf_level*100}%</div>
        </div>
    </div>
    <div class='context-item'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
        <div>
            <div class='context-label'>Time Horizon</div>
            <div class='context-value'>{time_horizon} Days</div>
        </div>
    </div>
    <div class='context-item'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06B6D4" stroke-width="2.2"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>
        <div>
            <div class='context-label'>Simulation Paths</div>
            <div class='context-value'>{mc_sims:,}</div>
        </div>
    </div>
    <div class='context-item'>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        <div>
            <div class='context-label'>Primary Model</div>
            <div class='context-value'>Vectorized GBM</div>
        </div>
    </div>
</div>
""")

# ==============================================================================
# 5 & 6. CROSS-METHODOLOGY COMPARISON & BAR CHART VISUAL
# ==============================================================================
mid_left, mid_right = st.columns([6.4, 3.6], gap="medium")

with mid_left:
    st.html(f"""
    <div class='terminal-card' style='height: 295px; display:flex; flex-direction:column; justify-content:space-between;'>
        <div>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom: 3px;'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                <span style='font-size:14px; font-weight:700; color:#F8FAFC;'>Cross-Methodology Risk Comparison</span>
            </div>
            <div style='font-size:11px; color:#94A3B8; margin-bottom: 12px;'>
                Comparison of VaR and CVaR estimates across statistical and simulation-based methodologies.
            </div>
            <div style='overflow-x: auto;'>
                <table class='terminal-table'>
                    <thead>
                        <tr>
                            <th>Methodology</th>
                            <th>VaR 90% (₹)</th>
                            <th>VaR 95% (₹)</th>
                            <th>VaR 99% (₹)</th>
                            <th>CVaR 95% (₹)</th>
                            <th>Assumption</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Historical Simulation (Empirical)</td>
                            <td>-₹{hist_res['results']['90%']['var_cash']:,.0f}</td>
                            <td>-₹{hist_res['results']['95%']['var_cash']:,.0f}</td>
                            <td>-₹{hist_res['results']['99%']['var_cash']:,.0f}</td>
                            <td>-₹{hist_res['results']['95%']['cvar_cash']:,.0f}</td>
                            <td>No distribution assumed (past repeats)</td>
                        </tr>
                        <tr>
                            <td>Parametric Gaussian (Normal)</td>
                            <td>-₹{norm_90['var_cash']:,.0f}</td>
                            <td>-₹{norm_95['var_cash']:,.0f}</td>
                            <td>-₹{norm_99['var_cash']:,.0f}</td>
                            <td>-₹{norm_95['cvar_cash']:,.0f}</td>
                            <td>Strictly Gaussian i.i.d. returns</td>
                        </tr>
                        <tr>
                            <td>Parametric Student-t (Fat Tails)</td>
                            <td>-₹{student_90['var_cash']:,.0f}</td>
                            <td>-₹{student_95['var_cash']:,.0f}</td>
                            <td>-₹{student_99['var_cash']:,.0f}</td>
                            <td>-₹{student_95['cvar_cash']:,.0f}</td>
                            <td>Fitted leptokurtic Student-t</td>
                        </tr>
                        <tr>
                            <td>Classical Monte Carlo (Vectorized GBM)</td>
                            <td>-₹{mc_res['risk_metrics']['90%']['var_cash']:,.0f}</td>
                            <td>-₹{mc_res['risk_metrics']['95%']['var_cash']:,.0f}</td>
                            <td>-₹{mc_res['risk_metrics']['99%']['var_cash']:,.0f}</td>
                            <td>-₹{mc_res['risk_metrics']['95%']['cvar_cash']:,.0f}</td>
                            <td>Multivariate correlated diffusion</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """)

with mid_right:
    # Header with alpha selector dropdown
    top_c1, top_c2 = st.columns([6, 4])
    with top_c1:
        st.html("""
        <div style='display:flex; align-items:center; gap:8px; margin-top: 4px; margin-bottom: 2px;'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
            <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>VaR Comparison Across Methodologies</span>
        </div>
        """)
    with top_c2:
        chart_conf_choice = st.selectbox(
            "Target VaR",
            ["VaR 90%", "VaR 95%", "VaR 99%"],
            index=1,
            key="p2_var_chart_conf",
            label_visibility="collapsed"
        )

    # Determine values for the bar chart
    if chart_conf_choice == "VaR 90%":
        b_hist = -hist_res['results']['90%']['var_cash']
        b_norm = -norm_90['var_cash']
        b_stud = -student_90['var_cash']
        b_mc = -mc_res['risk_metrics']['90%']['var_cash']
    elif chart_conf_choice == "VaR 99%":
        b_hist = -hist_res['results']['99%']['var_cash']
        b_norm = -norm_99['var_cash']
        b_stud = -student_99['var_cash']
        b_mc = -mc_res['risk_metrics']['99%']['var_cash']
    else:
        b_hist = -hist_res['results']['95%']['var_cash']
        b_norm = -norm_95['var_cash']
        b_stud = -student_95['var_cash']
        b_mc = -mc_res['risk_metrics']['95%']['var_cash']

    bar_labels = ['Historical', 'Gaussian', 'Student-t', 'Monte Carlo']
    bar_values = [b_hist, b_norm, b_stud, b_mc]
    bar_colors = ['#2563EB', '#7C3AED', '#E11D48', '#F59E0B']

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=bar_labels,
        x=bar_values,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.08)', width=1)),
        text=[f'-₹{-v:,.0f}' for v in bar_values],
        textposition='outside',
        textfont=dict(color='#F8FAFC', family='JetBrains Mono, monospace', size=11)
    ))
    fig_bar.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1621',
        plot_bgcolor='#0E1621',
        margin=dict(l=80, r=65, t=10, b=30),
        xaxis=dict(
            title=dict(text='VaR (₹)', font=dict(size=11, color='#94A3B8')),
            gridcolor='rgba(255, 255, 255, 0.05)',
            tickformat='~s',
            tickfont=dict(size=10, family='JetBrains Mono, monospace')
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(size=11, color='#CBD5E1', family='Inter, sans-serif')
        ),
        height=225,
        showlegend=False
    )
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

# ==============================================================================
# 7 & 8. INTERACTIVE LOSS DISTRIBUTION & TAIL LOSS SCENARIOS
# ==============================================================================
bot_left, bot_right = st.columns([6.4, 3.6], gap="medium")

with bot_left:
    st.html("""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span style='font-size:14px; font-weight:700; color:#F8FAFC;'>Interactive Loss Distribution & CVaR Tail Shading</span>
        </div>
    </div>
    """)

    # Compact Control Row with metric overlay pills
    c_row1, c_row2, c_row3, c_row4, c_row5 = st.columns([1.8, 1.8, 1.8, 2.2, 3.8], gap="small")
    with c_row1:
        dist_conf = st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1, key="p2_dist_conf")
    with c_row2:
        dist_paths = st.selectbox("Simulation Paths", ["10,000", "20,000", "50,000"], index=1, key="p2_dist_paths")
    with c_row3:
        dist_horizon = st.selectbox("Time Horizon", ["5 Days", "10 Days", "21 Days"], index=1, key="p2_dist_horizon")
    with c_row4:
        dist_model = st.selectbox("Model", ["GBM (Vectorized)", "MVN (Normal)"], index=0, key="p2_dist_model")
    with c_row5:
        # 3 metric pill boxes
        st.html(f"""
        <div style='display:flex; gap:8px; justify-content:flex-end; align-items:center; height:100%; padding-top:14px;'>
            <div class='metric-pill' style='background:rgba(245, 158, 11, 0.15); border:1px solid rgba(245, 158, 11, 0.4);'>
                <div style='font-size:9px; color:#F59E0B; font-weight:600;'>VaR {dist_conf}</div>
                <div style='font-size:12px; color:#F59E0B; font-weight:700; font-family:monospace;'>{mc_res['risk_metrics'].get(dist_conf, mc_m)['var_pct']*100:.2f}%</div>
            </div>
            <div class='metric-pill' style='background:rgba(239, 68, 68, 0.15); border:1px solid rgba(239, 68, 68, 0.4);'>
                <div style='font-size:9px; color:#EF4444; font-weight:600;'>CVaR {dist_conf}</div>
                <div style='font-size:12px; color:#EF4444; font-weight:700; font-family:monospace;'>{mc_res['risk_metrics'].get(dist_conf, mc_m)['cvar_pct']*100:.2f}%</div>
            </div>
            <div class='metric-pill' style='background:rgba(59, 130, 246, 0.15); border:1px solid rgba(59, 130, 246, 0.4);'>
                <div style='font-size:9px; color:#60A5FA; font-weight:600;'>Tail Probability</div>
                <div style='font-size:12px; color:#60A5FA; font-weight:700; font-family:monospace;'>{100 - int(dist_conf.replace('%', ''))}%</div>
            </div>
        </div>
        """)

    # Interactive Loss Distribution Plot
    active_dist_m = mc_res['risk_metrics'].get(dist_conf, mc_m)
    active_var_pct = active_dist_m['var_pct']
    active_cvar_pct = active_dist_m['cvar_pct']
    active_alpha_num = int(dist_conf.replace('%', '')) / 100.0

    counts, bin_edges = np.histogram(mc_res["losses_pct"] * 100.0, bins=65, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    tail_mask = bin_centers >= (active_var_pct * 100.0)
    body_mask = ~tail_mask

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Bar(
        x=bin_centers[body_mask],
        y=counts[body_mask],
        width=bin_width,
        marker=dict(color='#2563EB', opacity=0.85, line=dict(color='#1D4ED8', width=0.5)),
        name='Loss Distribution'
    ))
    fig_loss.add_trace(go.Bar(
        x=bin_centers[tail_mask],
        y=counts[tail_mask],
        width=bin_width,
        marker=dict(color='#EF4444', opacity=0.85, line=dict(color='#DC2626', width=0.5)),
        name=f'Tail Loss (Top {int((1-active_alpha_num)*100)}%)'
    ))
    fig_loss.add_vline(
        x=active_var_pct * 100.0,
        line_dash='dash',
        line_color='#F59E0B',
        line_width=2,
        annotation_text=f'VaR {dist_conf} | {active_var_pct*100:.2f}%',
        annotation_position='top left',
        annotation_font=dict(color='#F59E0B', size=11, family='JetBrains Mono, monospace')
    )
    fig_loss.add_vline(
        x=active_cvar_pct * 100.0,
        line_dash='dash',
        line_color='#EF4444',
        line_width=2,
        annotation_text=f'CVaR {dist_conf} | {active_cvar_pct*100:.2f}%',
        annotation_position='top right',
        annotation_font=dict(color='#EF4444', size=11, family='JetBrains Mono, monospace')
    )
    fig_loss.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1621',
        plot_bgcolor='#0E1621',
        margin=dict(l=45, r=20, t=25, b=40),
        xaxis=dict(
            title=dict(text='Loss (%)', font=dict(size=11, color='#94A3B8')),
            gridcolor='rgba(255, 255, 255, 0.05)',
            tickfont=dict(size=10, family='JetBrains Mono, monospace')
        ),
        yaxis=dict(
            title=dict(text='Probability Density', font=dict(size=11, color='#94A3B8')),
            gridcolor='rgba(255, 255, 255, 0.05)',
            tickfont=dict(size=10, family='JetBrains Mono, monospace')
        ),
        height=330,
        legend=dict(
            orientation='v',
            x=0.98,
            xanchor='right',
            y=0.95,
            bgcolor='rgba(14, 22, 33, 0.7)',
            bordercolor='rgba(255,255,255,0.08)',
            borderwidth=1,
            font=dict(size=10, color='#94A3B8')
        )
    )
    st.plotly_chart(fig_loss, width="stretch", config={"displayModeBar": False})

with bot_right:
    # Header with View All toggle button
    t_top1, t_top2 = st.columns([7, 3])
    with t_top1:
        st.html("""
        <div style='display:flex; align-items:center; gap:8px; margin-top:2px;'>
            <span style='color:#EF4444;'>⚠️</span>
            <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>Worst-Case Tail Loss Scenarios</span>
        </div>
        """)
    with t_top2:
        view_all = st.button("View All", key="p2_view_all_tail", width="stretch")

    # Tail Observations Table
    tail_obs_mask = mc_res["losses_pct"] >= mc_m["var_pct"]
    tail_losses = mc_res["losses_pct"][tail_obs_mask]
    tail_sorted = np.sort(tail_losses)[::-1]
    n_display = 10 if view_all else 5
    tail_display = tail_sorted[:n_display]

    tail_rows_html = ""
    for idx, l_val in enumerate(tail_display):
        tail_rows_html += f"""
        <tr>
            <td style='text-align:left; color:#94A3B8; font-weight:600;'>{idx+1}</td>
            <td>{l_val*100:.2f}%</td>
            <td>-₹{l_val*port_val:,.0f}</td>
            <td style='color:#EF4444; font-size:11px;'>Severe Tail Event (&gt; VaR)</td>
        </tr>
        """

    st.html(f"""
    <div style='background:#0E1621; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:10px 14px; margin-bottom:12px;'>
        <table class='terminal-table'>
            <thead>
                <tr>
                    <th style='text-align:left;'>Rank</th>
                    <th>Simulated Loss (%)</th>
                    <th>Simulated Loss (₹)</th>
                    <th style='text-align:left;'>Tail Classification</th>
                </tr>
            </thead>
            <tbody>
                {tail_rows_html}
            </tbody>
        </table>
    </div>
    """)

    # Tail Risk Summary & Interpretation row
    sub_c1, sub_c2 = st.columns([1, 1], gap="small")
    with sub_c1:
        tail_spread = mc_m['cvar_cash'] - mc_m['var_cash']
        ratio = mc_m['cvar_cash'] / max(mc_m['var_cash'], 1e-6)
        st.html(f"""
        <div style='background:#0E1621; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
            <div style='display:flex; align-items:center; gap:6px; font-size:12px; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                Tail Risk Summary
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px; font-size:11px;'>
                <span style='color:#94A3B8;'>VaR 95%</span>
                <span style='color:#F59E0B; font-weight:700; font-family:monospace;'>-₹{mc_m['var_cash']:,.0f}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px; font-size:11px;'>
                <span style='color:#94A3B8;'>CVaR 95%</span>
                <span style='color:#EF4444; font-weight:700; font-family:monospace;'>-₹{mc_m['cvar_cash']:,.0f}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px; font-size:11px;'>
                <span style='color:#94A3B8;'>Tail Spread</span>
                <span style='color:#22C55E; font-weight:700; font-family:monospace;'>₹{tail_spread:,.0f}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px; font-size:11px;'>
                <span style='color:#94A3B8;'>Tail / VaR Ratio</span>
                <span style='color:#F8FAFC; font-weight:700; font-family:monospace;'>{ratio:.2f}x</span>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:11px;'>
                <span style='color:#94A3B8;'>Tail Probability</span>
                <span style='color:#60A5FA; font-weight:700; font-family:monospace;'>5%</span>
            </div>
        </div>
        """)

    with sub_c2:
        dispersion_pct = abs(hist_m['var_cash'] - mc_m['var_cash']) / max(mc_m['var_cash'], 1e-6) * 100.0
        st.html(f"""
        <div style='background:#0E1621; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
            <div style='display:flex; align-items:center; gap:6px; font-size:12px; font-weight:700; color:#F8FAFC; margin-bottom:6px;'>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                Risk Interpretation
            </div>
            <div style='font-size:10px; color:#3B82F6; font-weight:700; margin-bottom:1px;'>PRIMARY RISK SIGNAL</div>
            <div style='font-size:10px; color:#94A3B8; line-height:1.3; margin-bottom:6px;'>
                Monte Carlo VaR estimates a 95% potential loss of approximately ₹{mc_m['var_cash']:,.0f} over the selected horizon.
            </div>
            <div style='font-size:10px; color:#F59E0B; font-weight:700; margin-bottom:1px;'>TAIL RISK</div>
            <div style='font-size:10px; color:#94A3B8; line-height:1.3; margin-bottom:6px;'>
                CVaR estimates the average loss within the worst 5% of simulated outcomes at approximately ₹{mc_m['cvar_cash']:,.0f}.
            </div>
            <div style='font-size:10px; color:#A855F7; font-weight:700; margin-bottom:1px;'>METHODOLOGY DISPERSION</div>
            <div style='font-size:10px; color:#94A3B8; line-height:1.3;'>
                Different methodologies produce varying risk estimates, with a dispersion of ~{dispersion_pct:.1f}% between highest and lowest 95% VaR estimates.
            </div>
        </div>
        """)

# ==============================================================================
# 9. EXPLANATIONS (PRESERVED)
# ==============================================================================
st.html("<div style='height: 8px;'></div>")
explain_var()
explain_cvar()
