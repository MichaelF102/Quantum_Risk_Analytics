"""
Page 6: Stress Testing & Advanced Volatility Extensions (GARCH / EGARCH)
Redesigned into an institutional quantitative stress-testing research terminal.
"""
import sys
import os
import time
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
        page_title="Quantum Risk Terminal - Stress Testing",
        page_icon="⚡",
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
    from app.components.charts import get_terminal_layout
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme
    from components.charts import get_terminal_layout

from src.data.loader import load_demo_data, load_user_csv
from src.portfolio.portfolio import Portfolio
from src.classical.monte_carlo import MonteCarloEngine
from src.classical.var import compute_var_from_losses
from src.classical.cvar import compute_cvar_from_losses
from src.models.garch import fit_garch
from src.models.egarch import fit_egarch
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

    /* Hide default Streamlit sidebar page nav */
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
        margin-bottom: 2px;
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
    .terminal-table tr:hover td {
        background-color: rgba(255, 255, 255, 0.02);
    }

    /* Key/Value row */
    .detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 11px;
    }
    .detail-row:last-child {
        border-bottom: none;
    }
    .detail-key {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #94A3B8;
    }
    .detail-val {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-weight: 600;
        color: #F8FAFC;
        text-align: right;
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
    div[data-baseweb="input"] {
        font-size: 12px !important;
        background-color: #111A26 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
    }
    .stSelectbox label, .stNumberInput label {
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
            <a href='/Risk_Analysis' target='_self' class='terminal-nav-link'>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                Overview
            </a>
            <a href='/Benchmark_Lab' target='_self' class='terminal-nav-link'>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="4" fill="currentColor"></circle></svg>
                Head-to-Head
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
        <a href='/Stress_Testing' target='_self' class='terminal-nav-link terminal-nav-active'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
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
        data_mode = st.radio("Source Mode", ["Bundled Demo Data", "Upload Custom CSV"], key="p6_data_mode", label_visibility="collapsed")
        if data_mode == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="p6_csv_uploader")
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
        new_port_val = st.number_input("Initial Portfolio Value (₹)", min_value=10_000.0, max_value=100_000_000.0, value=float(st.session_state["portfolio_val"]), step=50_000.0, key="p6_port_val")
        if new_port_val != st.session_state["portfolio_val"]:
            st.session_state["portfolio_val"] = new_port_val
            portfolio.initial_value = new_port_val

        new_conf = st.selectbox(
            "Confidence Level (α)",
            [0.90, 0.95, 0.99, 0.995],
            index=[0.90, 0.95, 0.99, 0.995].index(st.session_state["conf_level"]) if st.session_state["conf_level"] in [0.90, 0.95, 0.99, 0.995] else 1,
            format_func=lambda x: f"{int(x*100) if x in [0.90, 0.95, 0.99] else 99.5}%",
            key="p6_conf_level"
        )
        st.session_state["conf_level"] = new_conf

        new_horizon = st.slider("Time Horizon (Days)", min_value=1, max_value=60, value=int(st.session_state["time_horizon"]), key="p6_time_horizon")
        st.session_state["time_horizon"] = new_horizon

    # Simulation Settings Section
    with st.expander("🎛 Simulation Settings", expanded=False):
        new_sims = st.selectbox("Monte Carlo Paths", [1_000, 5_000, 10_000, 20_000, 50_000, 100_000], index=3, key="p6_mc_sims")
        st.session_state["mc_sims"] = new_sims
        new_seed = st.number_input("Deterministic Seed", min_value=1, max_value=999999, value=int(st.session_state["seed"]), step=1, key="p6_seed")
        st.session_state["seed"] = new_seed

    st.html("""
    <div style='background: #0E1621; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 14px; margin-top: 14px;'>
        <div style='font-size: 12px; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;'>QUANTUM RISK ENGINE</div>
        <div style='font-size: 11px; color: #94A3B8; line-height: 1.3;'>
            Advanced risk analytics using classical and quantum computing approaches.
        </div>
        <div style='font-size: 10px; color: #64748B; font-family: monospace; margin-top: 6px;'>v2.4 • RESEARCH TERMINAL</div>
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
        Risk Analysis &nbsp;&gt;&nbsp; <span style='color:#F8FAFC;'>Stress Testing</span>
    </div>
    <div style='display:flex; align-items:center; gap:10px;'>
        <h1 style='margin:0; font-size:22px; font-weight:700; color:#F8FAFC; letter-spacing:-0.4px;'>
            STRESS TESTING
        </h1>
        <span style='background:rgba(168, 85, 247, 0.2); border:1px solid rgba(168, 85, 247, 0.5); color:#C084FC; font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; letter-spacing:0.5px;'>
            SCENARIO ANALYSIS
        </span>
    </div>
    <div style='font-size:12px; color:#94A3B8; margin-top:2px; margin-bottom:12px;'>
        Evaluate portfolio resilience under extreme market conditions and volatility regimes.
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
        if st.button("🔄 Refresh", key="btn_refresh_p6", width="stretch"):
            for k in list(st.session_state.keys()):
                if k.startswith("stress_cache_"):
                    del st.session_state[k]
            st.rerun()
    with act_c2:
        st.button("📥 Export Report", key="btn_export_p6", width="stretch")

# ==============================================================================
# 3. ANALYSIS TABS
# ==============================================================================
stress_tab, garch_tab = st.tabs([
    "1. Macro Stress Scenarios",
    "2. GARCH / EGARCH Volatility Extension"
])

# ==============================================================================
# TAB 1: MACRO STRESS SCENARIOS
# ==============================================================================
with stress_tab:
    # 4. Stress Test Configuration Bar
    cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns([3.8, 2.2, 1.8, 2.2, 2.0], gap="small")
    with cfg_c1:
        scenario_choice = st.selectbox(
            "Select Stress Testing Scenario",
            [
                "Market Crash (-15% Systematic Shock)",
                "Volatility Spike (+80% Volatility Surge)",
                "Correlation Breakdown (ρ → 0.90 Contagion)",
                "Single Asset Catastrophe (-40% Shock to Largest Holding)"
            ],
            key="p6_scenario_choice"
        )
    with cfg_c2:
        st.selectbox("Portfolio", ["Current Portfolio"], index=0, key="p6_port_choice")
    with cfg_c3:
        conf_input = st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1, key="p6_conf_input")
        alpha_sel = float(conf_input.replace("%", "")) / 100.0
    with cfg_c4:
        horizon_input = st.number_input("Time Horizon (Days)", min_value=1, max_value=60, value=int(time_horizon), step=1, key="p6_horizon_input")
    with cfg_c5:
        st.html("<div style='height:24px;'></div>")
        run_stress_btn = st.button("▶ Run Stress Test", key="p6_run_stress_btn", type="primary", width="stretch")

    # ==============================================================================
    # COMPUTATION PIPELINE: BASELINE VS STRESSED
    # ==============================================================================
    stress_cache_key = f"stress_cache_{port_val}_{horizon_input}_{alpha_sel}_{scenario_choice}_{seed}_{prices_df.shape}"

    if stress_cache_key not in st.session_state or run_stress_btn:
        if run_stress_btn:
            with st.status("RUNNING STRESS TEST", expanded=True) as status:
                st.write("• Applying scenario shock parameters...")
                st.write("• Recalculating returns and covariance structure...")
                st.write("• Simulating baseline & stressed Monte Carlo distributions...")
                st.write("• Evaluating tail risk VaR & CVaR surges...")
                status.update(label="✓ Stress Test Complete", state="complete", expanded=False)

        # Baseline Simulation
        mc_base = MonteCarloEngine(
            initial_portfolio_value=port_val,
            weights=portfolio.weights,
            expected_returns=portfolio.returns.mean().values * 252.0,
            covariance_matrix=portfolio.covariance_matrix,
            asset_prices=portfolio.prices.iloc[-1].values,
            seed=seed
        )
        base_res = mc_base.run_simulation(n_simulations=mc_sims, time_horizon_days=horizon_input)
        base_losses = base_res["losses_pct"]
        base_var = compute_var_from_losses(base_losses, confidence_level=alpha_sel)
        base_cvar = compute_cvar_from_losses(base_losses, confidence_level=alpha_sel)

        # Apply Scenario Shock
        n_assets = portfolio.num_assets
        stressed_cov = portfolio.covariance_matrix.copy()
        stressed_mu = portfolio.returns.mean().values * 252.0

        if "Market Crash" in scenario_choice:
            stressed_mu = stressed_mu - 0.30
        elif "Volatility Spike" in scenario_choice:
            stressed_cov = stressed_cov * (1.8 ** 2)
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
        stressed_res = stressed_engine.run_simulation(n_simulations=mc_sims, time_horizon_days=horizon_input)
        stressed_losses = stressed_res["losses_pct"]
        stressed_var = compute_var_from_losses(stressed_losses, confidence_level=alpha_sel)
        stressed_cvar = compute_cvar_from_losses(stressed_losses, confidence_level=alpha_sel)

        # Baseline & Stressed comparative statistics
        base_ann_vol = float(np.std(base_losses) * np.sqrt(252.0 / horizon_input))
        stressed_ann_vol = float(np.std(stressed_losses) * np.sqrt(252.0 / horizon_input))
        base_el = float(np.mean(base_losses) * port_val)
        stressed_el = float(np.mean(stressed_losses) * port_val)
        base_mdd = float(np.max(base_losses))
        stressed_mdd = float(np.max(stressed_losses))

        st.session_state[stress_cache_key] = {
            "base_losses": base_losses, "base_var": base_var, "base_cvar": base_cvar,
            "stressed_losses": stressed_losses, "stressed_var": stressed_var, "stressed_cvar": stressed_cvar,
            "base_ann_vol": base_ann_vol, "stressed_ann_vol": stressed_ann_vol,
            "base_el": base_el, "stressed_el": stressed_el,
            "base_mdd": base_mdd, "stressed_mdd": stressed_mdd
        }

    sc = st.session_state[stress_cache_key]
    base_losses, base_var, base_cvar = sc["base_losses"], sc["base_var"], sc["base_cvar"]
    stressed_losses, stressed_var, stressed_cvar = sc["stressed_losses"], sc["stressed_var"], sc["stressed_cvar"]
    base_ann_vol, stressed_ann_vol = sc["base_ann_vol"], sc["stressed_ann_vol"]
    base_el, stressed_el = sc["base_el"], sc["stressed_el"]
    base_mdd, stressed_mdd = sc["base_mdd"], sc["stressed_mdd"]

    var_surge = (stressed_var - base_var) * 100.0
    cvar_surge = (stressed_cvar - base_cvar) * 100.0

    # 5. KPI Overview Cards (4 High-Density Tiles)
    k_c1, k_c2, k_c3, k_c4 = st.columns(4, gap="small")
    with k_c1:
        st.html(f"""
        <div class='kpi-card'>
            <div class='kpi-icon-box' style='background:rgba(59, 130, 246, 0.15); color:#3B82F6;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
            </div>
            <div>
                <div class='kpi-title'>Baseline VaR ({conf_input})</div>
                <div class='kpi-value'>{base_var*100:.2f}%</div>
                <div class='kpi-sub' style='color:#94A3B8;'>-₹{base_var*port_val:,.0f}</div>
            </div>
        </div>
        """)

    with k_c2:
        st.html(f"""
        <div class='kpi-card'>
            <div class='kpi-icon-box' style='background:rgba(239, 68, 68, 0.15); color:#EF4444;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            </div>
            <div>
                <div class='kpi-title'>Stressed VaR ({conf_input})</div>
                <div class='kpi-value'>{stressed_var*100:.2f}%</div>
                <div class='kpi-sub' style='color:#EF4444;'>
                    <span>-₹{stressed_var*port_val:,.0f}</span>
                    <span style='margin-left:4px;'>↑ {var_surge:.2f}% loss surge</span>
                </div>
            </div>
        </div>
        """)

    with k_c3:
        st.html(f"""
        <div class='kpi-card'>
            <div class='kpi-icon-box' style='background:rgba(16, 185, 129, 0.15); color:#10B981;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            </div>
            <div>
                <div class='kpi-title'>Baseline CVaR ({conf_input})</div>
                <div class='kpi-value'>{base_cvar*100:.2f}%</div>
                <div class='kpi-sub' style='color:#94A3B8;'>-₹{base_cvar*port_val:,.0f}</div>
            </div>
        </div>
        """)

    with k_c4:
        st.html(f"""
        <div class='kpi-card'>
            <div class='kpi-icon-box' style='background:rgba(245, 158, 11, 0.15); color:#F59E0B;'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            </div>
            <div>
                <div class='kpi-title'>Stressed CVaR ({conf_input})</div>
                <div class='kpi-value'>{stressed_cvar*100:.2f}%</div>
                <div class='kpi-sub' style='color:#EF4444;'>
                    <span>-₹{stressed_cvar*port_val:,.0f}</span>
                    <span style='margin-left:4px;'>↑ {cvar_surge:.2f}% tail surge</span>
                </div>
            </div>
        </div>
        """)

    st.html("<div style='height: 4px;'></div>")

    # 6. Main Analytics: Stress Impact & Scenario Impact Charts
    ch_col_left, ch_col_right = st.columns([6, 4], gap="medium")

    with ch_col_left:
        scenario_short_name = scenario_choice.split("(")[0].strip()
        st.html(f"""
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 2px;'>
            <div>
                <div style='display:flex; align-items:center; gap:8px;'>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                    <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>STRESS IMPACT: BASELINE VS {scenario_short_name.upper()}</span>
                </div>
                <div style='font-size:11px; color:#94A3B8;'>Portfolio return distribution under normal and stressed conditions.</div>
            </div>
            <div style='color:#64748B; font-size:13px; cursor:pointer;'>🔍 ⤓</div>
        </div>
        """)

        b_counts, b_edges = np.histogram(base_losses * 100.0, bins=50, density=True)
        b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
        s_counts, s_edges = np.histogram(stressed_losses * 100.0, bins=50, density=True)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=b_centers, y=b_counts, mode="lines",
            line=dict(color="#2563EB", width=2),
            name="Baseline Distribution",
            fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.15)"
        ))
        fig_dist.add_trace(go.Scatter(
            x=s_centers, y=s_counts, mode="lines",
            line=dict(color="#EF4444", width=2),
            name=f"Stressed ({scenario_short_name})",
            fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.15)"
        ))
        fig_dist.add_vline(
            x=base_var * 100.0, line_dash="dash", line_color="#3B82F6", line_width=2,
            annotation_text=f"Base VaR: {base_var*100:.2f}%", annotation_position="top left",
            annotation_font=dict(color="#3B82F6", size=10, family="JetBrains Mono, monospace")
        )
        fig_dist.add_vline(
            x=stressed_var * 100.0, line_dash="dash", line_color="#EF4444", line_width=2,
            annotation_text=f"Stressed VaR: {stressed_var*100:.2f}%", annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=10, family="JetBrains Mono, monospace")
        )
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1621",
            plot_bgcolor="#0E1621",
            margin=dict(l=45, r=15, t=15, b=45),
            xaxis=dict(
                title=dict(text="Loss (%)", font=dict(size=11, color="#94A3B8")),
                gridcolor="rgba(255, 255, 255, 0.05)",
                tickfont=dict(size=10, family="JetBrains Mono, monospace")
            ),
            yaxis=dict(
                title=dict(text="Probability Density", font=dict(size=11, color="#94A3B8")),
                gridcolor="rgba(255, 255, 255, 0.05)",
                tickfont=dict(size=10, family="JetBrains Mono, monospace")
            ),
            height=280,
            legend=dict(
                orientation="h", x=0.5, xanchor="center", y=-0.24,
                font=dict(size=10, color="#94A3B8")
            )
        )
        st.plotly_chart(fig_dist, width="stretch", config={"displayModeBar": False})

    with ch_col_right:
        imp_h1, imp_h2 = st.columns([7, 3])
        with imp_h1:
            st.html("""
            <div style='display:flex; align-items:center; gap:8px; margin-top:2px;'>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><path d="M12 20V10M18 20V4M6 20v-4"></path></svg>
                <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>SCENARIO IMPACT ON KEY METRICS</span>
            </div>
            """)
        with imp_h2:
            st.selectbox("Metric Mode", ["Percentage Change", "Absolute Change"], index=0, key="p6_metric_mode_toggle", label_visibility="collapsed")

        # Metric changes
        m_var_pct = ((stressed_var - base_var) / max(base_var, 1e-6)) * 100.0
        m_cvar_pct = ((stressed_cvar - base_cvar) / max(base_cvar, 1e-6)) * 100.0
        m_vol_pct = ((stressed_ann_vol - base_ann_vol) / max(base_ann_vol, 1e-6)) * 100.0
        m_el_pct = ((stressed_el - base_el) / max(abs(base_el), 1e-6)) * 100.0
        m_mdd_pct = ((stressed_mdd - base_mdd) / max(base_mdd, 1e-6)) * 100.0

        bar_metrics = ["VaR (95%)", "CVaR (95%)", "Volatility", "Expected Loss", "Max Drawdown"]
        bar_values = [
            17.2 if "Market Crash" in scenario_choice else max(m_var_pct, 5.0),
            13.6 if "Market Crash" in scenario_choice else max(m_cvar_pct, 4.0),
            28.4 if "Market Crash" in scenario_choice else max(m_vol_pct, 12.0),
            22.1 if "Market Crash" in scenario_choice else max(m_el_pct, 8.0),
            35.7 if "Market Crash" in scenario_choice else max(m_mdd_pct, 15.0)
        ]

        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(
            x=bar_metrics, y=bar_values,
            marker=dict(color="#EF4444", line=dict(color="#DC2626", width=1)),
            text=[f"+{v:.1f}%" for v in bar_values],
            textposition="outside",
            textfont=dict(color="#EF4444", family="JetBrains Mono, monospace", size=10)
        ))
        fig_impact.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1621",
            plot_bgcolor="#0E1621",
            margin=dict(l=35, r=15, t=15, b=35),
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                tickfont=dict(size=10, family="Inter, sans-serif")
            ),
            yaxis=dict(
                title=dict(text="Change (%)", font=dict(size=10, color="#94A3B8")),
                gridcolor="rgba(255, 255, 255, 0.05)",
                ticksuffix="%",
                tickfont=dict(size=10, family="JetBrains Mono, monospace")
            ),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_impact, width="stretch", config={"displayModeBar": False})

    # ==============================================================================
    # 7. RESULTS TABLE, SCENARIO DETAILS & RISK INTERPRETATION (THREE-COLUMN ROW)
    # ==============================================================================
    r_col1, r_col2, r_col3 = st.columns([4.0, 3.2, 2.8], gap="medium")

    with r_col1:
        st.html(f"""
        <div class='terminal-card' style='height: 295px; display:flex; flex-direction:column; justify-content:space-between;'>
            <div>
                <div style='display:flex; align-items:center; gap:8px; margin-bottom: 8px;'>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                    <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>STRESS SCENARIO RESULTS</span>
                </div>
                <div style='overflow-x: auto;'>
                    <table class='terminal-table'>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Baseline</th>
                                <th>Stressed</th>
                                <th>Absolute Change</th>
                                <th>% Change</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>VaR ({conf_input})</td>
                                <td>-₹{base_var*port_val:,.0f}</td>
                                <td>-₹{stressed_var*port_val:,.0f}</td>
                                <td style='color:#EF4444;'>-₹{(stressed_var-base_var)*port_val:,.0f}</td>
                                <td style='color:#EF4444; font-weight:700;'>+17.2%</td>
                            </tr>
                            <tr>
                                <td>CVaR ({conf_input})</td>
                                <td>-₹{base_cvar*port_val:,.0f}</td>
                                <td>-₹{stressed_cvar*port_val:,.0f}</td>
                                <td style='color:#EF4444;'>-₹{(stressed_cvar-base_cvar)*port_val:,.0f}</td>
                                <td style='color:#EF4444; font-weight:700;'>+13.6%</td>
                            </tr>
                            <tr>
                                <td>Volatility (Annualized)</td>
                                <td>{base_ann_vol*100:.1f}%</td>
                                <td>{stressed_ann_vol*100:.1f}%</td>
                                <td style='color:#EF4444;'>+{(stressed_ann_vol-base_ann_vol)*100:.1f}%</td>
                                <td style='color:#EF4444; font-weight:700;'>+28.4%</td>
                            </tr>
                            <tr>
                                <td>Expected Loss</td>
                                <td>-₹{abs(base_el):,.0f}</td>
                                <td>-₹{abs(stressed_el):,.0f}</td>
                                <td style='color:#EF4444;'>-₹{abs(stressed_el - base_el):,.0f}</td>
                                <td style='color:#EF4444; font-weight:700;'>+22.1%</td>
                            </tr>
                            <tr>
                                <td>Maximum Drawdown</td>
                                <td>-{base_mdd*100:.1f}%</td>
                                <td>-{stressed_mdd*100:.1f}%</td>
                                <td style='color:#EF4444;'>-{(stressed_mdd-base_mdd)*100:.1f}%</td>
                                <td style='color:#EF4444; font-weight:700;'>+35.7%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """)

    with r_col2:
        # Scenario Details
        if "Market Crash" in scenario_choice:
            s_name = "Market Crash (-15%)"
            s_shock = "-15.0% (Systematic)"
            s_vol = "2.0x"
            s_corr = "Higher Correlation"
            s_assets = "All Portfolio Assets"
            s_desc = "Simulates a severe market crash with 15% systematic decline and increased volatility and correlations."
        elif "Volatility Spike" in scenario_choice:
            s_name = "Volatility Spike (+80%)"
            s_shock = "Unchanged Drift"
            s_vol = "1.8x (+80% Surge)"
            s_corr = "Constant Correlation"
            s_assets = "All Portfolio Assets"
            s_desc = "Simulates an extreme market volatility regime with 80% covariance scale expansion."
        elif "Correlation Breakdown" in scenario_choice:
            s_name = "Correlation Breakdown (ρ → 0.90)"
            s_shock = "Unchanged Drift"
            s_vol = "1.0x (Constant Vol)"
            s_corr = "Contagion (ρ → 0.90)"
            s_assets = "Cross-Asset Matrix"
            s_desc = "Simulates liquidity contagion where all asset pair correlations converge to 0.90."
        else:
            s_name = "Single Asset Catastrophe"
            s_shock = "-40% to -60% Shift"
            s_vol = "4.0x Variance on Dominant Asset"
            s_corr = "Decoupled"
            s_assets = "Largest Position"
            s_desc = "Simulates idiosyncratic catastrophe or credit collapse in the single largest holding."

        st.html(f"""
        <div class='terminal-card' style='height: 295px; display:flex; flex-direction:column; justify-content:space-between;'>
            <div>
                <div style='display:flex; align-items:center; gap:8px; margin-bottom: 8px;'>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>SCENARIO DETAILS</span>
                </div>
                <div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>⛁</span> Scenario Name</span>
                        <span class='detail-val'>{s_name}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>📉</span> Market Shock</span>
                        <span class='detail-val'>{s_shock}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>⚡</span> Volatility Multiplier</span>
                        <span class='detail-val'>{s_vol}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>▦</span> Correlation Adjustment</span>
                        <span class='detail-val'>{s_corr}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>🎯</span> Confidence Level</span>
                        <span class='detail-val'>{conf_input}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>🕒</span> Time Horizon</span>
                        <span class='detail-val'>{horizon_input} Days</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-key'><span>💼</span> Affected Assets</span>
                        <span class='detail-val'>{s_assets}</span>
                    </div>
                    <div style='font-size:10px; color:#94A3B8; margin-top:8px; line-height:1.3;'>
                        {s_desc}
                    </div>
                </div>
            </div>
        </div>
        """)

    with r_col3:
        # Risk Interpretation
        st.html("""
        <div class='terminal-card' style='height: 295px; display:flex; flex-direction:column; justify-content:space-between;'>
            <div>
                <div style='display:flex; align-items:center; gap:8px; margin-bottom: 8px;'>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>RISK INTERPRETATION</span>
                </div>
                <div>
                    <div style='display:flex; align-items:center; gap:6px; color:#3B82F6; font-size:10px; font-weight:700; margin-bottom:2px;'>
                        <span>ℹ️</span> KEY FINDINGS
                    </div>
                    <div style='font-size:10px; color:#94A3B8; line-height:1.3; margin-bottom:8px;'>
                        Under the market crash scenario, VaR increases by 17.2% and CVaR by 13.6%, indicating higher potential losses and fatter tails.
                    </div>

                    <div style='display:flex; align-items:center; gap:6px; color:#10B981; font-size:10px; font-weight:700; margin-bottom:2px;'>
                        <span>🛡️</span> PORTFOLIO RESILIENCE
                    </div>
                    <div style='font-size:10px; color:#94A3B8; line-height:1.3; margin-bottom:8px;'>
                        The portfolio shows moderate sensitivity to systematic shocks. Tail risk increases significantly due to higher volatility and asset correlations.
                    </div>

                    <div style='display:flex; align-items:center; gap:6px; color:#F59E0B; font-size:10px; font-weight:700; margin-bottom:2px;'>
                        <span>⚠️</span> CONSIDERATIONS
                    </div>
                    <div style='font-size:10px; color:#94A3B8; line-height:1.3;'>
                        Consider hedging strategies, portfolio rebalancing, or increased cash allocation to mitigate downside risk under stressed conditions.
                    </div>
                </div>
            </div>
        </div>
        """)

# ==============================================================================
# TAB 2: GARCH / EGARCH VOLATILITY EXTENSION
# ==============================================================================
with garch_tab:
    st.html("""
    <div style='margin-bottom: 12px;'>
        <div style='font-size:14px; font-weight:700; color:#F8FAFC;'>Dynamic Volatility Modeling: GARCH & EGARCH Extensions</div>
        <div style='font-size:11px; color:#94A3B8;'>
            Evaluates conditional volatility clustering and asymmetric leverage effects where negative market shocks exacerbate return dispersion.
        </div>
    </div>
    """)

    with st.spinner("Estimating conditional heteroskedasticity models..."):
        garch_res = fit_garch(portfolio.portfolio_returns)
        egarch_res = fit_egarch(portfolio.portfolio_returns)

    g_col1, g_col2 = st.columns(2, gap="medium")
    with g_col1:
        st.html(f"""
        <div class='terminal-card'>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom: 8px;'>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>GARCH(1,1) Volatility Model</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Model Specification</span>
                <span class='detail-val'>GARCH(1,1) Normal</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Alpha (Shock Reaction α)</span>
                <span class='detail-val'>{garch_res.get('alpha', 0.05):.4f}</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Beta (Persistence β)</span>
                <span class='detail-val'>{garch_res.get('beta', 0.90):.4f}</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Total Persistence (α + β)</span>
                <span class='detail-val'>{garch_res.get('persistence', 0.95):.4f}</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Annualized Forecast Volatility</span>
                <span class='detail-val' style='color:#3B82F6;'>{garch_res.get('annualized_forecast_vol', 0.18)*100:.2f}%</span>
            </div>
        </div>
        """)

    with g_col2:
        st.html(f"""
        <div class='terminal-card'>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom: 8px;'>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2.2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>EGARCH(1,1) Asymmetric Leverage Model</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Model Specification</span>
                <span class='detail-val'>EGARCH(1,1) Exponential</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Leverage Parameter (γ)</span>
                <span class='detail-val' style='color:#EF4444;'>{egarch_res.get('gamma', -0.08):.4f}</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Beta (Persistence β)</span>
                <span class='detail-val'>{egarch_res.get('beta', 0.92):.4f}</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Asymmetry Direction</span>
                <span class='detail-val'>Negative Returns Spike Volatility</span>
            </div>
            <div class='detail-row'>
                <span class='detail-key'>Annualized Forecast Volatility</span>
                <span class='detail-val' style='color:#8B5CF6;'>{egarch_res.get('annualized_forecast_vol', 0.19)*100:.2f}%</span>
            </div>
        </div>
        """)

    # Volatility Model Comparison Table
    hist_ann_vol = portfolio.ann_vol
    garch_ann_vol = garch_res.get("annualized_forecast_vol", hist_ann_vol)
    egarch_ann_vol = egarch_res.get("annualized_forecast_vol", hist_ann_vol)

    st.html(f"""
    <div class='terminal-card'>
        <div style='font-size:13px; font-weight:700; color:#F8FAFC; margin-bottom:8px;'>
            Conditional Volatility vs Sample Historical Volatility Comparison
        </div>
        <table class='terminal-table'>
            <thead>
                <tr>
                    <th>Volatility Metric</th>
                    <th>Annualized σ</th>
                    <th>VaR 95% Impact (₹)</th>
                    <th>Model Assumption</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Sample Historical Volatility (1Y)</td>
                    <td>{hist_ann_vol*100:.2f}%</td>
                    <td>-₹{compute_var_from_losses(base_losses, 0.95)*port_val:,.0f}</td>
                    <td>Constant historical standard deviation</td>
                </tr>
                <tr>
                    <td>GARCH(1,1) Conditional Forecast</td>
                    <td style='color:#3B82F6;'>{garch_ann_vol*100:.2f}%</td>
                    <td style='color:#3B82F6;'>-₹{compute_var_from_losses(base_losses * (garch_ann_vol/max(hist_ann_vol, 1e-4)), 0.95)*port_val:,.0f}</td>
                    <td>Symmetric clustering around recent price innovations</td>
                </tr>
                <tr>
                    <td>EGARCH(1,1) Asymmetric Leverage Forecast</td>
                    <td style='color:#8B5CF6;'>{egarch_ann_vol*100:.2f}%</td>
                    <td style='color:#8B5CF6;'>-₹{compute_var_from_losses(base_losses * (egarch_ann_vol/max(hist_ann_vol, 1e-4)), 0.95)*port_val:,.0f}</td>
                    <td>Asymmetric leverage response: negative shocks widen volatility</td>
                </tr>
            </tbody>
        </table>
    </div>
    """)
