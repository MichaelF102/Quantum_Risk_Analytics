"""
Page 3: Classical Monte Carlo Risk Simulation Engine
Redesigned to a Bloomberg / TradingView-inspired institutional quantitative-risk research terminal.
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
        page_title="Quantum Risk Terminal - Classical Engine",
        page_icon="📈",
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
    from app.components.explanations import explain_monte_carlo
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme
    from components.explanations import explain_monte_carlo

from src.data.loader import load_demo_data, load_user_csv
from src.portfolio.portfolio import Portfolio
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

    /* Maximize container resolution and eliminate cramped layout */
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

    /* Simulation Control Bar */
    .sim-ctrl-bar {
        background-color: #0E1621;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 16px;
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
        font-size: 12px;
        color: #94A3B8;
        font-weight: 500;
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
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
        font-weight: 500;
    }

    /* Risk Metrics Table */
    .risk-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .risk-table th {
        background-color: #111B27;
        color: #60A5FA;
        font-weight: 600;
        font-size: 11px;
        text-align: right;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        letter-spacing: 0.3px;
    }
    .risk-table th:first-child {
        text-align: left;
    }
    .risk-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        color: #F8FAFC;
        text-align: right;
    }
    .risk-table td:first-child {
        text-align: left;
        color: #F8FAFC;
        font-weight: 600;
    }
    .risk-table tr:hover td {
        background-color: rgba(255, 255, 255, 0.02);
    }
    .risk-ci-cell {
        color: #94A3B8;
    }
    .risk-se-cell {
        color: #94A3B8;
    }

    /* Configuration key-value rows */
    .config-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 12px;
    }
    .config-row:last-child {
        border-bottom: none;
    }
    .config-key {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #94A3B8;
    }
    .config-val {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-weight: 600;
        color: #F8FAFC;
    }

    /* Streamlit input styling overrides to match compact terminal */
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
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 11px !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        margin-bottom: 2px !important;
    }

    /* Sidebar Page Link styling */
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        background: transparent !important;
        border-radius: 6px !important;
        color: #94A3B8 !important;
        font-size: 13px !important;
        padding: 7px 12px !important;
        margin-bottom: 2px !important;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #F8FAFC !important;
    }
    .active-nav-pill {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 7px 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 12px rgba(29, 78, 216, 0.5);
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

# Ensure prices and portfolio exist
if "prices_df" not in st.session_state:
    st.session_state["prices_df"] = load_demo_data()

prices_df = st.session_state["prices_df"]
if "portfolio" not in st.session_state or st.session_state.get("_last_prices_shape") != prices_df.shape:
    st.session_state["portfolio"] = Portfolio(prices_df, initial_value=st.session_state["portfolio_val"])
    st.session_state["_last_prices_shape"] = prices_df.shape

portfolio = st.session_state["portfolio"]

# ==============================================================================
# LEFT SIDEBAR: CUSTOM QUANTITATIVE RESEARCH TERMINAL NAVIGATION
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
            <a href='/Classical_Engine' target='_self' class='terminal-nav-link terminal-nav-active'>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <circle cx="12" cy="12" r="4" fill="#FFFFFF"></circle>
                </svg>
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
            key="p3_data_mode",
            label_visibility="collapsed"
        )
        if data_mode == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="p3_csv_uploader")
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
            key="p3_port_val"
        )
        if new_port_val != st.session_state["portfolio_val"]:
            st.session_state["portfolio_val"] = new_port_val
            portfolio.initial_value = new_port_val

        new_conf = st.selectbox(
            "Confidence Level (α)",
            [0.90, 0.95, 0.99, 0.995],
            index=[0.90, 0.95, 0.99, 0.995].index(st.session_state["conf_level"]) if st.session_state["conf_level"] in [0.90, 0.95, 0.99, 0.995] else 1,
            format_func=lambda x: f"{int(x*100) if x in [0.90, 0.95, 0.99] else 99.5}%",
            key="p3_conf_level"
        )
        st.session_state["conf_level"] = new_conf

    # Simulation Settings Section
    with st.expander("🎛 Simulation Settings", expanded=False):
        new_seed = st.number_input("Deterministic Seed", min_value=1, max_value=999999, value=int(st.session_state["seed"]), step=1, key="p3_seed")
        st.session_state["seed"] = new_seed

    st.html("<div style='height: 12px;'></div>")

    # Bottom summary card
    st.html("""
    <div style='background: #0E1621; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 14px; margin-top: 10px;'>
        <div style='font-size: 12px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;'>Classical Engine</div>
        <div style='font-size: 11px; color: #94A3B8; line-height: 1.4;'>
            Monte Carlo simulation using Geometric Brownian Motion with multivariate normal sampling.
        </div>
    </div>
    <div style='margin-top: 14px; font-size: 11px; color: #64748B; font-family: monospace;'>
        v2.4 • Research Terminal
    </div>
    """)

# Retrieve active parameters
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
# 2. BREADCRUMB & PAGE HEADER
# ==============================================================================
now_str = datetime.now().strftime("%b %d, %Y, %I:%M %p")

header_col_left, header_col_right = st.columns([8.2, 3.8])

with header_col_left:
    st.html(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;'>
        <div style='font-size:12px; color:#94A3B8; font-weight:500;'>
            Risk Analysis &nbsp;&gt;&nbsp; <span style='color:#F8FAFC;'>Classical Engine</span>
        </div>
    </div>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:4px;'>
        <h1 style='margin:0; font-size:22px; font-weight:700; color:#F8FAFC; letter-spacing:-0.4px;'>
            03. Classical Monte Carlo Engine
        </h1>
        <span style='background:rgba(37,99,235,0.15); border:1px solid rgba(59,130,246,0.4); color:#60A5FA; font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; letter-spacing:0.5px;'>
            CLASSICAL
        </span>
    </div>
    <div style='font-size:12px; color:#94A3B8; margin-bottom:12px;'>
        High-performance vectorized geometric Brownian motion & multivariate normal simulation for portfolio risk analysis.
    </div>
    """)

with header_col_right:
    st.html(f"""
    <div style='text-align:right; font-size:11px; color:#94A3B8; font-family:monospace; margin-bottom:10px;'>
        📅 {now_str}
    </div>
    """)
    btn_col1, btn_col2 = st.columns([4, 1])
    with btn_col1:
        run_sim_clicked = st.button("▶ Run Monte Carlo Simulation", key="btn_run_monte_carlo", type="primary", width="stretch")
    with btn_col2:
        settings_clicked = st.button("⚙️", key="btn_settings_icon", width="stretch", help="Simulation Parameters")

# ==============================================================================
# 3. SIMULATION CONTROL BAR
# ==============================================================================
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([3.5, 3.5, 2.5, 2.5], gap="small")

with ctrl_c1:
    sim_model_choice = st.selectbox(
        "Simulation Model ⓘ",
        ["GBM (Geometric Brownian Motion)", "MVN (Multivariate Normal)"],
        index=0,
        key="ctrl_sim_model",
        help="Geometric Brownian Motion integrates drift and volatility with exponential price dynamics."
    )

with ctrl_c2:
    cov_choice = st.selectbox(
        "Covariance Estimation ⓘ",
        ["Sample Covariance", "Ledoit-Wolf Shrinkage"],
        index=0,
        key="ctrl_cov_est",
        help="Sample covariance vs shrinkage estimator for conditioned variance."
    )

with ctrl_c3:
    time_horizon_input = st.number_input(
        "Time Horizon (Days)",
        min_value=1,
        max_value=60,
        value=int(st.session_state["time_horizon"]),
        step=1,
        key="ctrl_time_horizon"
    )
    if time_horizon_input != st.session_state["time_horizon"]:
        st.session_state["time_horizon"] = time_horizon_input
        time_horizon = time_horizon_input

with ctrl_c4:
    paths_options = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    curr_paths = int(st.session_state["mc_sims"])
    if curr_paths not in paths_options:
        paths_options.append(curr_paths)
        paths_options.sort()
    paths_choice = st.selectbox(
        "Number of Paths ⓘ",
        paths_options,
        index=paths_options.index(curr_paths) if curr_paths in paths_options else 3,
        format_func=lambda x: f"{x:,}",
        key="ctrl_num_paths",
        help="Total number of simulated portfolio sample paths."
    )
    if paths_choice != st.session_state["mc_sims"]:
        st.session_state["mc_sims"] = paths_choice
        mc_sims = paths_choice

# ==============================================================================
# RUN SIMULATION ENGINE & EXECUTION STATE
# ==============================================================================
method_code = "gbm" if "GBM" in sim_model_choice else "mvn"

# Covariance matrix calculation
if "Ledoit-Wolf" in cov_choice:
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(portfolio.returns)
        active_cov = lw.covariance_ * 252.0
    except Exception:
        active_cov = portfolio.covariance_matrix
else:
    active_cov = portfolio.covariance_matrix

if run_sim_clicked:
    with st.status("RUNNING MONTE CARLO ENGINE", expanded=True) as status:
        st.write("• Vectorizing multivariate Geometric Brownian Motion paths...")
        st.write(f"• Estimating covariance ({cov_choice})...")
        st.write("• Simulating terminal portfolio values across 20,000 draws...")
        st.write("• Evaluating empirical loss quantiles (VaR & CVaR)...")
        st.write("• Calculating asymptotic 95% confidence intervals & standard errors...")
        status.update(label="✓ Simulation Complete", state="complete", expanded=False)

try:
    mc_engine = MonteCarloEngine(
        initial_portfolio_value=port_val,
        weights=portfolio.weights,
        expected_returns=portfolio.returns.mean().values * 252.0,
        covariance_matrix=active_cov,
        asset_prices=portfolio.prices.iloc[-1].values,
        seed=seed
    )

    sim_res = mc_engine.run_simulation(
        n_simulations=mc_sims,
        time_horizon_days=time_horizon,
        confidence_levels=[0.90, 0.95, 0.995],
        method=method_code
    )
except Exception as e:
    st.error(f"SIMULATION ERROR: Unable to execute the Monte Carlo simulation: {e}")
    st.button("Retry Simulation", on_click=lambda: st.rerun())
    st.stop()

# Retrieve risk metrics
risk_metrics = sim_res["risk_metrics"]
# Ensure standard keys
m_90 = risk_metrics.get("90%", None)
m_95 = risk_metrics.get("95%", None)
m_995 = risk_metrics.get("99%", None) or risk_metrics.get("99.5%", None)

# 95% default for primary cards
ref_m = m_95 if m_95 is not None else list(risk_metrics.values())[0]
var_95_cash = ref_m["var_cash"]
var_95_pct = ref_m["var_pct"]
cvar_95_cash = ref_m["cvar_cash"]
cvar_95_pct = ref_m["cvar_pct"]
exec_time_ms = sim_res["runtime_seconds"] * 1000.0

# ==============================================================================
# 4. KPI CARDS (4 POLISHED INSTITUTIONAL TILES)
# ==============================================================================
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4, gap="small")

with kpi_c1:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(59, 130, 246, 0.15); color:#3B82F6;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Simulated Paths</div>
            <div class='kpi-value'>{mc_sims:,}</div>
            <div class='kpi-sub' style='color:#22C55E;'>
                <span>↑</span> Vectorized (NumPy)
            </div>
        </div>
    </div>
    """)

with kpi_c2:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(16, 185, 129, 0.15); color:#10B981;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Execution Time</div>
            <div class='kpi-value'>{exec_time_ms:.1f} ms</div>
            <div class='kpi-sub' style='color:#94A3B8;'>
                <span>⨁</span> Deterministic Seed = {seed}
            </div>
        </div>
    </div>
    """)

with kpi_c3:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(239, 68, 68, 0.15); color:#EF4444;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Classical VaR (95%)</div>
            <div class='kpi-value'>-₹{var_95_cash:,.0f}</div>
            <div class='kpi-sub' style='color:#EF4444;'>
                <span>↓</span> {var_95_pct * 100:.2f}% loss
            </div>
        </div>
    </div>
    """)

with kpi_c4:
    st.html(f"""
    <div class='kpi-card'>
        <div class='kpi-icon-box' style='background:rgba(249, 115, 22, 0.15); color:#F97316;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
        </div>
        <div>
            <div class='kpi-title'>Classical CVaR (95%)</div>
            <div class='kpi-value'>-₹{cvar_95_cash:,.0f}</div>
            <div class='kpi-sub' style='color:#EF4444;'>
                <span>↓</span> {cvar_95_pct * 100:.2f}% tail loss
            </div>
        </div>
    </div>
    """)

st.html("<div style='height: 4px;'></div>")

# ==============================================================================
# 5 & 6. RISK METRICS SECTION (LEFT) & SIMULATION CONFIGURATION PANEL (RIGHT)
# ==============================================================================
mid_left, mid_right = st.columns([6.8, 3.2], gap="medium")

with mid_left:
    # Build dynamic table rows
    rows_html = ""
    for alpha_label, conf_val in [("90.0%", 0.90), ("95.0%", 0.95), ("99.5%", 0.995)]:
        m_item = None
        for k, v in risk_metrics.items():
            if abs(v["confidence_level"] - conf_val) < 1e-4:
                m_item = v
                break
        if m_item is None:
            # Fallback to key index
            if conf_val == 0.90 and "90%" in risk_metrics:
                m_item = risk_metrics["90%"]
            elif conf_val == 0.95 and "95%" in risk_metrics:
                m_item = risk_metrics["95%"]
            elif conf_val == 0.995 and ("99%" in risk_metrics or "99.5%" in risk_metrics):
                m_item = risk_metrics.get("99.5%", risk_metrics.get("99%"))

        if m_item is not None:
            v_pct = m_item["var_pct"] * 100.0
            v_cash = m_item["var_cash"]
            ci_l = m_item["var_ci_95"][0] * 100.0
            ci_h = m_item["var_ci_95"][1] * 100.0
            se_pct = m_item["var_se"] * 100.0
            cv_pct = m_item["cvar_pct"] * 100.0
            cv_cash = m_item["cvar_cash"]

            rows_html += f"""
            <tr>
                <td>{alpha_label}</td>
                <td>{v_pct:.3f}%</td>
                <td>-₹{v_cash:,.0f}</td>
                <td class='risk-ci-cell'>[{ci_l:.3f}%, {ci_h:.3f}%]</td>
                <td class='risk-se-cell'>{se_pct:.4f}%</td>
                <td>{cv_pct:.3f}%</td>
                <td>-₹{cv_cash:,.0f}</td>
            </tr>
            """

    st.html(f"""
    <div class='terminal-card' style='height: 250px; display:flex; flex-direction:column; justify-content:space-between;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;'>
            <div style='display:flex; align-items:center; gap:8px;'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
                <span style='font-size:14px; font-weight:700; color:#F8FAFC;'>
                    Estimated Risk Metrics (95% Asymptotic Confidence Intervals)
                </span>
            </div>
            <div style='color:#64748B; cursor:pointer;'>⋮</div>
        </div>
        <div style='overflow-x: auto;'>
            <table class='risk-table'>
                <thead>
                    <tr>
                        <th>Confidence Level</th>
                        <th>VaR (%)</th>
                        <th>VaR (₹)</th>
                        <th>95% CI (VaR %)</th>
                        <th>Std. Error (SE)</th>
                        <th>CVaR (%)</th>
                        <th>CVaR (₹)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """)

with mid_right:
    st.html(f"""
    <div class='terminal-card' style='height: 250px; display:flex; flex-direction:column; justify-content:space-between;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;'>
            <div style='display:flex; align-items:center; gap:8px;'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
                <span style='font-size:14px; font-weight:700; color:#F8FAFC;'>Simulation Configuration</span>
            </div>
            <div style='color:#64748B;'>⌃</div>
        </div>
        <div>
            <div class='config-row'>
                <span class='config-key'>
                    <span style='font-size:13px;'>🪙</span> Initial Portfolio Value
                </span>
                <span class='config-val'>₹{port_val:,.0f}</span>
            </div>
            <div class='config-row'>
                <span class='config-key'>
                    <span style='font-size:13px;'>📊</span> Model
                </span>
                <span class='config-val'>{sim_model_choice}</span>
            </div>
            <div class='config-row'>
                <span class='config-key'>
                    <span style='font-size:13px;'>▦</span> Covariance
                </span>
                <span class='config-val'>{cov_choice}</span>
            </div>
            <div class='config-row'>
                <span class='config-key'>
                    <span style='font-size:13px;'>🕒</span> Time Horizon
                </span>
                <span class='config-val'>{time_horizon} Days</span>
            </div>
            <div class='config-row'>
                <span class='config-key'>
                    <span style='font-size:13px;'>⎘</span> Number of Paths
                </span>
                <span class='config-val'>{mc_sims:,}</span>
            </div>
            <div class='config-row'>
                <span class='config-key'>
                    <span style='font-size:13px;'>&lt;/&gt;</span> Deterministic Seed
                </span>
                <span class='config-val'>{seed}</span>
            </div>
        </div>
    </div>
    """)

# ==============================================================================
# 7. CHART SECTION (TWO-COLUMN ANALYTICAL LAYOUT)
# ==============================================================================
chart_left, chart_right = st.columns([6, 6], gap="medium")

# --- LEFT CHART: TERMINAL PORTFOLIO VALUES ---
with chart_left:
    st.html(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px; padding: 0 4px;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>
                Simulated Terminal Portfolio Values ({time_horizon}D Horizon)
            </span>
        </div>
        <div style='color:#64748B; font-size:14px; cursor:pointer;'>⤓</div>
    </div>
    """)

    fig_term = go.Figure()
    fig_term.add_trace(go.Histogram(
        x=sim_res["simulated_values"],
        nbinsx=65,
        marker=dict(
            color="#2563EB",
            opacity=0.85,
            line=dict(color="#1D4ED8", width=0.5)
        ),
        name="Terminal Value"
    ))

    # Initial Portfolio Value marker
    fig_term.add_vline(
        x=port_val,
        line_dash="dash",
        line_color="#CBD5E1",
        line_width=2,
        annotation_text=f"Initial: ₹{port_val:,.0f}",
        annotation_position="top",
        annotation_font=dict(color="#CBD5E1", size=11, family="JetBrains Mono, monospace")
    )

    fig_term.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1621",
        plot_bgcolor="#0E1621",
        margin=dict(l=45, r=20, t=25, b=40),
        xaxis=dict(
            title=dict(text="Terminal Value (₹)", font=dict(size=11, color="#94A3B8")),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickformat="~s",
            tickfont=dict(size=10, family="JetBrains Mono, monospace")
        ),
        yaxis=dict(
            title=dict(text="Frequency", font=dict(size=11, color="#94A3B8")),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(size=10, family="JetBrains Mono, monospace")
        ),
        height=360,
        showlegend=False
    )
    st.plotly_chart(fig_term, width="stretch", config={"displayModeBar": False})

# --- RIGHT CHART: LOSS DISTRIBUTION WITH VAR/CVAR ---
with chart_right:
    st.html(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px; padding: 0 4px;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <span style='font-size:13px; font-weight:700; color:#F8FAFC;'>
                Classical Monte Carlo Loss Distribution (α = {conf_level})
            </span>
        </div>
        <div style='color:#64748B; font-size:14px; cursor:pointer;'>⤓</div>
    </div>
    """)

    # P&L cash distribution where loss is negative
    pnl_cash = sim_res["simulated_values"] - port_val
    counts, bin_edges = np.histogram(pnl_cash, bins=60, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    tail_cutoff = -var_95_cash
    tail_mask = bin_centers <= tail_cutoff
    body_mask = ~tail_mask

    fig_loss = go.Figure()
    # Main body
    fig_loss.add_trace(go.Bar(
        x=bin_centers[body_mask],
        y=counts[body_mask],
        width=bin_width,
        marker=dict(color="#F87171", opacity=0.75, line=dict(color="#EF4444", width=0.5)),
        name="Loss Distribution"
    ))
    # Tail loss (beyond VaR)
    fig_loss.add_trace(go.Bar(
        x=bin_centers[tail_mask],
        y=counts[tail_mask],
        width=bin_width,
        marker=dict(color="#DC2626", opacity=0.95, line=dict(color="#991B1B", width=0.5)),
        name=f"Tail Loss (Top {int((1-conf_level)*100)}%)"
    ))

    # VaR vertical dashed line
    fig_loss.add_vline(
        x=-var_95_cash,
        line_dash="dash",
        line_color="#F59E0B",
        line_width=2,
        annotation_text=f"VaR {int(conf_level*100)}%: -₹{var_95_cash:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color="#F59E0B", size=11, family="JetBrains Mono, monospace")
    )

    # CVaR vertical dashed line
    fig_loss.add_vline(
        x=-cvar_95_cash,
        line_dash="dash",
        line_color="#EF4444",
        line_width=2,
        annotation_text=f"CVaR {int(conf_level*100)}%: -₹{cvar_95_cash:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11, family="JetBrains Mono, monospace")
    )

    fig_loss.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1621",
        plot_bgcolor="#0E1621",
        margin=dict(l=45, r=20, t=25, b=40),
        xaxis=dict(
            title=dict(text="Portfolio Loss (₹)", font=dict(size=11, color="#94A3B8")),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickformat="~s",
            tickfont=dict(size=10, family="JetBrains Mono, monospace")
        ),
        yaxis=dict(
            title=dict(text="Density", font=dict(size=11, color="#94A3B8")),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(size=10, family="JetBrains Mono, monospace")
        ),
        height=360,
        legend=dict(
            orientation="v",
            x=0.98,
            xanchor="right",
            y=0.95,
            bgcolor="rgba(14, 22, 33, 0.7)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=10, color="#94A3B8")
        )
    )
    st.plotly_chart(fig_loss, width="stretch", config={"displayModeBar": False})

# ==============================================================================
# 8. EXPLANATION ACCORDION (PRESERVED)
# ==============================================================================
st.html("<div style='height: 8px;'></div>")
explain_monte_carlo()
