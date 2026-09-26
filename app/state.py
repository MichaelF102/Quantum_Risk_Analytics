"""
Shared state and sidebar management for Quantum Risk Analytics Engine.
"""
import os
import streamlit as st
import numpy as np
import pandas as pd

from src.data.loader import load_demo_data, load_user_csv
from src.portfolio.portfolio import Portfolio
from src.classical.monte_carlo import MonteCarloEngine
from src.quantum.state_preparation import LossDistributionDiscretizer
from src.utils.config import CONFIG
from src.utils.reproducibility import set_seed


def initialize_app_theme():
    """Inject custom CSS for dark research terminal styling."""
    st.markdown("""
    <style>
        /* Base styling */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: #f0f6fc !important;
            font-family: 'JetBrains Mono', SFMono-Regular, Consolas, Menlo, monospace !important;
            letter-spacing: -0.5px;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #21262d;
            color: #58a6ff;
            border: 1px solid #30363d;
            border-radius: 4px;
            font-family: monospace;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #30363d;
            border-color: #58a6ff;
            color: #79c0ff;
        }
        
        /* Cards */
        div[data-testid="stExpander"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
        }
        
        /* Dataframes & Tables */
        .dataframe {
            background-color: #161b22 !important;
            color: #c9d1d9 !important;
            border: 1px solid #30363d !important;
            font-family: monospace !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #161b22;
            padding: 4px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: monospace;
            font-size: 13px;
            color: #8b949e;
            border-radius: 4px;
            padding: 6px 14px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #21262d !important;
            color: #58a6ff !important;
        }
    </style>
    """, unsafe_allow_html=True)


def get_shared_sidebar_inputs():
    """Render sidebar inputs and store selections in session state."""
    st.sidebar.markdown(
        "<div style='font-family:monospace; font-weight:700; color:#58a6ff; font-size:16px; margin-bottom:15px;'>"
        "QUANTUM RISK ENGINE<br><span style='font-size:11px; color:#8b949e;'>v2.4 // RESEARCH TERMINAL</span>"
        "</div>",
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-family:monospace; font-size:11px; color:#8b949e; text-transform:uppercase;'>// 1. DATA SOURCE</div>", unsafe_allow_html=True)
    
    data_mode = st.sidebar.radio("Data Mode", ["Bundled Demo Data", "Upload Custom CSV"], label_visibility="collapsed")
    
    if data_mode == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (Wide or Long)", type=["csv"])
        if uploaded_file is not None:
            try:
                prices_df = load_user_csv(uploaded_file)
                st.sidebar.success(f"Loaded {len(prices_df.columns)} assets, {len(prices_df)} rows")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                prices_df = load_demo_data()
        else:
            st.sidebar.info("Awaiting file. Using demo data.")
            prices_df = load_demo_data()
    else:
        prices_df = load_demo_data()
        
    st.sidebar.markdown("<div style='font-family:monospace; font-size:11px; color:#8b949e; text-transform:uppercase; margin-top:10px;'>// 2. RISK PARAMETERS</div>", unsafe_allow_html=True)
    
    portfolio_val = st.sidebar.number_input(
        "Initial Portfolio Value ($)",
        min_value=10_000.0,
        max_value=100_000_000.0,
        value=1_000_000.0,
        step=50_000.0
    )
    
    conf_level = st.sidebar.selectbox(
        "Confidence Level (α)",
        [0.90, 0.95, 0.99, 0.995],
        index=1,
        format_func=lambda x: f"{int(x*100) if x in [0.9, 0.95, 0.99] else x*100}%"
    )
    
    time_horizon = st.sidebar.slider(
        "Time Horizon (Days)",
        min_value=1,
        max_value=30,
        value=10,
        step=1
    )
    
    st.sidebar.markdown("<div style='font-family:monospace; font-size:11px; color:#8b949e; text-transform:uppercase; margin-top:10px;'>// 3. ENGINE SETTINGS</div>", unsafe_allow_html=True)
    
    mc_sims = st.sidebar.select_slider(
        "Monte Carlo Simulations",
        options=[1_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        value=20_000
    )
    
    num_qubits = st.sidebar.slider(
        "Quantum State Qubits (2ⁿ bins)",
        min_value=3,
        max_value=6,
        value=4,
        help="2^3=8 bins, 2^4=16 bins, 2^5=32 bins, 2^6=64 bins"
    )
    
    backend_choice = st.sidebar.selectbox(
        "Quantum Backend",
        ["Aer Simulator (Shot-based)", "Aer Simulator (Statevector)", "Theoretical QAE"]
    )
    
    seed = st.sidebar.number_input("Deterministic Seed", min_value=1, max_value=999999, value=42, step=1)
    
    # Store in session state
    set_seed(seed)
    st.session_state["prices_df"] = prices_df
    st.session_state["portfolio_val"] = portfolio_val
    st.session_state["conf_level"] = conf_level
    st.session_state["time_horizon"] = time_horizon
    st.session_state["mc_sims"] = mc_sims
    st.session_state["num_qubits"] = num_qubits
    st.session_state["backend_choice"] = backend_choice
    st.session_state["seed"] = seed
    
    # Lazy initialize Portfolio object
    if "portfolio" not in st.session_state or st.session_state.get("_last_prices_shape") != prices_df.shape:
        st.session_state["portfolio"] = Portfolio(prices_df, initial_value=portfolio_val)
        st.session_state["_last_prices_shape"] = prices_df.shape
        
    return {
        "prices_df": prices_df,
        "portfolio_val": portfolio_val,
        "conf_level": conf_level,
        "time_horizon": time_horizon,
        "mc_sims": mc_sims,
        "num_qubits": num_qubits,
        "backend_choice": backend_choice,
        "seed": seed,
        "portfolio": st.session_state["portfolio"]
    }
