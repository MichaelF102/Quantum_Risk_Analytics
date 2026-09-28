"""
Page 1: QUANTUM PORTFOLIO BUILDER
High-Performance Two-Stage Quantitative Portfolio Construction Terminal.
Step 1: Select Assets (Left Column) | Step 2: Configure Portfolio & Allocate (Right Column)
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

import importlib
for _mod_name in [
    "app.state", "state",
    "src.portfolio.portfolio_data",
    "src.portfolio.portfolio_validation",
    "src.portfolio.portfolio",
    "app.components.charts", "components.charts"
]:
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass

import streamlit as st

st.set_page_config(
    page_title="Quantum Portfolio Builder",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from app.state import initialize_app_theme, get_shared_sidebar_inputs
    from app.components.charts import get_terminal_layout
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.charts import get_terminal_layout

from src.portfolio.portfolio import Portfolio
from src.portfolio.portfolio_data import (
    load_stock_universe,
    fetch_historical_prices,
    search_universe
)
from src.portfolio.portfolio_validation import (
    validate_custom_portfolio,
    PortfolioValidationResult
)

# Initialize application theme & shared sidebar
initialize_app_theme()
inputs = get_shared_sidebar_inputs()

# =======================================================
# 1. CUSTOM QUANTUM TERMINAL CSS
# =======================================================
st.html("""
<style>
    /* Base Quantitative Terminal Layout */
    .stApp {
        background-color: #070B12 !important;
        color: #F1F5F9 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    .stAppViewContainer, .main, [data-testid="stAppViewContainer"] {
        width: 100% !important;
    }
    .main .block-container, [data-testid="stAppViewBlockContainer"], [data-testid="block-container"] {
        max-width: 98% !important;
        width: 98% !important;
        padding-top: 0.8rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    
    /* Institutional Terminal Card */
    .qb-card {
        background-color: #101720;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    
    .qb-card-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
        font-weight: 700;
        color: #8B98A8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Terminal Badges */
    .qb-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: JetBrains Mono, monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    
    .qb-badge-valid {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.45);
        color: #3fb950;
    }
    
    .qb-badge-invalid {
        background: rgba(255, 77, 90, 0.15);
        border: 1px solid rgba(255, 77, 90, 0.45);
        color: #FF4D5A;
    }
    
    .qb-badge-info {
        background: rgba(41, 151, 255, 0.12);
        border: 1px solid rgba(41, 151, 255, 0.35);
        color: #2997FF;
    }
    
    /* Terminal Stock Item Row */
    .stock-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        background: #121A24;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        margin-bottom: 6px;
        transition: border 0.15s ease;
    }
    .stock-row:hover {
        border-color: rgba(41, 151, 255, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        white-space: nowrap !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
        padding: 4px 10px !important;
    }
</style>
""")

# =======================================================
# 2. SESSION STATE MANAGEMENT
# =======================================================
if "pb_market" not in st.session_state:
    st.session_state["pb_market"] = "INDIA"
if "pb_alloc_mode" not in st.session_state:
    st.session_state["pb_alloc_mode"] = "Portfolio Weight"
if "pb_total_value_india" not in st.session_state:
    st.session_state["pb_total_value_india"] = 1_000_000.0
if "pb_total_value_us" not in st.session_state:
    st.session_state["pb_total_value_us"] = 100_000.0
if "pb_total_value_global" not in st.session_state:
    st.session_state["pb_total_value_global"] = 100_000.0
if "pb_selected_india" not in st.session_state:
    st.session_state["pb_selected_india"] = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
if "pb_selected_us" not in st.session_state:
    st.session_state["pb_selected_us"] = ["AAPL", "MSFT", "NVDA"]
if "pb_selected_global" not in st.session_state:
    st.session_state["pb_selected_global"] = ["RELIANCE.NS", "AAPL", "MSFT"]
if "pb_weights" not in st.session_state:
    st.session_state["pb_weights"] = {}
if "pb_investments" not in st.session_state:
    st.session_state["pb_investments"] = {}

market = st.session_state["pb_market"]
is_india = (market == "INDIA")
is_us = (market == "US")
is_global = (market == "GLOBAL")

# Dynamic Currency Setting
currency_symbol = "₹" if is_india else ("$" if is_us else "$")
currency_code = "INR" if is_india else ("USD" if is_us else "USD")

# Universe key mapping
if is_india:
    selected_tickers_key = "pb_selected_india"
    default_total_val = float(st.session_state["pb_total_value_india"])
elif is_us:
    selected_tickers_key = "pb_selected_us"
    default_total_val = float(st.session_state["pb_total_value_us"])
else:
    selected_tickers_key = "pb_selected_global"
    default_total_val = float(st.session_state["pb_total_value_global"])

# Load Stock Universe dynamically from CSV
with st.spinner(f"Loading {market} stock universe..."):
    universe_df = load_stock_universe(market)

universe_lookup = {row["ticker"]: row for _, row in universe_df.iterrows()}
all_universe_tickers = set(universe_df["ticker"].tolist())
selected_tickers = [t for t in st.session_state.get(selected_tickers_key, []) if t in universe_lookup]
st.session_state[selected_tickers_key] = selected_tickers
n_assets = len(selected_tickers)

# Helpers for synchronizing widget state with portfolio weights/investments
def set_portfolio_weights(weights_dict: dict, market_key: str):
    """Sets portfolio weights and synchronizes Streamlit widget state."""
    st.session_state["pb_weights"] = dict(weights_dict)
    for t, w in weights_dict.items():
        st.session_state[f"input_w_{t}_{market_key}"] = float(w)

def set_portfolio_investments(investments_dict: dict, market_key: str):
    """Sets portfolio investments and synchronizes Streamlit widget state."""
    st.session_state["pb_investments"] = dict(investments_dict)
    for t, inv in investments_dict.items():
        st.session_state[f"input_inv_{t}_{market_key}"] = float(inv)

def calculate_equal_weights(tickers: list) -> dict:
    """Calculates equal weights ensuring the sum is exactly 100.00%."""
    n = len(tickers)
    if n == 0:
        return {}
    base = round(100.0 / n, 2)
    w_map = {t: base for t in tickers}
    diff = round(100.0 - sum(w_map.values()), 2)
    if diff != 0 and tickers:
        w_map[tickers[-1]] = round(w_map[tickers[-1]] + diff, 2)
    return w_map

def calculate_equal_investments(tickers: list, total_val: float) -> dict:
    """Calculates equal cash investments ensuring the sum equals total_val."""
    n = len(tickers)
    if n == 0:
        return {}
    base = round(total_val / n, 2)
    inv_map = {t: base for t in tickers}
    diff = round(total_val - sum(inv_map.values()), 2)
    if diff != 0 and tickers:
        inv_map[tickers[-1]] = round(inv_map[tickers[-1]] + diff, 2)
    return inv_map

# =======================================================
# 3. TOP HEADER & WORKFLOW PROGRESS INDICATOR
# =======================================================
header_left, header_right = st.columns([7.5, 4.5], gap="medium")

with header_left:
    st.html("""
    <div style='margin-bottom: 2px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='font-size:28px; color:#2997FF; font-weight:800; line-height:1;'>⚛</div>
            <div>
                <h1 style='margin:0; font-size:23px; font-weight:800; color:#F1F5F9; letter-spacing:-0.5px;'>
                    QUANTUM PORTFOLIO BUILDER
                </h1>
                <div style='font-size:12px; color:#8B98A8; margin-top:2px;'>
                    Build a custom multi-asset portfolio for quantum tail-risk analysis.
                </div>
            </div>
        </div>
    </div>
    """)

with header_right:
    st.html(f"""
    <div style='display:flex; justify-content:flex-end; align-items:center; gap:8px; height:100%; padding-top:4px;'>
        <div class='qb-badge qb-badge-info'>
            UNIVERSE: {market}
        </div>
        <div class='qb-badge' style='background:rgba(24,199,181,0.15); border:1px solid rgba(24,199,181,0.35); color:#18C7B5;'>
            CURRENCY: {"INR (₹)" if is_india else "USD ($)"}
        </div>
        <div class='qb-badge' style='background:rgba(155,108,255,0.15); border:1px solid rgba(155,108,255,0.35); color:#9B6CFF;'>
            QISKIT AER
        </div>
    </div>
    """)

# Section 14: Progress Indicator
st.html("""
<div style='display:flex; justify-content:space-between; align-items:center; background:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:8px 16px; margin-top:6px; margin-bottom:12px;'>
    <div style='font-family:monospace; font-size:11px; font-weight:700; color:#8B98A8; letter-spacing:0.8px;'>
        PORTFOLIO WORKFLOW:
    </div>
    <div style='display:flex; align-items:center; gap:12px; font-family:JetBrains Mono, monospace; font-size:11px;'>
        <span style='color:#2997FF; font-weight:700;'>01 SELECT ASSETS</span>
        <span style='color:rgba(255,255,255,0.2);'>━━━</span>
        <span style='color:#9B6CFF; font-weight:700;'>02 ALLOCATE CAPITAL</span>
        <span style='color:rgba(255,255,255,0.2);'>━━━</span>
        <span style='color:#18C7B5; font-weight:700;'>03 QUANTUM RISK ANALYSIS</span>
    </div>
</div>
""")

# =======================================================
# 4. COMPACT MARKET SWITCHER & QUICK PRESETS
# =======================================================
mkt_ctrl_col, preset_ctrl_col = st.columns([6, 6], gap="medium")

with mkt_ctrl_col:
    st.html("<div style='font-size:11px; font-weight:700; color:#8B98A8; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>MARKET UNIVERSE</div>")
    m1, m2, m3 = st.columns(3, gap="small")
    
    with m1:
        type_in = "primary" if is_india else "secondary"
        if st.button("🇮🇳 INDIA", key="sw_mkt_in", type=type_in, width="stretch"):
            st.session_state["pb_market"] = "INDIA"
            st.rerun()
        st.html("<div style='font-size:10px; color:#8B98A8; text-align:center; margin-top:2px;'>NSE / BSE · INR</div>")
        
    with m2:
        type_us = "primary" if is_us else "secondary"
        if st.button("🇺🇸 UNITED STATES", key="sw_mkt_us", type=type_us, width="stretch"):
            st.session_state["pb_market"] = "US"
            st.rerun()
        st.html("<div style='font-size:10px; color:#8B98A8; text-align:center; margin-top:2px;'>NYSE / NASDAQ · USD</div>")
        
    with m3:
        type_gl = "primary" if is_global else "secondary"
        if st.button("🌐 GLOBAL", key="sw_mkt_gl", type=type_gl, width="stretch"):
            st.session_state["pb_market"] = "GLOBAL"
            st.rerun()
        st.html("<div style='font-size:10px; color:#8B98A8; text-align:center; margin-top:2px;'>Multi-Market · USD</div>")

with preset_ctrl_col:
    st.html("<div style='font-size:11px; font-weight:700; color:#8B98A8; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>QUICK START PRESETS</div>")
    p1, p2, p3, p4 = st.columns(4, gap="small")
    
    if is_india:
        with p1:
            if st.button("Top 3", key="qs_in_3", width="stretch", help="Load three representative Indian blue-chips (RELIANCE, TCS, INFY)"):
                st.session_state[selected_tickers_key] = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
                set_portfolio_weights({"RELIANCE.NS": 40.0, "TCS.NS": 35.0, "INFY.NS": 25.0}, market)
                st.rerun()
        with p2:
            if st.button("Top 5", key="qs_in_5", width="stretch", help="Load top 5 leaders across energy, IT, and banking"):
                st.session_state[selected_tickers_key] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS"]
                set_portfolio_weights({t: 20.0 for t in st.session_state[selected_tickers_key]}, market)
                st.rerun()
        with p3:
            if st.button("Single Asset", key="qs_in_1", width="stretch", help="Single-stock benchmark portfolio (100% RELIANCE)"):
                st.session_state[selected_tickers_key] = ["RELIANCE.NS"]
                set_portfolio_weights({"RELIANCE.NS": 100.0}, market)
                st.rerun()
    elif is_us:
        with p1:
            if st.button("Top 3", key="qs_us_3", width="stretch", help="Load mega-cap US leaders (AAPL, MSFT, NVDA)"):
                st.session_state[selected_tickers_key] = ["AAPL", "MSFT", "NVDA"]
                set_portfolio_weights({"AAPL": 50.0, "MSFT": 30.0, "NVDA": 20.0}, market)
                st.rerun()
        with p2:
            if st.button("Top 5", key="qs_us_5", width="stretch", help="Load 5 mega-cap tech leaders (AAPL, MSFT, NVDA, GOOGL, AMZN)"):
                st.session_state[selected_tickers_key] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
                set_portfolio_weights({t: 20.0 for t in st.session_state[selected_tickers_key]}, market)
                st.rerun()
        with p3:
            if st.button("Single Asset", key="qs_us_1", width="stretch", help="Single-stock benchmark portfolio (100% AAPL)"):
                st.session_state[selected_tickers_key] = ["AAPL"]
                set_portfolio_weights({"AAPL": 100.0}, market)
                st.rerun()
    else:
        with p1:
            if st.button("Top 3", key="qs_gl_3", width="stretch", help="Load diversified global mega-caps"):
                st.session_state[selected_tickers_key] = ["RELIANCE.NS", "AAPL", "MSFT"]
                set_portfolio_weights({"RELIANCE.NS": 40.0, "AAPL": 35.0, "MSFT": 25.0}, market)
                st.rerun()
        with p2:
            if st.button("Top 4", key="qs_gl_4", width="stretch", help="Load multi-asset global leaders"):
                st.session_state[selected_tickers_key] = ["RELIANCE.NS", "TCS.NS", "AAPL", "NVDA"]
                set_portfolio_weights({t: 25.0 for t in st.session_state[selected_tickers_key]}, market)
                st.rerun()
        with p3:
            if st.button("Single Asset", key="qs_gl_1", width="stretch", help="Single asset benchmark portfolio"):
                st.session_state[selected_tickers_key] = ["AAPL"]
                set_portfolio_weights({"AAPL": 100.0}, market)
                st.rerun()

    with p4:
        if st.button("⟲ Clear", key="qs_clear", width="stretch", help="Clear all currently selected holdings"):
            st.session_state[selected_tickers_key] = []
            st.session_state["pb_weights"] = {}
            st.session_state["pb_investments"] = {}
            for k in list(st.session_state.keys()):
                if k.startswith("input_w_") or k.startswith("input_inv_"):
                    del st.session_state[k]
            st.rerun()

st.html("<div style='margin-bottom:10px;'></div>")

# =======================================================
# 5. DATA AVAILABILITY PRE-CHECK
# =======================================================
price_df = pd.DataFrame()
data_availability: dict[str, bool] = {}
data_messages: dict[str, str] = {}

if selected_tickers:
    with st.spinner("Verifying market price availability..."):
        price_df, data_availability, data_messages = fetch_historical_prices(selected_tickers)

# =======================================================
# 6. TWO-COLUMN WORKFLOW: STEP 1 (LEFT) vs STEP 2 (RIGHT)
# =======================================================
col_left, col_right = st.columns([6, 6], gap="large")

# -------------------------------------------------------
# LEFT COLUMN: STEP 1 — SELECT ASSETS
# -------------------------------------------------------
with col_left:
    st.html(f"""
    <div class='qb-card'>
        <div class='qb-card-header'>
            <div>
                <span style='color:#2997FF; font-family:monospace; font-weight:800; font-size:15px; margin-right:6px;'>01</span>
                <span style='color:#F1F5F9; font-size:14px; font-weight:700;'>SELECT ASSETS</span>
                <div style='font-size:11px; color:#8B98A8; font-weight:400; text-transform:none; margin-top:2px;'>Choose the stocks that will form your portfolio.</div>
            </div>
            <div class='qb-badge qb-badge-info'>
                {n_assets} / 20 SELECTED
            </div>
        </div>
    """)
    
    # Database size indicator
    st.html(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; background:#121A24; border:1px solid rgba(41,151,255,0.2); border-radius:4px; padding:6px 10px; margin-bottom:8px; font-size:11px; font-family:JetBrains Mono, monospace;'>
        <div style='color:#8B98A8;'>DATABASE: <b style='color:#F1F5F9;'>{len(universe_df):,} STOCKS LOADED</b></div>
        <div style='color:#18C7B5; font-weight:600;'>● ALL ACTIVE</div>
    </div>
    """)

    # Section 4: Search Interface
    st.html("<div style='font-size:11px; font-weight:700; color:#8B98A8; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>SEARCH ASSETS (ANY TICKER OR COMPANY)</div>")
    search_query = st.text_input(
        "Search assets...",
        placeholder="🔍 Type ticker or company name (e.g. RELIANCE, TCS, TATAMOTORS, AAPL, NVDA)...",
        label_visibility="collapsed",
        key=f"search_input_{market}"
    )
    
    # Display Search Results if user searched
    if search_query and search_query.strip():
        search_matches = search_universe(universe_df, search_query.strip(), limit=12)
        total_found = len(search_universe(universe_df, search_query.strip(), limit=500))
        st.html(f"<div style='font-size:10px; font-weight:700; color:#2997FF; text-transform:uppercase; letter-spacing:0.8px; margin-top:6px; margin-bottom:4px;'>SEARCH RESULTS ({min(12, total_found)} of {total_found} matches)</div>")
        
        if len(search_matches) == 0:
            st.html(f"<div style='font-size:12px; color:#FF4D5A; padding:6px 0;'>No matching stocks found for '{search_query}'. Try searching by symbol or company name.</div>")
        else:
            for _, row in search_matches.iterrows():
                t_sym = row["ticker"]
                t_name = str(row["name"])[:32]
                t_exch = str(row["exchange"])
                t_pr = float(row["price"])
                is_already_added = t_sym in selected_tickers
                
                sr_col1, sr_col2 = st.columns([8.2, 3.8], gap="small")
                with sr_col1:
                    st.html(f"""
                    <div style='padding:5px 0;'>
                        <div style='font-family:JetBrains Mono, monospace; font-size:12px; font-weight:700; color:#F1F5F9;'>
                            {t_sym} <span style='font-weight:400; font-size:11px; color:#8B98A8;'>[{t_exch}] · {currency_symbol}{t_pr:,.2f}</span>
                        </div>
                        <div style='font-size:11px; color:#8B98A8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>
                            {t_name}
                        </div>
                    </div>
                    """)
                with sr_col2:
                    if is_already_added:
                        st.button("✓ ADDED", key=f"btn_added_{t_sym}", disabled=True, width="stretch")
                    else:
                        if st.button("+ ADD", key=f"btn_add_{t_sym}", type="primary", width="stretch"):
                            if t_sym not in selected_tickers and len(selected_tickers) < 20:
                                selected_tickers.append(t_sym)
                                st.session_state[selected_tickers_key] = selected_tickers
                                st.rerun()

    # Quick Browse / Add Dropdown across entire universe
    avail_to_browse = [t for t in universe_lookup.keys() if t not in selected_tickers]
    if avail_to_browse:
        def format_browse_option(ticker: str) -> str:
            r = universe_lookup.get(ticker)
            if r is not None:
                return f"{ticker} — {str(r['name'])[:28]} [{r['exchange']}]"
            return ticker
            
        b_col1, b_col2 = st.columns([8.2, 3.8], gap="small")
        with b_col1:
            picked_from_browse = st.selectbox(
                "Quick Select",
                options=[f"-- Or browse all {len(avail_to_browse):,} available stocks --"] + avail_to_browse,
                format_func=format_browse_option,
                label_visibility="collapsed",
                key=f"browse_box_{market}"
            )
        with b_col2:
            is_valid_pick = picked_from_browse and not picked_from_browse.startswith("--")
            if st.button("+ Add Asset", key=f"btn_add_pick_{market}", disabled=not is_valid_pick, width="stretch"):
                if picked_from_browse not in selected_tickers and len(selected_tickers) < 20:
                    selected_tickers.append(picked_from_browse)
                    st.session_state[selected_tickers_key] = selected_tickers
                    st.rerun()

    # Section 5 & 6: Dedicated SELECTED ASSETS Section
    st.html(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:14px; margin-bottom:8px; border-top:1px solid rgba(255,255,255,0.06); padding-top:10px;'>
        <div style='font-size:11px; font-weight:700; color:#8B98A8; text-transform:uppercase; letter-spacing:0.8px;'>
            SELECTED ASSETS
        </div>
        <span class='qb-badge' style='background:#16202E; color:#2997FF; font-size:11px; font-weight:700;'>
            {n_assets}
        </span>
    </div>
    """)
    
    if n_assets == 0:
        st.html("<div style='font-size:12px; color:#FF4D5A; padding:8px 0;'>⚠️ No assets selected yet. Search above to add stocks to your portfolio.</div>")
    else:
        # Display each selected asset as a clean, compact terminal row
        for ticker in selected_tickers:
            meta = universe_lookup.get(ticker, {})
            c_name = meta.get("name", ticker)[:24]
            exch = meta.get("exchange", "")
            pr = meta.get("price", 0.0)
            
            # Allocation weight to display
            saved_w = st.session_state.get("pb_weights", {}).get(ticker)
            if saved_w is not None and float(saved_w) > 0:
                w_str = f"{float(saved_w):.1f}%"
            else:
                w_str = "-- %"
                
            r_c1, r_c2, r_c3 = st.columns([7.8, 2.7, 1.5], gap="small")
            with r_c1:
                st.html(f"""
                <div style='padding:4px 0;'>
                    <div style='font-family:JetBrains Mono, monospace; font-size:12px; font-weight:700; color:#F1F5F9;'>
                        <span style='color:#18C7B5; font-size:9px; margin-right:4px;'>●</span>{ticker}
                        <span style='font-weight:400; font-size:11px; color:#8B98A8;'>[{exch}]</span>
                    </div>
                    <div style='font-size:11px; color:#8B98A8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>
                        {c_name} · {currency_symbol}{pr:,.2f}
                    </div>
                </div>
                """)
            with r_c2:
                st.html(f"""
                <div style='text-align:right; font-family:JetBrains Mono, monospace; font-size:13px; font-weight:700; color:#2997FF; padding-top:6px;'>
                    {w_str}
                </div>
                """)
            with r_c3:
                if st.button("×", key=f"btn_rm_{ticker}_{market}", help=f"Remove {ticker}", width="stretch"):
                    selected_tickers.remove(ticker)
                    st.session_state[selected_tickers_key] = selected_tickers
                    if ticker in st.session_state.get("pb_weights", {}):
                        del st.session_state["pb_weights"][ticker]
                    if ticker in st.session_state.get("pb_investments", {}):
                        del st.session_state["pb_investments"][ticker]
                    if f"input_w_{ticker}_{market}" in st.session_state:
                        del st.session_state[f"input_w_{ticker}_{market}"]
                    if f"input_inv_{ticker}_{market}" in st.session_state:
                        del st.session_state[f"input_inv_{ticker}_{market}"]
                    st.rerun()

    st.html("</div>")

# -------------------------------------------------------
# RIGHT COLUMN: STEP 2 — CONFIGURE PORTFOLIO
# -------------------------------------------------------
with col_right:
    st.html(f"""
    <div class='qb-card'>
        <div class='qb-card-header'>
            <div>
                <span style='color:#9B6CFF; font-family:monospace; font-weight:800; font-size:15px; margin-right:6px;'>02</span>
                <span style='color:#F1F5F9; font-size:14px; font-weight:700;'>PORTFOLIO CONFIGURATION</span>
                <div style='font-size:11px; color:#8B98A8; font-weight:400; text-transform:none; margin-top:2px;'>Specify capital valuation and allocation methodology.</div>
            </div>
            <div class='qb-badge qb-badge-info'>
                {currency_code} ({currency_symbol})
            </div>
        </div>
    """)
    
    # Section 11: Portfolio Value Input
    st.html(f"<div style='font-size:11px; font-weight:700; color:#8B98A8; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>TOTAL PORTFOLIO VALUE ({currency_symbol})</div>")
    
    total_portfolio_value = st.number_input(
        f"Total Portfolio Value ({currency_symbol})",
        min_value=1.0,
        max_value=10_000_000_000.0,
        value=default_total_val,
        step=50_000.0 if is_india else 10_000.0,
        format="%.2f",
        label_visibility="collapsed",
        key=f"num_total_val_{market}"
    )
    if is_india:
        st.session_state["pb_total_value_india"] = total_portfolio_value
    elif is_us:
        st.session_state["pb_total_value_us"] = total_portfolio_value
    else:
        st.session_state["pb_total_value_global"] = total_portfolio_value

    # Compact Value Presets
    val_cols = st.columns(4, gap="small")
    if is_india:
        if val_cols[0].button("₹50K", key="val_in_50k", width="stretch", help="Set total to ₹50,000"):
            st.session_state["pb_total_value_india"] = 50_000.0
            st.rerun()
        if val_cols[1].button("₹1 Lakh", key="val_in_1l", width="stretch", help="Set total to ₹1,00,000"):
            st.session_state["pb_total_value_india"] = 100_000.0
            st.rerun()
        if val_cols[2].button("₹10 Lakh", key="val_in_10l", width="stretch", help="Set total to ₹10,00,000"):
            st.session_state["pb_total_value_india"] = 1_000_000.0
            st.rerun()
        if val_cols[3].button("₹1 Crore", key="val_in_1cr", width="stretch", help="Set total to ₹1,00,00,000"):
            st.session_state["pb_total_value_india"] = 10_000_000.0
            st.rerun()
    else:
        if val_cols[0].button("$10K", key="val_us_10k", width="stretch", help="Set total to $10,000"):
            st.session_state["pb_total_value_us"] = 10_000.0
            st.rerun()
        if val_cols[1].button("$50K", key="val_us_50k", width="stretch", help="Set total to $50,000"):
            st.session_state["pb_total_value_us"] = 50_000.0
            st.rerun()
        if val_cols[2].button("$100K", key="val_us_100k", width="stretch", help="Set total to $100,000"):
            st.session_state["pb_total_value_us"] = 100_000.0
            st.rerun()
        if val_cols[3].button("$1M", key="val_us_1m", width="stretch", help="Set total to $1,000,000"):
            st.session_state["pb_total_value_us"] = 1_000_000.0
            st.rerun()

    # Section 12: Allocation Method
    st.html("<div style='margin-top:12px; font-size:11px; font-weight:700; color:#8B98A8; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>ALLOCATION METHOD</div>")
    alloc_mode = st.selectbox(
        "Allocation Method",
        ["Portfolio Weight", "Investment Amount"],
        index=0 if st.session_state["pb_alloc_mode"] == "Portfolio Weight" else 1,
        label_visibility="collapsed",
        key="alloc_mode_select"
    )
    st.session_state["pb_alloc_mode"] = alloc_mode
    
    if alloc_mode == "Portfolio Weight":
        st.html("<div style='font-size:11px; color:#8B98A8; margin-top:-4px; margin-bottom:10px;'>Portfolio Weight: Define the percentage exposure for each selected asset (must sum to 100%).</div>")
    else:
        st.html(f"<div style='font-size:11px; color:#8B98A8; margin-top:-4px; margin-bottom:10px;'>Investment Amount: Define the cash capital allocated to each selected asset (must equal {currency_symbol}{total_portfolio_value:,.2f}).</div>")

    # Allocation Controls per Holding
    weights_input: dict[str, float] = {}
    investment_input: dict[str, float] = {}

    if n_assets == 0:
        st.caption("No assets selected. Add assets on the left to configure allocation.")
    else:
        curr_saved_w = st.session_state.get("pb_weights", {})
        equal_share = round(100.0 / n_assets, 2)
        equal_inv = round(total_portfolio_value / n_assets, 2)
        
        # Equal Weights Helper Button
        eq_c1, eq_c2 = st.columns([5, 7], gap="small")
        with eq_c1:
            if alloc_mode == "Portfolio Weight":
                if st.button("⚖️ Set Equal Weights", key="btn_eq_w", width="stretch"):
                    eq_w = calculate_equal_weights(selected_tickers)
                    set_portfolio_weights(eq_w, market)
                    st.rerun()
            else:
                if st.button("⚖️ Set Equal Cash", key="btn_eq_inv", width="stretch"):
                    eq_inv = calculate_equal_investments(selected_tickers, total_portfolio_value)
                    set_portfolio_investments(eq_inv, market)
                    st.rerun()

        # Display input rows per asset
        for ticker in selected_tickers:
            c_name = universe_lookup[ticker]["name"][:20]
            
            if alloc_mode == "Portfolio Weight":
                k_w = f"input_w_{ticker}_{market}"
                if k_w not in st.session_state:
                    st.session_state[k_w] = float(curr_saved_w.get(ticker, equal_share))
                c_lbl, c_inp = st.columns([6, 6], gap="small")
                with c_lbl:
                    st.markdown(f"**{ticker}**")
                    st.caption(f"{c_name}")
                with c_inp:
                    w_val = st.number_input(
                        f"{ticker} Weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=5.0,
                        key=k_w,
                        label_visibility="collapsed"
                    )
                    weights_input[ticker] = w_val
                    st.session_state["pb_weights"][ticker] = w_val
                    calc_inv = (w_val / 100.0) * total_portfolio_value
                    st.caption(f"Val: {currency_symbol}{calc_inv:,.0f}")
            else:
                k_inv = f"input_inv_{ticker}_{market}"
                if k_inv not in st.session_state:
                    st.session_state[k_inv] = float(st.session_state.get("pb_investments", {}).get(ticker, equal_inv))
                c_lbl, c_inp = st.columns([6, 6], gap="small")
                with c_lbl:
                    st.markdown(f"**{ticker}**")
                    st.caption(f"{c_name}")
                with c_inp:
                    inv_val = st.number_input(
                        f"{ticker} Amount ({currency_symbol})",
                        min_value=0.0,
                        max_value=total_portfolio_value * 2.0,
                        step=10_000.0 if is_india else 1_000.0,
                        key=k_inv,
                        label_visibility="collapsed"
                    )
                    investment_input[ticker] = inv_val
                    st.session_state["pb_investments"][ticker] = inv_val
                    calc_w = (inv_val / total_portfolio_value * 100.0) if total_portfolio_value > 0 else 0.0
                    st.caption(f"Weight: {calc_w:.2f}%")

    st.html("</div>")

# =======================================================
# 7. VALIDATE PORTFOLIO
# =======================================================
validation_res: PortfolioValidationResult = validate_custom_portfolio(
    selected_tickers=selected_tickers,
    total_portfolio_value=total_portfolio_value,
    allocation_mode=alloc_mode,
    weights_input=weights_input,
    investment_input=investment_input,
    data_availability=data_availability,
    known_universe_tickers=all_universe_tickers
)

# Section 13: Allocation Status Strip directly below configuration
strip_c1, strip_c2, strip_c3, strip_c4 = st.columns([3, 3, 3, 3], gap="small")

with strip_c1:
    alloc_status_text = "● ALLOCATION VALID" if abs(validation_res.total_allocation_pct - 100.0) <= 0.05 else ("⚠ ALLOCATION EXCEEDS 100%" if validation_res.total_allocation_pct > 100.05 else "● ALLOCATION INCOMPLETE")
    alloc_status_color = "#3fb950" if abs(validation_res.total_allocation_pct - 100.0) <= 0.05 else ("#FF4D5A" if validation_res.total_allocation_pct > 100.05 else "#f59e0b")
    st.html(f"""
    <div class='qb-card' style='padding:10px 14px; margin-bottom:12px;'>
        <div style='font-size:11px; color:#8B98A8; text-transform:uppercase;'>TOTAL ALLOCATION</div>
        <div style='font-size:18px; font-weight:700; color:{alloc_status_color}; font-family:JetBrains Mono, monospace;'>
            {validation_res.total_allocation_pct:.2f}%
        </div>
        <div style='font-size:10px; color:{alloc_status_color}; font-weight:600;'>{alloc_status_text}</div>
    </div>
    """)

with strip_c2:
    st.html(f"""
    <div class='qb-card' style='padding:10px 14px; margin-bottom:12px;'>
        <div style='font-size:11px; color:#8B98A8; text-transform:uppercase;'>TOTAL INVESTED</div>
        <div style='font-size:18px; font-weight:700; color:#F1F5F9; font-family:JetBrains Mono, monospace;'>
            {currency_symbol}{validation_res.total_investment_val:,.2f}
        </div>
        <div style='font-size:10px; color:#8B98A8;'>Of {currency_symbol}{total_portfolio_value:,.2f}</div>
    </div>
    """)

with strip_c3:
    unallocated_cash = total_portfolio_value - validation_res.total_investment_val
    unalloc_color = "#3fb950" if abs(unallocated_cash) < 1.0 else ("#FF4D5A" if unallocated_cash < 0 else "#f59e0b")
    st.html(f"""
    <div class='qb-card' style='padding:10px 14px; margin-bottom:12px;'>
        <div style='font-size:11px; color:#8B98A8; text-transform:uppercase;'>UNALLOCATED CASH</div>
        <div style='font-size:18px; font-weight:700; color:{unalloc_color}; font-family:JetBrains Mono, monospace;'>
            {currency_symbol}{unallocated_cash:,.2f}
        </div>
        <div style='font-size:10px; color:#8B98A8;'>{"Balanced" if abs(unallocated_cash) < 1.0 else "Requires balancing"}</div>
    </div>
    """)

with strip_c4:
    badge_cls = "qb-badge-valid" if validation_res.is_valid else "qb-badge-invalid"
    st.html(f"""
    <div class='qb-card' style='padding:10px 14px; margin-bottom:12px;'>
        <div style='font-size:11px; color:#8B98A8; text-transform:uppercase;'>PORTFOLIO STATUS</div>
        <div style='margin-top:2px;'>
            <span class='qb-badge {badge_cls}' style='font-size:13px;'>
                ● {validation_res.status}
            </span>
        </div>
        <div style='font-size:10px; color:{"#3fb950" if validation_res.is_valid else "#FF4D5A"};'>
            {"Ready for quantum analysis" if validation_res.is_valid else "Action required"}
        </div>
    </div>
    """)

# Error / Warning Banners
if not validation_res.is_valid:
    for err in validation_res.errors:
        st.html(f"""
        <div style='background:rgba(255, 77, 90, 0.12); border:1px solid rgba(255, 77, 90, 0.35); border-radius:6px; padding:8px 14px; margin-bottom:8px; font-size:12px; color:#FF4D5A;'>
            ⚠️ <b>Portfolio Validation Error:</b> {err}
        </div>
        """)

# =======================================================
# 8. SECTION 9: SELECTED ASSET SUMMARY BAR & 4 KPI CARDS
# =======================================================
st.html(f"""
<div style='background:#101720; border:1px solid rgba(41,151,255,0.25); border-left:3px solid #2997FF; border-radius:6px; padding:10px 16px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>
    <div style='display:flex; align-items:center; gap:16px; flex-wrap:wrap;'>
        <div style='font-size:11px; font-weight:700; color:#2997FF; letter-spacing:0.8px;'>PORTFOLIO UNIVERSE:</div>
        <div style='font-family:JetBrains Mono, monospace; font-size:13px; font-weight:700; color:#F1F5F9;'>
            {n_assets} ASSETS
        </div>
        <div style='font-size:12px; color:#8B98A8;'>
            {market} ({universe_df['exchange'].iloc[0] if len(universe_df) > 0 else 'EXCHANGE'})
        </div>
        <div style='font-family:JetBrains Mono, monospace; font-size:13px; font-weight:700; color:#18C7B5;'>
            {currency_symbol}{total_portfolio_value:,.2f}
        </div>
    </div>
    <div style='font-size:11px; color:#8B98A8; font-family:JetBrains Mono, monospace;'>
        {" • ".join([f"{t} {validation_res.weights.get(t,0)*100:.0f}%" for t in selected_tickers[:4]])}{'...' if n_assets > 4 else ''}
    </div>
</div>
""")

def render_kpi_card(title: str, value: str, subvalue: str, color: str):
    st.html(f"""
    <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 18px; margin-bottom:16px;'>
        <div style='font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size:12px; font-weight:600; color:#8B98A8; text-transform:uppercase; letter-spacing:0.5px;'>
            {title}
        </div>
        <div style='margin-top:8px;'>
            <div style='font-family:JetBrains Mono, monospace; font-size:24px; font-weight:700; color:{color}; letter-spacing:-0.5px;'>
                {value}
            </div>
            <div style='font-family:JetBrains Mono, monospace; font-size:12px; color:#8B98A8; margin-top:2px;'>
                {subvalue}
            </div>
        </div>
    </div>
    """)

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4, gap="small")

with kpi_c1:
    render_kpi_card(
        "TOTAL PORTFOLIO",
        f"{currency_symbol}{total_portfolio_value:,.0f}",
        f"Currency: {currency_code}",
        "#F1F5F9"
    )

with kpi_c2:
    render_kpi_card(
        "NUMBER OF ASSETS",
        f"{n_assets}",
        f"{'Equal weighted' if len(set(validation_res.weights.values())) <= 1 and n_assets > 0 else 'Custom weighted'}",
        "#2997FF"
    )

with kpi_c3:
    render_kpi_card(
        "MARKET",
        f"{market}",
        f"{universe_df['exchange'].iloc[0] if len(universe_df) > 0 else 'Multi-Exchange'} Universe",
        "#18C7B5"
    )

with kpi_c4:
    top_ticker = "None"
    top_w_pct = 0.0
    if validation_res.weights:
        top_ticker, top_dec = max(validation_res.weights.items(), key=lambda x: x[1])
        top_w_pct = top_dec * 100.0
    render_kpi_card(
        "TOP ALLOCATION",
        f"{top_ticker}",
        f"{top_w_pct:.1f}% Weight",
        "#9B6CFF"
    )

# =======================================================
# 9. PORTFOLIO TABLE & VISUALIZATIONS
# =======================================================
mid_l, mid_r = st.columns([6, 6], gap="medium")

with mid_l:
    # Portfolio Table
    st.html("""
    <div class='qb-card'>
        <div class='qb-card-header'>
            <span>PORTFOLIO HOLDINGS &amp; ALLOCATION TABLE</span>
            <span style='font-size:11px; font-weight:500; color:#8B98A8;'>Multi-Asset Breakdown</span>
        </div>
    """)
    
    if n_assets > 0:
        table_rows = []
        for ticker in selected_tickers:
            meta = universe_lookup.get(ticker, {})
            c_name = meta.get("name", ticker)
            sector = meta.get("sector", "General")
            price = meta.get("price", 0.0)
            inv_amt = validation_res.investment_amounts.get(ticker, 0.0)
            w_pct = validation_res.weights.get(ticker, 0.0) * 100.0
            
            table_rows.append({
                "TICKER": ticker,
                "COMPANY": c_name[:26],
                "SECTOR": sector[:16],
                f"PRICE ({currency_symbol})": f"{price:,.2f}",
                f"INVESTMENT ({currency_symbol})": f"{currency_symbol}{inv_amt:,.2f}",
                "WEIGHT": f"{w_pct:.2f}%"
            })
            
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, hide_index=True, width="stretch")
        
        # Summary row
        st.html(f"""
        <div style='background:#16202E; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px 14px; margin-top:8px; display:flex; justify-content:space-between; font-family:JetBrains Mono, monospace; font-size:12px;'>
            <div><b>TOTAL PORTFOLIO:</b> {currency_symbol}{validation_res.total_investment_val:,.2f}</div>
            <div><b>TOTAL WEIGHT:</b> {validation_res.total_allocation_pct:.2f}%</div>
        </div>
        """)
    else:
        st.caption("No assets selected.")
        
    st.html("</div>")

with mid_r:
    # Visualizations: Donut Chart & Horizontal Bar Chart
    st.html("""
    <div class='qb-card'>
        <div class='qb-card-header'>
            <span>ALLOCATION VISUALIZATION</span>
            <span style='font-size:11px; font-weight:500; color:#8B98A8;'>Weights &amp; Exposure Distribution</span>
        </div>
    """)
    
    if n_assets > 0 and sum(validation_res.weights.values()) > 0:
        chart_tab1, chart_tab2 = st.tabs(["🍩 Donut Chart", "📊 Horizontal Bar Chart"])
        
        color_palette = [
            "#2997FF", "#9B6CFF", "#18C7B5", "#F59E0B",
            "#EC4899", "#10B981", "#3B82F6", "#8B5CF6", "#F97316"
        ]
        
        with chart_tab1:
            labels = list(validation_res.weights.keys())
            values = [validation_res.weights[k] * 100.0 for k in labels]
            
            fig_donut = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.62,
                    textinfo="label+percent",
                    marker=dict(
                        colors=[color_palette[i % len(color_palette)] for i in range(len(labels))],
                        line=dict(color="#101720", width=2)
                    ),
                    textfont=dict(size=11, family="JetBrains Mono, monospace", color="#F1F5F9"),
                    hoverinfo="label+value+percent",
                    hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<br>Allocation: " + currency_symbol + "%{customdata:,.2f}<extra></extra>",
                    customdata=[validation_res.investment_amounts[k] for k in labels]
                )
            ])
            
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{n_assets}</b><br><span style='font-size:10px; color:#8B98A8;'>ASSETS</span>",
                    x=0.5, y=0.5,
                    font_size=15,
                    font_family="JetBrains Mono, monospace",
                    font_color="#F1F5F9",
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})
            
        with chart_tab2:
            sorted_items = sorted(validation_res.weights.items(), key=lambda x: x[1], reverse=False)
            bar_labels = [x[0] for x in sorted_items]
            bar_weights = [x[1] * 100.0 for x in sorted_items]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    y=bar_labels,
                    x=bar_weights,
                    orientation="h",
                    marker=dict(
                        color="#2997FF",
                        line=dict(color="rgba(41, 151, 255, 0.4)", width=1)
                    ),
                    text=[f"{w:.1f}%" for w in bar_weights],
                    textposition="auto",
                    textfont=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9")
                )
            ])
            
            bar_layout = get_terminal_layout("Portfolio Allocation (%)", height=240)
            bar_layout["margin"] = dict(l=80, r=20, t=20, b=25)
            bar_layout["xaxis"]["ticksuffix"] = "%"
            fig_bar.update_layout(**bar_layout)
            st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("No valid weights to visualize.")
        
    st.html("</div>")

# =======================================================
# 10. DATA AVAILABILITY STATUS CARDS
# =======================================================
st.html("""
<div class='qb-card'>
    <div class='qb-card-header'>
        <span>4. DATA AVAILABILITY</span>
        <span style='font-size:11px; font-weight:500; color:#8B98A8;'>Historical Time-Series Integrity</span>
    </div>
""")

if not selected_tickers:
    st.caption("Select assets above to inspect historical time series integrity.")
else:
    avail_cols = st.columns(min(n_assets, 4))
    for idx, ticker in enumerate(selected_tickers):
        c_idx = idx % 4
        is_avail = data_availability.get(ticker, False)
        status_msg = data_messages.get(ticker, "Checking status...")
        
        status_icon = "✓" if is_avail else "⚠"
        status_color = "#3fb950" if is_avail else "#FF4D5A"
        bg_color = "rgba(46, 160, 67, 0.1)" if is_avail else "rgba(255, 77, 90, 0.1)"
        border_color = "rgba(46, 160, 67, 0.3)" if is_avail else "rgba(255, 77, 90, 0.3)"
        
        with avail_cols[c_idx]:
            st.html(f"""
            <div style='background:{bg_color}; border:1px solid {border_color}; border-radius:6px; padding:10px 12px; margin-bottom:8px;'>
                <div style='font-family:monospace; font-weight:700; font-size:13px; color:#F1F5F9;'>
                    {ticker}
                </div>
                <div style='font-size:11px; color:{status_color}; margin-top:4px; font-weight:600;'>
                    {status_icon} {status_msg}
                </div>
            </div>
            """)

st.html("</div>")

# =======================================================
# 11. ACTION BUTTON: RUN QUANTUM RISK ANALYSIS
# =======================================================
st.html("<div style='margin-top:8px;'></div>")

cta_col1, cta_col2 = st.columns([8, 4])

with cta_col1:
    btn_disabled = not validation_res.is_valid
    run_btn = st.button(
        "▶ RUN QUANTUM RISK ANALYSIS",
        key="btn_run_quantum_portfolio",
        type="primary",
        disabled=btn_disabled,
        width="stretch"
    )

with cta_col2:
    if st.button("⚛ Open Quantum Risk Engine", key="btn_open_qengine", width="stretch"):
        try:
            st.switch_page("pages/4_Quantum_Engine.py")
        except Exception:
            st.info("Navigate to 'Quantum Engine' from the sidebar to view quantum tail risk results.")

if run_btn:
    with st.spinner("Processing custom portfolio and synchronizing quantum engine state..."):
        try:
            # 1. Align prices DataFrame with selected weights
            ordered_weights = np.array([validation_res.weights[t] for t in selected_tickers])
            aligned_prices = price_df[selected_tickers].copy()
            
            # 2. Build official Portfolio object
            custom_portfolio = Portfolio(
                prices=aligned_prices,
                weights=ordered_weights,
                initial_value=total_portfolio_value
            )
            
            # 3. Calculate portfolio returns and loss distribution
            port_returns = custom_portfolio.portfolio_returns
            port_losses = -port_returns
            
            # 4. Construct comprehensive session state custom_portfolio dictionary
            custom_portfolio_data = {
                "market": market,
                "total_value": total_portfolio_value,
                "currency": currency_code,
                "currency_symbol": currency_symbol,
                "assets": selected_tickers,
                "weights": validation_res.weights,
                "investment_amounts": validation_res.investment_amounts,
                "portfolio_returns": port_returns,
                "portfolio_losses": port_losses,
                "tickers": selected_tickers,
                "company_names": {t: universe_lookup[t]["name"] for t in selected_tickers if t in universe_lookup}
            }
            
            # 5. Commit to Streamlit session state
            st.session_state["custom_portfolio"] = custom_portfolio_data
            st.session_state["portfolio"] = custom_portfolio
            st.session_state["prices_df"] = aligned_prices
            st.session_state["portfolio_val"] = total_portfolio_value
            st.session_state["portfolio_currency"] = currency_code
            st.session_state["is_custom_portfolio"] = True
            st.session_state["global_data_mode"] = "Custom Portfolio (Builder)"
            st.session_state["_sync_sidebar_val"] = True
            st.session_state["_last_prices_shape"] = aligned_prices.shape
            
            # Invalidate any old Monte Carlo simulation caches
            for k in list(st.session_state.keys()):
                if "mc_sim" in k or "mc_cache" in k or "page4" in k:
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass
            
            st.success(f"✓ Custom {market} portfolio successfully compiled! {n_assets} assets ready for quantum risk estimation.")
            
            st.html(f"""
            <div style='background:rgba(41, 151, 255, 0.12); border:1px solid rgba(41, 151, 255, 0.35); border-radius:6px; padding:12px 16px; margin-top:10px;'>
                <div style='font-size:13px; font-weight:700; color:#2997FF;'>
                    🚀 PORTFOLIO READY FOR QUANTUM TAIL RISK ANALYSIS
                </div>
                <div style='font-size:12px; color:#F1F5F9; margin-top:4px;'>
                    Allocations ({', '.join([f'{t}: {validation_res.weights[t]*100:.1f}%' for t in selected_tickers])}) and empirical return distributions are synchronized.
                </div>
            </div>
            """)
            
            try:
                st.switch_page("pages/4_Quantum_Engine.py")
            except Exception:
                pass
                
        except Exception as e:
            st.error(f"Failed to prepare quantum portfolio: {e}")

# =======================================================
# 12. HISTORICAL PRICE TRAJECTORIES & CORRELATION EXPANDER
# =======================================================
if n_assets > 0 and len(price_df) > 0:
    with st.expander("📈 Historical Price Trajectories & Correlation Structure", expanded=False):
        exp_col1, exp_col2 = st.columns([7, 5], gap="medium")
        
        with exp_col1:
            st.caption("Normalized Asset Performance (Base 100)")
            norm_prices = (price_df[selected_tickers] / price_df[selected_tickers].iloc[0]) * 100.0
            
            fig_perf = go.Figure()
            palette = ["#2997FF", "#9B6CFF", "#18C7B5", "#F59E0B", "#EC4899", "#10B981"]
            for idx, col in enumerate(selected_tickers):
                fig_perf.add_trace(go.Scatter(
                    x=norm_prices.index,
                    y=norm_prices[col],
                    mode="lines",
                    name=col,
                    line=dict(color=palette[idx % len(palette)], width=1.8),
                    hovertemplate=f"<b>{col}:</b> %{{y:.2f}}<extra></extra>"
                ))
            layout_perf = get_terminal_layout("", height=220)
            layout_perf["margin"] = dict(l=35, r=15, t=10, b=25)
            fig_perf.update_layout(**layout_perf)
            st.plotly_chart(fig_perf, width="stretch", config={"displayModeBar": False})
            
        with exp_col2:
            st.caption("Asset Correlation Matrix")
            returns_subset = np.log(price_df[selected_tickers] / price_df[selected_tickers].shift(1)).dropna()
            corr_mat = returns_subset.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_mat.values,
                x=selected_tickers,
                y=selected_tickers,
                colorscale="Viridis",
                zmin=-1.0,
                zmax=1.0,
                text=np.round(corr_mat.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10, "family": "monospace"}
            ))
            corr_layout = get_terminal_layout("", height=220)
            corr_layout["margin"] = dict(l=40, r=20, t=10, b=30)
            fig_corr.update_layout(**corr_layout)
            st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})
