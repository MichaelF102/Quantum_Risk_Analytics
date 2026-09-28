"""
Page 4: Quantum Risk Terminal - State Preparation, Risk Oracle, and Quantum Amplitude Estimation (QAE)
Redesigned to a premium dark quantitative-finance / quantum-computing research terminal.
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

st.set_page_config(
    page_title="Quantum Risk Terminal",
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
    from app.components.metrics import render_terminal_kpi
    from app.components.explanations import explain_amplitude_estimation
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.charts import get_terminal_layout
    from components.metrics import render_terminal_kpi
    from components.explanations import explain_amplitude_estimation

from src.classical.monte_carlo import MonteCarloEngine
from src.quantum.state_preparation import LossDistributionDiscretizer, create_state_preparation_circuit
from src.quantum.risk_oracle import build_risk_oracle_circuit, build_full_risk_operator, build_grover_operator
from src.quantum.amplitude_estimation import QuantumAmplitudeEstimator
from src.quantum.quantum_var import QuantumVaREstimator
from src.quantum.quantum_cvar import QuantumCVaREstimator

initialize_app_theme()
inputs = get_shared_sidebar_inputs()
portfolio = inputs["portfolio"]
global_conf_level = inputs["conf_level"]
time_horizon = inputs["time_horizon"]
port_val = inputs["portfolio_val"]
backend_choice = inputs["backend_choice"]
seed = inputs["seed"]

# Custom Terminal CSS Styling
st.html("""
<style>
    /* Dark Terminal Layout */
    .stApp {
        background-color: #070B12 !important;
    }
    
    /* Maximize container resolution and eliminate cramped layout */
    .main .block-container {
        max-width: 98% !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    
    /* Typography sizing to prevent overflow */
    h1 {
        font-size: 22px !important;
        white-space: nowrap !important;
    }
    div[data-baseweb="select"] {
        font-size: 13px !important;
    }
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        font-size: 11px !important;
        color: #8B98A8 !important;
    }
    
    /* Terminal Card */
    .q-card {
        background-color: #101720;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    
    /* Status Badge */
    .q-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
</style>
""")

# =======================================================
# 1. TOP HEADER & STATUS STRIP
# =======================================================
header_left, header_right = st.columns([5.2, 6.8], gap="medium")

# Session state defaults for top controls
if "q_num_qubits" not in st.session_state:
    st.session_state["q_num_qubits"] = 4
if "q_algo" not in st.session_state:
    st.session_state["q_algo"] = "Maximum Likelihood QAE"
if "q_conf_level" not in st.session_state:
    st.session_state["q_conf_level"] = 0.95
if "q_shots" not in st.session_state:
    st.session_state["q_shots"] = 10000
if "q_iterations" not in st.session_state:
    st.session_state["q_iterations"] = 20

with header_left:
    st.html(f"""
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>
        <div style='font-size:30px; color:#2997FF; line-height:1;'>⚛</div>
        <div>
            <div style='display:flex; align-items:center; gap:8px; flex-wrap:wrap;'>
                <div style='font-size:21px; font-weight:800; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; letter-spacing:-0.5px; white-space:nowrap;'>
                    QUANTUM RISK TERMINAL
                </div>
                <div class='q-badge' style='background:rgba(46,160,67,0.15); border:1px solid rgba(46,160,67,0.4); color:#3fb950; white-space:nowrap;'>
                    ● SIMULATED
                </div>
                <div class='q-badge' style='background:rgba(41,151,255,0.12); border:1px solid rgba(41,151,255,0.35); color:#2997FF; white-space:nowrap;'>
                    QISKIT AER
                </div>
                <div class='q-badge' style='background:rgba(155,108,255,0.12); border:1px solid rgba(155,108,255,0.35); color:#9B6CFF; white-space:nowrap;'>
                    {st.session_state['q_num_qubits']} QUBITS
                </div>
                <div class='q-badge' style='background:rgba(24,199,181,0.12); border:1px solid rgba(24,199,181,0.35); color:#18C7B5; white-space:nowrap;'>
                    {2**st.session_state['q_num_qubits']} STATES
                </div>
            </div>
            <div style='font-size:12px; color:#8B98A8; margin-top:2px; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                Quantum-Enhanced Tail Risk Analytics &amp; Amplitude Estimation
            </div>
        </div>
    </div>
    """)

with header_right:
    # Quick controls at top right with generous spacing
    h_col1, h_col2, h_col3, h_col4 = st.columns([3.6, 2.2, 2.2, 3.8], gap="small")
    with h_col1:
        st.caption("Algorithm")
        top_algo = st.selectbox(
            "Algorithm",
            ["Maximum Likelihood QAE", "Exact Statevector", "Canonical QAE"],
            index=0 if "MLQAE" in st.session_state["q_algo"] or "Maximum" in st.session_state["q_algo"] else (1 if "Exact" in st.session_state["q_algo"] else 2),
            label_visibility="collapsed",
            key="top_algo_sel"
        )
        st.session_state["q_algo"] = top_algo
    with h_col2:
        st.caption("Confidence")
        top_conf = st.selectbox(
            "Confidence Level",
            ["90%", "95%", "99%"],
            index=1,
            label_visibility="collapsed",
            key="top_conf_sel"
        )
        st.session_state["q_conf_level"] = float(top_conf.replace("%", "")) / 100.0
    with h_col3:
        st.caption("Shots")
        top_shots = st.selectbox(
            "Shots",
            ["1,000", "5,000", "10,000", "20,000"],
            index=2,
            label_visibility="collapsed",
            key="top_shots_sel"
        )
        st.session_state["q_shots"] = int(top_shots.replace(",", ""))
    with h_col4:
        st.caption("Execution")
        if st.button("▶ Run Analysis", key="btn_run_top", type="primary", use_container_width=True):
            st.rerun()

num_qubits = st.session_state["q_num_qubits"]
conf_level = st.session_state["q_conf_level"]
algo_choice = st.session_state["q_algo"]
shots_count = st.session_state["q_shots"]

# =======================================================
# 2. BACKEND CALCULATIONS (Preserved Functionality)
# =======================================================
# 1. Classical Monte Carlo simulation to generate empirical loss distribution
mc_engine = MonteCarloEngine(
    initial_portfolio_value=port_val,
    weights=portfolio.weights,
    expected_returns=portfolio.returns.mean().values * 252,
    covariance_matrix=portfolio.covariance_matrix,
    asset_prices=portfolio.prices.iloc[-1].values,
    seed=seed
)
sim_res = mc_engine.run_simulation(n_simulations=10_000, time_horizon_days=time_horizon)
losses = sim_res["losses_pct"]

# 2. Build Discretizer with user-specified qubits
discretizer = LossDistributionDiscretizer(losses, num_qubits=num_qubits)

# 3. Candidate Threshold T calculation
default_thresh = float(np.percentile(losses, conf_level * 100.0))
if "page4_cand_threshold" not in st.session_state:
    st.session_state["page4_cand_threshold"] = float(default_thresh * 100.0)

# Clamp threshold within discretizer bounds
min_loss_pct = float(discretizer.bin_edges[0] * 100.0)
max_loss_pct = float(discretizer.bin_edges[-1] * 100.0)
cand_threshold = max(min_loss_pct, min(max_loss_pct, float(st.session_state["page4_cand_threshold"]))) / 100.0
k_thresh = discretizer.value_to_bin_index(cand_threshold)
ref_tail_prob = discretizer.get_classical_tail_probability(cand_threshold)

# 4. Run Quantum Amplitude Estimation Engine
qae = QuantumAmplitudeEstimator(discretizer, backend_name=backend_choice, seed=seed)
if "MLQAE" in algo_choice or "Maximum Likelihood" in algo_choice:
    qae_res = qae.run_mlqae(cand_threshold, shots_per_circuit=shots_count)
elif "Exact" in algo_choice:
    qae_res = qae.run_statevector_exact(cand_threshold)
else:
    qae_res = qae.run_canonical_qae(cand_threshold, num_eval_qubits=4, shots=shots_count)

# 5. Run Quantum VaR and Quantum CVaR Estimators
q_var_estimator = QuantumVaREstimator(discretizer, seed=seed)
q_cvar_estimator = QuantumCVaREstimator(discretizer, seed=seed)

q_var_res = q_var_estimator.estimate_var(confidence_level=conf_level, method="mlqae", portfolio_value=port_val)
q_cvar_res = q_cvar_estimator.estimate_cvar(
    confidence_level=conf_level,
    quantum_var_pct=q_var_res["quantum_var_pct"],
    method="mlqae",
    portfolio_value=port_val
)

currency_sym = "₹" if st.session_state.get("portfolio_currency") == "INR" or st.session_state.get("custom_portfolio", {}).get("currency") in ["INR", "₹"] else "$"

if st.session_state.get("custom_portfolio"):
    cp = st.session_state["custom_portfolio"]
    mkt = cp.get("market", "CUSTOM")
    tot_val = cp.get("total_value", port_val)
    n_assets = len(cp.get("assets", []))
    w_items = [f"{t} {w*100:.0f}%" for t, w in list(cp.get("weights", {}).items())[:6]]
    w_str = " • ".join(w_items)
    if len(cp.get("weights", {})) > 6:
        w_str += "..."
    st.html(f"""
    <div class='q-card' style='display:flex; justify-content:space-between; align-items:center; border-left:3px solid #2997FF; padding:10px 16px; margin-bottom:14px; background:#0E1622; flex-wrap:wrap; gap:10px;'>
        <div style='display:flex; align-items:center; gap:14px; flex-wrap:wrap;'>
            <div style='display:flex; align-items:center; gap:6px;'>
                <span style='color:#2997FF; font-weight:700; font-size:11px; font-family:monospace;'>PORTFOLIO:</span>
                <span class='q-badge' style='background:rgba(41,151,255,0.15); color:#2997FF; border:1px solid rgba(41,151,255,0.4);'>{n_assets} ASSETS</span>
            </div>
            <div style='font-size:13px; font-weight:700; color:#F1F5F9; font-family:monospace;'>
                {currency_sym}{tot_val:,.0f}
            </div>
            <div style='font-size:12px; color:#8B98A8; font-family:monospace;'>
                {w_str}
            </div>
        </div>
        <div style='display:flex; align-items:center; gap:8px;'>
            <div class='q-badge' style='background:rgba(24,199,181,0.15); color:#18C7B5; border:1px solid rgba(24,199,181,0.4);'>
                ● {mkt} UNIVERSE LOADED
            </div>
        </div>
    </div>
    """)

# =======================================================
# 3. TOP KPI STRIP (4 Large Cards matching screenshot)
# =======================================================
def render_terminal_kpi_card(title: str, value: str, subvalue: str, theme_color: str, spark_color: str, icon_svg: str):
    card_html = f"""
    <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 18px; margin-bottom:18px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div style='display:flex; align-items:center; gap:6px;'>
                {icon_svg}
                <div style='font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size:12px; font-weight:600; color:{theme_color}; text-transform:uppercase; letter-spacing:0.5px;'>
                    {title}
                </div>
            </div>
        </div>
        <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-top:10px;'>
            <div>
                <div style='font-family:JetBrains Mono, monospace; font-size:26px; font-weight:700; color:#F1F5F9; letter-spacing:-0.5px;'>
                    {value}
                </div>
                <div style='font-family:JetBrains Mono, monospace; font-size:12px; color:#8B98A8; margin-top:2px;'>
                    {subvalue}
                </div>
            </div>
            <svg width='80' height='28' viewBox='0 0 80 28' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path d='M2 20 L15 16 L28 22 L42 12 L55 17 L68 6 L78 4' stroke='{spark_color}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>
                <path d='M2 20 L15 16 L28 22 L42 12 L55 17 L68 6 L78 4 V28 H2 Z' fill='{spark_color}' fill-opacity='0.12'/>
            </svg>
        </div>
    </div>
    """
    st.html(card_html)

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4, gap="small")

with kpi_c1:
    icon_qvar = "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='#2997FF' stroke-width='2'><path d='M3 3v18h18'/><path d='M19 9l-5 5-4-4-3 3'/></svg>"
    render_terminal_kpi_card(
        f"QUANTUM VaR ({int(conf_level*100)}%)",
        f"{q_var_res['quantum_var_pct']*100:.2f}%",
        f"-{currency_sym}{q_var_res['quantum_var_cash']:,.0f}",
        "#2997FF",
        "#2997FF",
        icon_qvar
    )

with kpi_c2:
    icon_cvar = "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='#8B98A8' stroke-width='2'><path d='M18 20V10'/><path d='M12 20V4'/><path d='M6 20v-6'/></svg>"
    render_terminal_kpi_card(
        f"CLASSICAL VaR ({int(conf_level*100)}%)",
        f"{q_var_res['classical_reference_var_pct']*100:.2f}%",
        f"-{currency_sym}{q_var_res['classical_reference_var_cash']:,.0f}",
        "#8B98A8",
        "#8B98A8",
        icon_cvar
    )

with kpi_c3:
    icon_qcvar = "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='#9B6CFF' stroke-width='2'><path d='M12 2v20'/><path d='M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6'/></svg>"
    render_terminal_kpi_card(
        f"QUANTUM CVaR ({int(conf_level*100)}%)",
        f"{q_cvar_res['quantum_cvar_pct']*100:.2f}%",
        f"-{currency_sym}{q_cvar_res['quantum_cvar_cash']:,.0f}",
        "#9B6CFF",
        "#9B6CFF",
        icon_qcvar
    )

with kpi_c4:
    icon_tail = "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='#FF4D5A' stroke-width='2'><path d='M12 2v4'/><path d='m4.93 4.93 2.83 2.83'/><path d='M2 12h4'/><path d='m4.93 19.07 2.83-2.83'/><path d='M12 18v4'/><path d='m19.07 19.07-2.83-2.83'/><path d='M18 12h4'/><path d='m19.07 4.93-2.83 2.83'/></svg>"
    render_terminal_kpi_card(
        "TAIL PROBABILITY P(L ≥ T)",
        f"{qae_res['estimated_probability']*100:.2f}%",
        f"T = {cand_threshold*100:.2f}%",
        "#FF4D5A",
        "#FF4D5A",
        icon_tail
    )

# =======================================================
# 4. MAIN WORKSPACE: LEFT CONTROL PANEL + RIGHT TAB PANEL
# =======================================================
ctrl_col, main_col = st.columns([2.2, 9.8], gap="medium")

# LEFT CONTROL PANEL (Simulation Settings)
with ctrl_col:
    st.html("""
    <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px;'>
        <div style='display:flex; align-items:center; gap:8px; font-size:13px; font-weight:700; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; text-transform:uppercase; letter-spacing:0.5px;'>
            ⚙ Simulation Settings
        </div>
    </div>
    """)
    
    with st.container():
        # Number of Qubits
        sel_qubits = st.selectbox(
            "Number of Qubits",
            [3, 4, 5, 6],
            index=[3, 4, 5, 6].index(num_qubits) if num_qubits in [3, 4, 5, 6] else 1,
            key="ctrl_qubits_sel"
        )
        if sel_qubits != num_qubits:
            st.session_state["q_num_qubits"] = sel_qubits
            st.rerun()

        # Read-only technical metadata boxes
        st.caption("Basis States (2^n)")
        st.text_input("Basis States (2^n)", value=str(2**num_qubits), disabled=True, label_visibility="collapsed")
        
        st.caption("Loss Bins")
        st.text_input("Loss Bins", value=str(discretizer.num_bins), disabled=True, label_visibility="collapsed")
        
        st.caption("Bin Resolution (ΔL)")
        st.text_input("Bin Resolution (ΔL)", value=f"{discretizer.bin_width * 100:.3f}%", disabled=True, label_visibility="collapsed")
        
        st.caption("Loss Range")
        lr_c1, lr_c2 = st.columns(2)
        with lr_c1:
            st.text_input("Min", value=f"{discretizer.lower_bound*100:.2f}%", disabled=True, label_visibility="collapsed")
        with lr_c2:
            st.text_input("Max", value=f"{discretizer.upper_bound*100:.2f}%", disabled=True, label_visibility="collapsed")

    st.html("""
    <div style='margin:14px 0 10px 0; border-top:1px solid rgba(255,255,255,0.08);'></div>
    <div style='display:flex; align-items:center; gap:8px; font-size:13px; font-weight:700; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;'>
        ⚛ QAE Configuration
    </div>
    """)
    
    with st.container():
        # Algorithm
        side_algo = st.selectbox(
            "Algorithm",
            ["Maximum Likelihood QAE", "Exact Statevector Simulation", "Canonical QAE"],
            index=["Maximum Likelihood QAE", "Exact Statevector Simulation", "Canonical QAE"].index(algo_choice) if algo_choice in ["Maximum Likelihood QAE", "Exact Statevector Simulation", "Canonical QAE"] else 0,
            key="side_algo_sel"
        )
        if side_algo != algo_choice:
            st.session_state["q_algo"] = side_algo
            st.rerun()
            
        # Confidence Level
        side_conf = st.selectbox(
            "Confidence Level",
            ["90%", "95%", "99%"],
            index=1 if conf_level == 0.95 else (0 if conf_level == 0.90 else 2),
            key="side_conf_sel"
        )
        val_conf = float(side_conf.replace("%", "")) / 100.0
        if val_conf != conf_level:
            st.session_state["q_conf_level"] = val_conf
            st.rerun()

        # Shots
        side_shots = st.selectbox(
            "Shots",
            ["1,000", "5,000", "10,000", "20,000"],
            index=["1,000", "5,000", "10,000", "20,000"].index(f"{shots_count:,}") if f"{shots_count:,}" in ["1,000", "5,000", "10,000", "20,000"] else 2,
            key="side_shots_sel"
        )
        val_shots = int(side_shots.replace(",", ""))
        if val_shots != shots_count:
            st.session_state["q_shots"] = val_shots
            st.rerun()

        # Iterations
        iterations = st.number_input(
            "Iterations",
            min_value=5,
            max_value=100,
            value=st.session_state["q_iterations"],
            step=5,
            key="side_iterations_input"
        )
        st.session_state["q_iterations"] = iterations
        
        st.html("<div style='margin-top:12px;'></div>")
        if st.button("▶ Run Analysis", key="btn_run_side", type="primary", use_container_width=True):
            st.rerun()
            
        if st.button("↻ Reset to Defaults", key="btn_reset_side", use_container_width=True):
            st.session_state["q_num_qubits"] = 4
            st.session_state["q_algo"] = "Maximum Likelihood QAE"
            st.session_state["q_conf_level"] = 0.95
            st.session_state["q_shots"] = 10000
            st.session_state["q_iterations"] = 20
            st.rerun()

# RIGHT MAIN PANEL: Tabs 1 through 6
with main_col:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Risk Overview",
        "⚛ Quantum State",
        "🎯 Tail Oracle",
        "📊 QAE Results",
        "⚖ Quantum vs Classical",
        "🔬 How It Works"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: RISK OVERVIEW (Matching exact screenshot layout)
    # -------------------------------------------------------------------------
    with tab1:
        tab1_col_chart, tab1_col_metrics = st.columns([6.4, 3.6], gap="medium")
        
        with tab1_col_chart:
            st.html("""
            <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px; margin-bottom:14px;'>
                <div style='display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;'>
                    <span style='color:#2997FF;'>📈</span> Loss Distribution & Tail Threshold
                </div>
            """)
            
            # Loss Distribution & Tail Cutoff Chart
            probs = discretizer.probabilities
            bin_labels = [f"|{i}⟩" for i in range(len(probs))]
            bar_colors = ["#FF4D5A" if i >= k_thresh else "#2997FF" for i in range(len(probs))]
            
            fig_overview = go.Figure()
            # Main Distribution bars
            fig_overview.add_trace(go.Bar(
                x=[bin_labels[i] for i in range(k_thresh)],
                y=[probs[i] for i in range(k_thresh)],
                marker=dict(color="#2997FF", line=dict(color="#101720", width=1.5)),
                name="Main Distribution",
                hovertemplate="Basis State: %{x}<br>Prob: %{y:.4f}<extra></extra>"
            ))
            # Tail Region bars
            fig_overview.add_trace(go.Bar(
                x=[bin_labels[i] for i in range(k_thresh, len(probs))],
                y=[probs[i] for i in range(k_thresh, len(probs))],
                marker=dict(color="#FF4D5A", line=dict(color="#101720", width=1.5)),
                name="Tail Region (L ≥ T)",
                hovertemplate="Tail State: %{x}<br>Prob: %{y:.4f}<extra></extra>"
            ))
            
            # Vertical dashed cutoff marker
            fig_overview.add_vline(
                x=k_thresh - 0.5,
                line_width=1.5,
                line_dash="dash",
                line_color="#FF4D5A"
            )
            # Annotation box for threshold
            fig_overview.add_annotation(
                x=k_thresh - 0.5,
                y=max(probs) * 0.95 if len(probs) > 0 else 0.15,
                text=f"T = {cand_threshold*100:.2f}%",
                showarrow=False,
                bordercolor="#FF4D5A",
                borderwidth=1,
                borderpad=4,
                bgcolor="#1a0f14",
                font=dict(size=11, color="#FF4D5A", family="monospace")
            )
            
            layout_over = get_terminal_layout("", height=265)
            layout_over["barmode"] = "stack"
            layout_over["margin"] = dict(l=40, r=20, t=10, b=35)
            layout_over["xaxis"]["title"] = "Computational Basis State |i⟩"
            layout_over["yaxis"]["title"] = "Amplitude Squared |⟨i|ψ⟩|²"
            layout_over["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=10))
            fig_overview.update_layout(**layout_over)
            st.plotly_chart(fig_overview, width="stretch", config={"displayModeBar": False})
            st.html("</div>")
            
        with tab1_col_metrics:
            # Key Risk Metrics Table Card
            st.html("""
            <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px; margin-bottom:14px;'>
                <div style='display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin-bottom:10px;'>
                    <span style='color:#FF4D5A;'>🎯</span> Key Risk Metrics
                </div>
            """)
            
            var_diff = (q_var_res['quantum_var_pct'] - q_var_res['classical_reference_var_pct']) * 100.0
            cvar_diff = (q_cvar_res['quantum_cvar_pct'] - q_cvar_res['classical_reference_cvar_pct']) * 100.0
            tail_diff = (qae_res['estimated_probability'] - ref_tail_prob) * 100.0
            
            metrics_table_df = pd.DataFrame([
                {
                    "Metric": f"VaR ({int(conf_level*100)}%)",
                    "Quantum": f"{q_var_res['quantum_var_pct']*100:.2f}%",
                    "Classical": f"{q_var_res['classical_reference_var_pct']*100:.2f}%",
                    "Difference": f"{'↑' if var_diff > 0 else '↓'} {abs(var_diff):.2f}%"
                },
                {
                    "Metric": f"CVaR ({int(conf_level*100)}%)",
                    "Quantum": f"{q_cvar_res['quantum_cvar_pct']*100:.2f}%",
                    "Classical": f"{q_cvar_res['classical_reference_cvar_pct']*100:.2f}%",
                    "Difference": f"{'-' if abs(cvar_diff) < 0.001 else ('↑' if cvar_diff > 0 else '↓')} {abs(cvar_diff):.2f}%"
                },
                {
                    "Metric": "Tail Prob. P(L ≥ T)",
                    "Quantum": f"{qae_res['estimated_probability']*100:.2f}%",
                    "Classical": f"{ref_tail_prob*100:.2f}%",
                    "Difference": f"{'↑' if tail_diff > 0 else '↓'} {abs(tail_diff):.2f}%"
                }
            ])
            st.dataframe(metrics_table_df, hide_index=True, width="stretch")
            
            # Quantum vs Classical Comparison Horizontal Bar Chart
            st.html("<div style='font-size:12px; font-weight:600; color:#F1F5F9; margin-top:10px; margin-bottom:4px; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;'>Quantum vs Classical Comparison</div>")
            fig_comp_h = go.Figure()
            fig_comp_h.add_trace(go.Bar(
                y=[f"VaR ({int(conf_level*100)}%)", f"CVaR ({int(conf_level*100)}%)", "Tail Prob."],
                x=[q_var_res['quantum_var_pct']*100, q_cvar_res['quantum_cvar_pct']*100, qae_res['estimated_probability']*100],
                orientation="h",
                name="Quantum",
                marker=dict(color="#2997FF")
            ))
            fig_comp_h.add_trace(go.Bar(
                y=[f"VaR ({int(conf_level*100)}%)", f"CVaR ({int(conf_level*100)}%)", "Tail Prob."],
                x=[q_var_res['classical_reference_var_pct']*100, q_cvar_res['classical_reference_cvar_pct']*100, ref_tail_prob*100],
                orientation="h",
                name="Classical",
                marker=dict(color="#475569")
            ))
            layout_comp_h = get_terminal_layout("", height=125)
            layout_comp_h["barmode"] = "group"
            layout_comp_h["margin"] = dict(l=75, r=15, t=5, b=20)
            layout_comp_h["xaxis"]["ticksuffix"] = "%"
            layout_comp_h["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
            fig_comp_h.update_layout(**layout_comp_h)
            st.plotly_chart(fig_comp_h, width="stretch", config={"displayModeBar": False})
            st.html("</div>")
            
        # Middle Control Row: Loss Threshold Slider + 3 Indicator Cards
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px; margin-bottom:14px;'>
        """)
        
        or_sl_col, or_cd1, or_cd2, or_cd3 = st.columns([5, 2.3, 2.3, 2.4], gap="small")
        
        with or_sl_col:
            st.html(f"""
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                <div style='font-size:12px; font-weight:600; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;'>
                    % Loss Threshold
                </div>
                <div style='font-family:monospace; font-size:12px; color:#FF4D5A; font-weight:700;'>
                    {cand_threshold*100:.2f}
                </div>
            </div>
            <div style='font-size:11px; color:#8B98A8; margin-bottom:4px;'>Candidate Loss Threshold T (as %)</div>
            """)
            new_cand_slider = st.slider(
                "Candidate Loss Threshold T",
                min_value=min_loss_pct,
                max_value=max_loss_pct,
                value=float(cand_threshold * 100.0),
                step=float(discretizer.bin_width * 100.0),
                label_visibility="collapsed",
                key="slider_cand_thresh"
            )
            if abs(new_cand_slider - cand_threshold * 100.0) > 1e-4:
                st.session_state["page4_cand_threshold"] = new_cand_slider
                st.rerun()

        with or_cd1:
            st.html(f"""
            <div style='background:#121A24; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px;'>
                <div style='display:flex; align-items:center; gap:5px; font-size:11px; color:#9B6CFF; font-weight:600;'>
                    <span>🗄</span> Oracle Cutoff Bin
                </div>
                <div style='font-family:monospace; font-size:18px; font-weight:700; color:#F1F5F9; margin-top:4px;'>
                    |{k_thresh}⟩ / |{2**num_qubits-1}⟩
                </div>
                <div style='font-size:10px; color:#8B98A8;'>Marking basis states ≥ |{k_thresh}⟩</div>
            </div>
            """)

        with or_cd2:
            st.html(f"""
            <div style='background:#121A24; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px;'>
                <div style='display:flex; align-items:center; gap:5px; font-size:11px; color:#FF4D5A; font-weight:600;'>
                    <span>🎯</span> Loss Threshold (T)
                </div>
                <div style='font-family:monospace; font-size:18px; font-weight:700; color:#F1F5F9; margin-top:4px;'>
                    {cand_threshold*100:.2f}%
                </div>
                <div style='font-size:10px; color:#8B98A8;'>-${cand_threshold*port_val:,.0f}</div>
            </div>
            """)

        with or_cd3:
            st.html(f"""
            <div style='background:#121A24; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px;'>
                <div style='display:flex; align-items:center; gap:5px; font-size:11px; color:#18C7B5; font-weight:600;'>
                    <span>⚖</span> Classical Ref. P(L ≥ T)
                </div>
                <div style='font-family:monospace; font-size:18px; font-weight:700; color:#F1F5F9; margin-top:4px;'>
                    {ref_tail_prob*100:.2f}%
                </div>
                <div style='display:inline-block; font-size:9px; background:rgba(24,199,181,0.15); color:#18C7B5; padding:1px 4px; border-radius:3px; margin-top:2px;'>
                    GROUND TRUTH
                </div>
            </div>
            """)
            
        st.html("</div>")
        
        # Bottom Flow: Quantum Risk Pipeline (Horizontal Connected Cards)
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;'>
            <div style='display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin-bottom:12px;'>
                <span style='color:#2997FF;'>⚛</span> Quantum Risk Pipeline
            </div>
            <div style='display:flex; align-items:center; justify-content:space-between; gap:6px; overflow-x:auto;'>
                <div style='background:#121A24; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px;'>🗄</div>
                    <div style='font-size:11px; font-weight:700; color:#F1F5F9; margin-top:4px;'>Historical Loss Data</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>Asset returns / portfolio loss distribution</div>
                </div>
                <div style='color:#475569; font-size:16px;'>➔</div>
                <div style='background:#121A24; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px;'>🔢</div>
                    <div style='font-size:11px; font-weight:700; color:#F1F5F9; margin-top:4px;'>Discretization</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>Bin into """ + str(discretizer.num_bins) + f""" loss levels (ΔL = {discretizer.bin_width*100:.3f}%)</div>
                </div>
                <div style='color:#475569; font-size:16px;'>➔</div>
                <div style='background:#121A24; border:1px solid rgba(41,151,255,0.3); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px; color:#2997FF;'>⚡</div>
                    <div style='font-size:11px; font-weight:700; color:#2997FF; margin-top:4px;'>Quantum State Preparation</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>Encode distribution into """ + str(num_qubits) + """ qubits</div>
                </div>
                <div style='color:#475569; font-size:16px;'>➔</div>
                <div style='background:#121A24; border:1px solid rgba(155,108,255,0.3); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px; color:#9B6CFF;'>🎯</div>
                    <div style='font-size:11px; font-weight:700; color:#9B6CFF; margin-top:4px;'>Tail Risk Oracle</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>Mark states with L ≥ T</div>
                </div>
                <div style='color:#475569; font-size:16px;'>➔</div>
                <div style='background:#121A24; border:1px solid rgba(24,199,181,0.3); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px; color:#18C7B5;'>📊</div>
                    <div style='font-size:11px; font-weight:700; color:#18C7B5; margin-top:4px;'>Amplitude Estimation</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>Estimate P(L ≥ T) using QAE (MLQAE)</div>
                </div>
                <div style='color:#475569; font-size:16px;'>➔</div>
                <div style='background:#121A24; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px;'>📈</div>
                    <div style='font-size:11px; font-weight:700; color:#F1F5F9; margin-top:4px;'>Tail Probability</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>P(L ≥ T) = """ + f"{qae_res['estimated_probability']*100:.2f}%" + """</div>
                </div>
                <div style='color:#475569; font-size:16px;'>➔</div>
                <div style='background:#121A24; border:1px solid rgba(46,160,67,0.3); border-radius:6px; padding:10px 12px; flex:1; min-width:120px;'>
                    <div style='font-size:16px; color:#3fb950;'>🛡</div>
                    <div style='font-size:11px; font-weight:700; color:#3fb950; margin-top:4px;'>Quantum VaR / CVaR</div>
                    <div style='font-size:10px; color:#8B98A8; margin-top:2px;'>Compute tail metrics from estimated tail</div>
                </div>
            </div>
        </div>
        """)

    # -------------------------------------------------------------------------
    # TAB 2: QUANTUM STATE (State Preparation & Discretization)
    # -------------------------------------------------------------------------
    with tab2:
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:14px; font-weight:700; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin-bottom:12px;'>
                ⚛ QUANTUM STATE PREPARATION & DISCRETIZATION
            </div>
        """)
        
        # Top Metrics Strip
        t2_m1, t2_m2, t2_m3, t2_m4, t2_m5 = st.columns(5)
        with t2_m1:
            st.metric("State Qubits", f"{num_qubits}", help="Number of register qubits")
        with t2_m2:
            st.metric("Basis States", f"{2**num_qubits}", help="Dimension of computational basis |0⟩ to |2^n-1⟩")
        with t2_m3:
            st.metric("Loss Bins", f"{discretizer.num_bins}", help="Total discretized bins")
        with t2_m4:
            st.metric("Bin Resolution (ΔL)", f"{discretizer.bin_width * 100:.3f}%", help="Interval width per discrete state")
        with t2_m5:
            st.metric("Objective Ancilla", "1 Qubit", help="Flag qubit flipped by tail oracle")
            
        st.html("</div>")
        
        # State Probabilities Distribution Chart
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:13px; font-weight:600; color:#F1F5F9; margin-bottom:8px;'>
                QUANTUM STATE PROBABILITY DISTRIBUTION: |⟨i|ψ⟩|²
            </div>
        """)
        fig_state = go.Figure()
        fig_state.add_trace(go.Bar(
            x=[f"|{i}⟩" for i in range(discretizer.num_bins)],
            y=discretizer.probabilities,
            marker=dict(
                color=["#2997FF" if p < max(discretizer.probabilities)*0.8 else "#18C7B5" for p in discretizer.probabilities],
                line=dict(color="#101720", width=1.5)
            ),
            text=[f"{p*100:.1f}%" for p in discretizer.probabilities],
            textposition="auto",
            hovertemplate="Basis State: %{x}<br>Probability: %{y:.4f}<extra></extra>"
        ))
        layout_state = get_terminal_layout("", height=300)
        layout_state["xaxis"]["title"] = "Computational Basis State |i⟩"
        layout_state["yaxis"]["title"] = "Probability P(|i⟩)"
        layout_state["margin"] = dict(l=40, r=20, t=15, b=35)
        fig_state.update_layout(**layout_state)
        st.plotly_chart(fig_state, width="stretch", config={"displayModeBar": False})
        st.html("</div>")
        
        # Technical State Preparation Panel
        with st.expander("Technical State Preparation Derivation & Circuit Metadata", expanded=True):
            st.latex(r"\mathcal{A} |0\rangle^{\otimes n} |0\rangle = \sum_{i=0}^{2^n-1} \sqrt{p_i} |i\rangle |0\rangle")
            st.markdown(
                f"* **Loss Mapping**: Discrete loss bin center is given by $l_i = l_{{min}} + (i + 0.5) \\Delta L$.\n"
                f"* **Loss Bounds**: [{discretizer.lower_bound*100:.2f}%, {discretizer.upper_bound*100:.2f}%] over {len(losses):,} empirical Monte Carlo samples.\n"
                f"* **Ancilla Flip**: When coupled with oracle $S_\\chi$, basis states satisfying $i \\ge k_{{thresh}}$ flip the $(n+1)$-th objective qubit to $|1\\rangle$."
            )

    # -------------------------------------------------------------------------
    # TAB 3: TAIL ORACLE (Risk Oracle Construction & Cutoff Bin)
    # -------------------------------------------------------------------------
    with tab3:
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:14px; font-weight:700; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin-bottom:12px;'>
                🎯 TAIL RISK ORACLE (BOOLEAN COMPARATOR)
            </div>
        """)
        
        # 3 Indicator Cards
        o_c1, o_c2, o_c3 = st.columns(3)
        with o_c1:
            st.metric("Loss Threshold (T)", f"{cand_threshold*100:.2f}%", f"-${cand_threshold*port_val:,.0f}")
        with o_c2:
            st.metric("Oracle Cutoff Bin", f"|{k_thresh}⟩ / |{2**num_qubits-1}⟩", f"States ≥ |{k_thresh}⟩ marked")
        with o_c3:
            st.metric("Classical Ref. P(L ≥ T)", f"{ref_tail_prob*100:.3f}%", "Ground Truth Empirical Tail")
            
        st.html("</div>")
        
        # Tail Oracle State Partition Visualization
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:13px; font-weight:600; color:#F1F5F9; margin-bottom:8px;'>
                ORACLE STATE PARTITION: SAFE STATES VS MARKED TAIL STATES
            </div>
        """)
        fig_oracle = go.Figure()
        fig_oracle.add_trace(go.Bar(
            x=[f"|{i}⟩" for i in range(k_thresh)],
            y=[discretizer.probabilities[i] for i in range(k_thresh)],
            marker=dict(color="#2997FF"),
            name="Safe Basis States (L < T) |0⟩ ancilla"
        ))
        fig_oracle.add_trace(go.Bar(
            x=[f"|{i}⟩" for i in range(k_thresh, discretizer.num_bins)],
            y=[discretizer.probabilities[i] for i in range(k_thresh, discretizer.num_bins)],
            marker=dict(color="#FF4D5A"),
            name="Marked Tail States (L ≥ T) |1⟩ ancilla"
        ))
        layout_oracle = get_terminal_layout("", height=280)
        layout_oracle["barmode"] = "stack"
        layout_oracle["xaxis"]["title"] = "Computational Basis State |i⟩"
        layout_oracle["yaxis"]["title"] = "Probability P(|i⟩)"
        layout_oracle["margin"] = dict(l=40, r=20, t=10, b=35)
        fig_oracle.update_layout(**layout_oracle)
        st.plotly_chart(fig_oracle, width="stretch", config={"displayModeBar": False})
        st.html("</div>")
        
        with st.expander("Oracle Mathematical Formulation & Grover Reflection", expanded=True):
            st.latex(r"\mathcal{S}_\chi |i\rangle |0\rangle = |i\rangle |f(i)\rangle")
            st.latex(r"S_\chi = I - 2 \sum_{i \ge k_{thresh}} |i\rangle\langle i|")
            st.markdown(
                "This marks all portfolio tail-risk basis states $|i\\rangle$ where the simulated loss exceeds threshold $T$, "
                "rotating the state in the two-dimensional Grover subspace by angle $2\\theta_a$ where $a = \\sin^2(\\theta_a) = P(L \\ge T)$."
            )

    # -------------------------------------------------------------------------
    # TAB 4: QAE RESULTS (Estimation Dashboard & Diagnostics)
    # -------------------------------------------------------------------------
    with tab4:
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:14px; font-weight:700; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin-bottom:12px;'>
                📊 QUANTUM AMPLITUDE ESTIMATION (QAE) RESULTS
            </div>
        """)
        
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        with q_c1:
            st.metric("Estimated P(L ≥ T)", f"{qae_res['estimated_probability']*100:.3f}%", help="Extracted quantum amplitude squared")
        with q_c2:
            st.metric("Absolute Error", f"{qae_res['absolute_error']*100:.4f}%", f"{qae_res['relative_error']*100:.2f}% relative", delta_color="inverse")
        with q_c3:
            st.metric("Quantum Circuit Depth", f"{qae_res.get('circuit_depth', 'N/A')}", f"Total Qubits: {qae_res['num_qubits']}")
        with q_c4:
            st.metric("Quantum Wall Time", f"{qae_res['runtime_seconds']*1000:.1f} ms", "Aer Simulator")
            
        st.html("</div>")
        
        # QAE Diagnostics Table & Comparison
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:13px; font-weight:600; color:#F1F5F9; margin-bottom:10px;'>
                QAE EXECUTION DIAGNOSTICS
            </div>
        """)
        diag_df = pd.DataFrame([
            {"Parameter": "Algorithm", "Value": str(qae_res.get("algorithm", algo_choice))},
            {"Parameter": "Confidence Level (α)", "Value": f"{int(conf_level*100)}%"},
            {"Parameter": "Total Oracle Queries (M)", "Value": f"{qae_res.get('total_oracle_queries', 'N/A')}"},
            {"Parameter": "Total Shot Budget", "Value": f"{qae_res.get('total_shots', shots_count):,}"},
            {"Parameter": "Quantum Circuit Depth", "Value": str(qae_res.get("circuit_depth", "N/A"))},
            {"Parameter": "State Qubits + Ancilla", "Value": f"{num_qubits} + 1 = {num_qubits + 1} qubits"},
            {"Parameter": "Execution Backend", "Value": str(qae_res.get("backend", backend_choice))},
            {"Parameter": "Wall-Clock Runtime", "Value": f"{qae_res['runtime_seconds']*1000:.2f} ms"}
        ])
        st.dataframe(diag_df, hide_index=True, width="stretch")
        st.html("</div>")
        
        # Diagnostic Notice if divergence is noticeable
        if qae_res['absolute_error'] > 0.05:
            st.warning(
                f"⚠ **ESTIMATION DIAGNOSTIC**: The Quantum Amplitude Estimate ({qae_res['estimated_probability']*100:.2f}%) "
                f"differs from the classical empirical ground truth ({ref_tail_prob*100:.2f}%) by {qae_res['absolute_error']*100:.2f}%. "
                f"This reflects discrete bin boundary effects ({discretizer.num_bins} bins) and finite sampling variance in the Aer simulator."
            )

    # -------------------------------------------------------------------------
    # TAB 5: QUANTUM VS CLASSICAL (Head-to-Head Risk Matrix)
    # -------------------------------------------------------------------------
    with tab5:
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:14px; font-weight:700; color:#F1F5F9; font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin-bottom:12px;'>
                ⚖ QUANTUM VS CLASSICAL RISK COMPARISON
            </div>
        """)
        
        head_rows = [
            {
                "Risk Metric": f"VaR ({int(conf_level*100)}%)",
                "Quantum Estimate": f"{q_var_res['quantum_var_pct']*100:.2f}% (-${q_var_res['quantum_var_cash']:,.0f})",
                "Classical Reference": f"{q_var_res['classical_reference_var_pct']*100:.2f}% (-${q_var_res['classical_reference_var_cash']:,.0f})",
                "Absolute Error": f"{q_var_res['absolute_error']*100:.2f}%",
                "Relative Error": f"{q_var_res['relative_error']*100:.2f}%"
            },
            {
                "Risk Metric": f"CVaR ({int(conf_level*100)}%)",
                "Quantum Estimate": f"{q_cvar_res['quantum_cvar_pct']*100:.2f}% (-${q_cvar_res['quantum_cvar_cash']:,.0f})",
                "Classical Reference": f"{q_cvar_res['classical_reference_cvar_pct']*100:.2f}% (-${q_cvar_res['classical_reference_cvar_cash']:,.0f})",
                "Absolute Error": f"{q_cvar_res['absolute_error']*100:.2f}%",
                "Relative Error": f"{q_cvar_res['relative_error']*100:.2f}%"
            },
            {
                "Risk Metric": "Tail Probability P(L ≥ T)",
                "Quantum Estimate": f"{qae_res['estimated_probability']*100:.3f}%",
                "Classical Reference": f"{ref_tail_prob*100:.3f}%",
                "Absolute Error": f"{qae_res['absolute_error']*100:.3f}%",
                "Relative Error": f"{qae_res['relative_error']*100:.2f}%"
            }
        ]
        st.dataframe(pd.DataFrame(head_rows), hide_index=True, width="stretch")
        st.html("</div>")
        
        # Risk Difference Panel
        st.html("""
        <div style='background-color:#101720; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; margin-bottom:16px;'>
            <div style='font-size:13px; font-weight:600; color:#F1F5F9; margin-bottom:8px;'>
                ASYMPTOTIC COMPLEXITY & SCIENTIFIC DISCLOSURE
            </div>
            <div style='font-size:12px; color:#8B98A8; line-height:1.6;'>
                <strong>Quadratic Quantum Speedup:</strong> Classical Monte Carlo achieves convergence error <code>ε ~ O(1/√N)</code>. 
                Quantum Amplitude Estimation extracts the target tail amplitude with query complexity <code>ε ~ O(1/M)</code>, 
                representing a fundamental quadratic speedup in query evaluations.<br><br>
                <strong>Classical Simulation Regime:</strong> On current classical CPU hardware, simulating 2<sup>n</sup> statevectors entails exponential matrix multiplication overhead. The operational speedup manifests when executed natively on fault-tolerant Quantum Processing Units (QPUs).
            </div>
        </div>
        """)
        st.info(f"🔬 **Scientific Disclosure on Quantum CVaR**: {q_cvar_res['scientific_disclosure']}")

    # -------------------------------------------------------------------------
    # TAB 6: HOW IT WORKS (Methodology & Technical Educational Callouts)
    # -------------------------------------------------------------------------
    with tab6:
        explain_amplitude_estimation()

st.html("<br>")
