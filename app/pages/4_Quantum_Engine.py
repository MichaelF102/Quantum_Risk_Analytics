"""
Page 4: Quantum Risk Engine - State Preparation, Risk Oracle, and Amplitude Estimation
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
import plotly.graph_objects as go

try:
    from app.state import initialize_app_theme, get_shared_sidebar_inputs
    from app.components.charts import get_terminal_layout
    from app.components.metrics import render_terminal_kpi, render_status_pill
    from app.components.explanations import explain_amplitude_estimation
except (ModuleNotFoundError, ImportError):
    from state import initialize_app_theme, get_shared_sidebar_inputs
    from components.charts import get_terminal_layout
    from components.metrics import render_terminal_kpi, render_status_pill
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
conf_level = inputs["conf_level"]
time_horizon = inputs["time_horizon"]
port_val = inputs["portfolio_val"]
num_qubits = inputs["num_qubits"]
backend_choice = inputs["backend_choice"]
seed = inputs["seed"]

st.markdown("""
<div style='border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;'>
    <h2 style='margin:0; font-family:monospace; color:#f0f6fc;'>// 04_QUANTUM_RISK_ENGINE</h2>
    <div style='font-family:monospace; font-size:12px; color:#58a6ff;'>State Preparation, Quantum Boolean Risk Oracle, and Amplitude Estimation (QAE)</div>
</div>
""", unsafe_allow_html=True)

render_status_pill("Simulating quantum circuits using Qiskit Aer backend with full statevector & shot sampling.", mode="SIMULATED")

# 1. Obtain loss distribution from classical simulation
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

# 2. Build Discretizer
discretizer = LossDistributionDiscretizer(losses, num_qubits=num_qubits)

# Quantum Engine Controls
st.markdown("### // 1. STATE PREPARATION & DISCRETIZATION")
c_p1, c_p2 = st.columns([6, 6])

with c_p1:
    st.markdown(rf"""
    * **State Register Qubits**: `{num_qubits}` qubits $\to$ `{2**num_qubits}` discrete loss bins.
    * **Objective Ancilla Qubit**: `1` qubit (marks $L \ge T$).
    * **Total System Qubits**: `{num_qubits + 1}` qubits.
    * **Bin Resolution ($\Delta L$)**: `{discretizer.bin_width * 100:.3f}%` loss per bin.
    * **Loss Bounds**: `[{discretizer.lower_bound*100:.2f}%, {discretizer.upper_bound*100:.2f}%]`.
    """)

with c_p2:
    # Plot discrete quantum state probabilities
    fig_bins = go.Figure()
    fig_bins.add_trace(go.Bar(
        x=[f"|{i}⟩" for i in range(discretizer.num_bins)],
        y=discretizer.probabilities,
        marker=dict(color="#58a6ff"),
        name="State Probabilities P(|i⟩)"
    ))
    layout_bins = get_terminal_layout(f"Quantum State Preparation: {2**num_qubits} Basis Amplitudes", height=280)
    layout_bins["xaxis"]["title"] = "Computational Basis State |i⟩"
    layout_bins["yaxis"]["title"] = "Amplitude Squared |⟨i|ψ⟩|²"
    fig_bins.update_layout(**layout_bins)
    st.plotly_chart(fig_bins, use_container_width=True)

st.markdown("---")
st.markdown("### // 2. INTERACTIVE RISK THRESHOLD & TAIL ORACLE")

# Threshold Slider
default_thresh = float(np.percentile(losses, conf_level * 100.0))
cand_threshold = st.slider(
    "Candidate Loss Threshold T (as %)",
    min_value=float(discretizer.bin_edges[0] * 100.0),
    max_value=float(discretizer.bin_edges[-1] * 100.0),
    value=float(default_thresh * 100.0),
    step=float(discretizer.bin_width * 100.0)
) / 100.0

k_thresh = discretizer.value_to_bin_index(cand_threshold)
ref_tail_prob = discretizer.get_classical_tail_probability(cand_threshold)

col_or1, col_or2, col_or3 = st.columns(3)
with col_or1:
    render_terminal_kpi("Loss Threshold (T)", f"{cand_threshold*100:.2f}%", sublabel=f"-${cand_threshold*port_val:,.0f}")
with col_or2:
    render_terminal_kpi("Oracle Cutoff Bin", f"|{k_thresh}⟩ / |{2**num_qubits-1}⟩", sublabel=f"Marking basis states ≥ |{k_thresh}⟩")
with col_or3:
    render_terminal_kpi("Classical Ref. P(L ≥ T)", f"{ref_tail_prob * 100:.3f}%", badge="GROUND TRUTH")

# Run QAE Engine
qae = QuantumAmplitudeEstimator(discretizer, backend_name=backend_choice, seed=seed)

qae_algorithm = st.selectbox(
    "Amplitude Estimation Algorithm",
    ["Maximum Likelihood QAE (MLQAE)", "Exact Statevector Simulation", "Canonical QAE (Phase Estimation)"]
)

if "MLQAE" in qae_algorithm:
    qae_res = qae.run_mlqae(cand_threshold, shots_per_circuit=1024)
elif "Exact" in qae_algorithm:
    qae_res = qae.run_statevector_exact(cand_threshold)
else:
    qae_res = qae.run_canonical_qae(cand_threshold, num_eval_qubits=4, shots=1024)

# Display QAE Results
st.markdown("### // 3. QUANTUM ESTIMATION RESULTS")
q_c1, q_c2, q_c3, q_c4 = st.columns(4)
with q_c1:
    render_terminal_kpi("Estimated P(L ≥ T)", f"{qae_res['estimated_probability']*100:.3f}%", badge="QUANTUM")
with q_c2:
    render_terminal_kpi("Absolute Error (|p̂ - p|)", f"{qae_res['absolute_error']*100:.4f}%", delta=f"{qae_res['relative_error']*100:.2f}% relative", delta_color="normal")
with q_c3:
    render_terminal_kpi("Quantum Circuit Depth", f"{qae_res.get('circuit_depth', 'N/A')}", sublabel=f"Total Qubits: {qae_res['num_qubits']}")
with q_c4:
    render_terminal_kpi("Quantum Wall Time", f"{qae_res['runtime_seconds']*1000:.1f} ms", delta="Aer Simulator")

st.markdown("---")
st.markdown("### // 4. QUANTUM VaR & QUANTUM CVaR SUMMARY")

# Run Quantum VaR and CVaR estimators
q_var_estimator = QuantumVaREstimator(discretizer, seed=seed)
q_cvar_estimator = QuantumCVaREstimator(discretizer, seed=seed)

q_var_res = q_var_estimator.estimate_var(confidence_level=conf_level, method="mlqae", portfolio_value=port_val)
q_cvar_res = q_cvar_estimator.estimate_cvar(confidence_level=conf_level, quantum_var_pct=q_var_res["quantum_var_pct"], method="mlqae", portfolio_value=port_val)

var_c1, var_c2, var_c3, var_c4 = st.columns(4)
with var_c1:
    render_terminal_kpi(f"Quantum VaR ({int(conf_level*100)}%)", f"{q_var_res['quantum_var_pct']*100:.2f}%", sublabel=f"-${q_var_res['quantum_var_cash']:,.0f}", delta_color="red")
with var_c2:
    render_terminal_kpi(f"Classical VaR ({int(conf_level*100)}%)", f"{q_var_res['classical_reference_var_pct']*100:.2f}%", sublabel=f"-${q_var_res['classical_reference_var_cash']:,.0f}", delta_color="red")
with var_c3:
    render_terminal_kpi(f"Quantum CVaR ({int(conf_level*100)}%)", f"{q_cvar_res['quantum_cvar_pct']*100:.2f}%", sublabel=f"-${q_cvar_res['quantum_cvar_cash']:,.0f}", delta_color="red")
with var_c4:
    render_terminal_kpi(f"Classical CVaR ({int(conf_level*100)}%)", f"{q_cvar_res['classical_reference_cvar_pct']*100:.2f}%", sublabel=f"-${q_cvar_res['classical_reference_cvar_cash']:,.0f}", delta_color="red")

st.markdown("<br>", unsafe_allow_html=True)
st.info(f"🔬 **Scientific Disclosure on Quantum CVaR**: {q_cvar_res['scientific_disclosure']}")

explain_amplitude_estimation()
