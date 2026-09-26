"""
Table formatting and presentation components for benchmarking.
"""
from typing import Dict, Any, List
import pandas as pd
import streamlit as st


def render_benchmark_comparison_table(
    classical_data: Dict[str, Any],
    quantum_data: Dict[str, Any],
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Format and render the head-to-head Classical Monte Carlo vs Quantum Amplitude Estimation table.
    """
    alpha_key = f"{int(confidence_level * 100)}%"
    
    cl_var = classical_data.get("risk_metrics", {}).get(alpha_key, {}).get("var_pct", 0.0)
    cl_cvar = classical_data.get("risk_metrics", {}).get(alpha_key, {}).get("cvar_pct", 0.0)
    cl_time = classical_data.get("runtime_seconds", 0.0)
    cl_samples = classical_data.get("n_simulations", 0)
    
    q_var = quantum_data.get("quantum_var_pct", 0.0)
    q_cvar = quantum_data.get("quantum_cvar_pct", 0.0)
    q_time = quantum_data.get("runtime_seconds", 0.0)
    q_samples = quantum_data.get("total_shots", quantum_data.get("shots", "N/A"))
    q_qubits = quantum_data.get("num_qubits", "N/A")
    q_depth = quantum_data.get("circuit_depth", "N/A")
    
    abs_var_err = abs(q_var - cl_var)
    rel_var_err = abs_var_err / max(abs(cl_var), 1e-6)
    
    rows = [
        {"Metric": "Confidence Level (α)", "Classical MC": f"{int(confidence_level*100)}%", "Quantum QAE": f"{int(confidence_level*100)}%", "Difference / Error": "-"},
        {"Metric": "VaR (Loss %)", "Classical MC": f"{cl_var * 100:.3f}%", "Quantum QAE": f"{q_var * 100:.3f}%", "Difference / Error": f"{abs_var_err * 100:.3f}% (Abs)"},
        {"Metric": "CVaR (Loss %)", "Classical MC": f"{cl_cvar * 100:.3f}%", "Quantum QAE": f"{q_cvar * 100:.3f}%", "Difference / Error": f"{abs(q_cvar - cl_cvar) * 100:.3f}% (Abs)"},
        {"Metric": "Relative Error (VaR)", "Classical MC": "Baseline (0.0%)", "Quantum QAE": f"{rel_var_err * 100:.2f}%", "Difference / Error": f"{rel_var_err:.4f}"},
        {"Metric": "Runtime (Seconds)", "Classical MC": f"{cl_time:.4f}s", "Quantum QAE": f"{q_time:.4f}s", "Difference / Error": f"{q_time / max(cl_time, 1e-6):.1f}x (Simulated overhead)"},
        {"Metric": "Samples / Shots", "Classical MC": f"{cl_samples:,}", "Quantum QAE": f"{q_samples}", "Difference / Error": "Quantum uses structured Grover queries"},
        {"Metric": "Qubits Required", "Classical MC": "N/A", "Quantum QAE": f"{q_qubits}", "Difference / Error": "Logarithmic state space: 2^n bins"},
        {"Metric": "Circuit Depth", "Classical MC": "N/A", "Quantum QAE": f"{q_depth}", "Difference / Error": "Bounded depth on Aer simulator"},
        {"Metric": "Execution Regime", "Classical MC": "Measured (CPU)", "Quantum QAE": "Simulated (Aer)", "Difference / Error": "No hardware noise"}
    ]
    
    df_table = pd.DataFrame(rows)
    return df_table
