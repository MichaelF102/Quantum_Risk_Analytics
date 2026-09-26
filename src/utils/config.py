"""
Configuration constants and settings for Quantum Risk Analytics Engine.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class EngineConfig:
    """Core configuration for classical and quantum risk analytics."""
    default_seed: int = 42
    default_portfolio_value: float = 1_000_000.0  # $1,000,000 nominal
    trading_days_per_year: int = 252
    
    # Risk parameters
    supported_confidence_levels: List[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.995]
    )
    default_confidence_level: float = 0.95
    default_time_horizon_days: int = 10  # 10-day Basel regulatory horizon
    
    # Classical Monte Carlo defaults
    default_mc_simulations: int = 20_000
    max_mc_simulations: int = 200_000
    
    # Quantum defaults
    default_quantum_qubits: int = 4  # 2^4 = 16 discrete loss bins
    max_quantum_qubits: int = 7       # 2^7 = 128 discrete loss bins (simulation cap)
    default_qae_shots: int = 2048
    supported_backends: List[str] = field(
        default_factory=lambda: ["Aer Simulator (Statevector)", "Aer Simulator (Shot-based)", "Qiskit Basic Simulator"]
    )
    
    # UI styling constants
    theme_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "background": "#0d1117",
            "surface": "#161b22",
            "border": "#30363d",
            "text": "#c9d1d9",
            "accent_blue": "#58a6ff",
            "accent_green": "#3fb950",
            "accent_red": "#f85149",
            "accent_amber": "#d29922",
            "accent_purple": "#bc8cff",
            "accent_cyan": "#39c5cf"
        }
    )


CONFIG = EngineConfig()
