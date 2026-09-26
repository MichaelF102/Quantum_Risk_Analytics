"""
Reproducibility utilities: setting deterministic random seeds and serializing experiment results.
"""
import os
import json
import random
import platform
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import qiskit

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python standard library, NumPy, and Aer simulator."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Deterministic seed set to {seed}")


def get_environment_metadata() -> Dict[str, str]:
    """Capture runtime environment versions for experimental reproducibility."""
    import scipy
    import streamlit
    import plotly
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qiskit_version": getattr(qiskit, "__version__", "unknown"),
        "streamlit_version": getattr(streamlit, "__version__", "unknown"),
        "plotly_version": getattr(plotly, "__version__", "unknown")
    }
    
    try:
        import qiskit_aer
        metadata["qiskit_aer_version"] = qiskit_aer.__version__
    except ImportError:
        metadata["qiskit_aer_version"] = "not_installed"
        
    return metadata


def save_experiment_results(
    experiment_name: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    dataframe: Optional[pd.DataFrame] = None,
    output_dir: str = "results"
) -> str:
    """Save an experiment's parameters, results, and environment metadata to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{experiment_name}_{timestamp}"
    
    payload = {
        "experiment_name": experiment_name,
        "parameters": parameters,
        "results": results,
        "environment": get_environment_metadata()
    }
    
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        
    if dataframe is not None:
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        dataframe.to_csv(csv_path, index=False)
        
    logger.info(f"Experiment saved to {json_path}")
    return json_path
