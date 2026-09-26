"""
Quantum Risk / Event Oracle module.
Marks basis states corresponding to tail losses exceeding a risk threshold.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from qiskit import QuantumCircuit
from src.quantum.state_preparation import LossDistributionDiscretizer, create_state_preparation_circuit


def build_risk_oracle_circuit(
    n_state_qubits: int,
    k_threshold_bin: int,
    name: str = "RiskOracle"
) -> QuantumCircuit:
    """
    Construct a quantum boolean comparator / risk oracle that flips the objective qubit (qubit n)
    if the state register represents an integer i >= k_threshold_bin.
    
    Total qubits: n_state_qubits + 1
    Register [0, ..., n-1]: State register
    Qubit n: Objective / Flag qubit
    """
    total_qubits = n_state_qubits + 1
    qc = QuantumCircuit(total_qubits, name=name)
    state_qubits = list(range(n_state_qubits))
    objective_qubit = n_state_qubits
    
    num_bins = 2 ** n_state_qubits
    k_thresh = int(np.clip(k_threshold_bin, 0, num_bins))
    
    # If k_thresh == 0, all states trigger loss condition (X on objective)
    if k_thresh <= 0:
        qc.x(objective_qubit)
        return qc
        
    # If k_thresh >= num_bins, no states trigger loss condition (identity)
    if k_thresh >= num_bins:
        return qc
        
    # Mark each state i in [k_thresh, num_bins - 1]
    for val in range(k_thresh, num_bins):
        bin_str = format(val, f"0{n_state_qubits}b")
        # Find which qubits have a '0' in the binary representation of val
        # reversed so bin_str[-1] corresponds to state_qubits[0] (little-endian)
        zero_qubits = [state_qubits[idx] for idx, bit in enumerate(reversed(bin_str)) if bit == "0"]
        
        if zero_qubits:
            qc.x(zero_qubits)
        if len(state_qubits) == 1:
            qc.cx(state_qubits[0], objective_qubit)
        elif len(state_qubits) == 2:
            qc.ccx(state_qubits[0], state_qubits[1], objective_qubit)
        else:
            qc.mcx(state_qubits, objective_qubit)
            
        if zero_qubits:
            qc.x(zero_qubits)
            
    return qc


def build_full_risk_operator(
    discretizer: LossDistributionDiscretizer,
    k_threshold_bin: int
) -> QuantumCircuit:
    r"""
    Build the complete state preparation and risk evaluation operator A:
    A |0>^{n+1} = \sum_{i < k} \sqrt{p_i} |i>|0> + \sum_{i >= k} \sqrt{p_i} |i>|1>
    """
    n_qubits = discretizer.num_qubits
    qc = create_state_preparation_circuit(discretizer, objective_qubit=True)
    oracle = build_risk_oracle_circuit(n_qubits, k_threshold_bin)
    qc.compose(oracle, inplace=True)
    return qc


def build_grover_operator(
    full_risk_op: QuantumCircuit,
    n_state_qubits: int
) -> QuantumCircuit:
    """
    Construct the canonical Grover diffusion / reflection operator Q:
    Q = - A S_0 A^dagger S_chi
    where:
      S_chi is phase flip on objective qubit (|1> -> -|1>)
      A^dagger is uncomputation of state preparation and risk oracle
      S_0 is reflection about zero state (2|0><0| - I)
      A is state preparation and risk oracle
    """
    total_qubits = n_state_qubits + 1
    objective_qubit = n_state_qubits
    q_circuit = QuantumCircuit(total_qubits, name="Grover_Q")
    
    # 1. S_chi: Phase flip if objective qubit is 1 (Pauli Z)
    q_circuit.z(objective_qubit)
    
    # 2. A^dagger
    a_inv = full_risk_op.inverse()
    q_circuit.compose(a_inv, inplace=True)
    
    # 3. S_0: Zero reflection 2|0><0| - I
    q_circuit.x(range(total_qubits))
    q_circuit.h(objective_qubit)
    if total_qubits > 2:
        q_circuit.mcx(list(range(total_qubits - 1)), objective_qubit)
    elif total_qubits == 2:
        q_circuit.cx(0, 1)
    else:
        q_circuit.z(0)
    q_circuit.h(objective_qubit)
    q_circuit.x(range(total_qubits))
    
    # 4. A
    q_circuit.compose(full_risk_op, inplace=True)
    return q_circuit
