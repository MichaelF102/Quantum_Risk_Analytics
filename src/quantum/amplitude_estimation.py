"""
Quantum Amplitude Estimation (QAE) module for tail probability estimation.
Supports Maximum Likelihood Amplitude Estimation (MLQAE), Canonical QAE (QPE-based),
and Statevector exact simulator with Qiskit Aer.
"""
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.risk_oracle import (
    build_full_risk_operator,
    build_grover_operator,
    build_risk_oracle_circuit
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QuantumAmplitudeEstimator:
    """
    Research-grade Quantum Amplitude Estimation engine.
    Estimates P(Loss >= threshold) by projecting loss onto quantum amplitudes
    and applying quantum amplitude estimation.
    """
    def __init__(
        self,
        discretizer: LossDistributionDiscretizer,
        backend_name: str = "Aer Simulator (Shot-based)",
        seed: int = 42
    ):
        self.discretizer = discretizer
        self.n_state_qubits = discretizer.num_qubits
        self.backend_name = backend_name
        self.seed = seed
        self.aer_sim = AerSimulator(seed_simulator=seed, max_parallel_threads=1, max_parallel_experiments=1)

    def run_mlqae(
        self,
        threshold: float,
        evaluation_schedule: Optional[List[int]] = None,
        shots_per_circuit: int = 1024
    ) -> Dict[str, Any]:
        """
        Run Maximum Likelihood Quantum Amplitude Estimation (MLQAE).
        Reference: Suzuki et al., 'Amplitude Estimation without Phase Estimation' (2020).
        
        Evaluates Grover operator powers Q^m for m in evaluation_schedule.
        Achieves Heisenberg-limited quadratic speedup O(1/M) without ancilla evaluation qubits.
        """
        start_time = time.perf_counter()
        if evaluation_schedule is None:
            # Grover powers: 0, 1, 2, 4
            evaluation_schedule = [0, 1, 2, 4]
            
        k_thresh = self.discretizer.value_to_bin_index(threshold)
        p_ref = self.discretizer.get_classical_tail_probability(threshold)
        
        # Build base operator A
        full_a_op = build_full_risk_operator(self.discretizer, k_thresh)
        grover_q = build_grover_operator(full_a_op, self.n_state_qubits)
        
        total_qubits = self.n_state_qubits + 1
        objective_qubit = self.n_state_qubits
        
        # Collect counts for each power m
        experiment_data = []
        total_oracle_queries = 0
        max_circuit_depth = full_a_op.depth()
        
        for m in evaluation_schedule:
            # Construct circuit: A followed by Q^m
            qc = QuantumCircuit(total_qubits, 1)
            qc.compose(full_a_op, inplace=True)
            
            for _ in range(m):
                qc.compose(grover_q, inplace=True)
                
            qc.measure(objective_qubit, 0)
            max_circuit_depth = max(max_circuit_depth, qc.depth())
            
            # Transpile circuit to native Aer basis gates
            transpiled_qc = transpile(qc, self.aer_sim, optimization_level=1)
            # Execute on Aer
            result = self.aer_sim.run(transpiled_qc, shots=shots_per_circuit, seed_simulator=self.seed).result()
            counts = result.get_counts()
            
            # '1' corresponds to objective qubit = 1
            h_m = counts.get("1", 0)
            experiment_data.append((m, h_m, shots_per_circuit))
            
            # Each Grover iteration Q contains 2 oracle queries (S_chi and S_0)
            total_oracle_queries += (2 * m + 1) * shots_per_circuit
            
        # Maximum Likelihood Optimization
        # P(1 | theta) = sin^2((2m + 1) * theta)
        def neg_log_likelihood(theta: float) -> float:
            nll = 0.0
            for m_val, h_val, n_shots in experiment_data:
                sin_sq = np.sin((2 * m_val + 1) * theta) ** 2
                p_m = np.clip(sin_sq, 1e-12, 1.0 - 1e-12)
                nll -= h_val * np.log(p_m) + (n_shots - h_val) * np.log(1.0 - p_m)
            return nll

        # Angle theta in [0, pi/2]
        opt_res = minimize_scalar(
            neg_log_likelihood,
            bounds=(1e-5, np.pi / 2.0 - 1e-5),
            method="bounded"
        )
        
        theta_est = float(opt_res.x)
        p_est = float(np.sin(theta_est) ** 2)
        abs_err = float(abs(p_est - p_ref))
        rel_err = float(abs_err / max(p_ref, 1e-6))
        
        runtime = time.perf_counter() - start_time
        
        return {
            "algorithm": "Maximum Likelihood QAE (MLQAE)",
            "backend": self.backend_name,
            "threshold": threshold,
            "k_threshold_bin": k_thresh,
            "num_qubits": total_qubits,
            "state_qubits": self.n_state_qubits,
            "ancilla_qubits": 1,
            "circuit_depth": max_circuit_depth,
            "evaluation_schedule": evaluation_schedule,
            "shots_per_evaluation": shots_per_circuit,
            "total_shots": shots_per_circuit * len(evaluation_schedule),
            "total_oracle_queries": total_oracle_queries,
            "estimated_amplitude": float(np.sin(theta_est)),
            "estimated_probability": p_est,
            "classical_reference_probability": p_ref,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "runtime_seconds": float(runtime),
            "theoretical_speedup": "Quadratic: O(1/M) vs Classical Monte Carlo O(1/sqrt(N))"
        }

    def run_canonical_qae(
        self,
        threshold: float,
        num_eval_qubits: int = 4,
        shots: int = 2048
    ) -> Dict[str, Any]:
        """
        Run Canonical Quantum Amplitude Estimation (Brassard et al., 2002).
        Uses Quantum Phase Estimation with num_eval_qubits and inverse QFT.
        """
        start_time = time.perf_counter()
        k_thresh = self.discretizer.value_to_bin_index(threshold)
        p_ref = self.discretizer.get_classical_tail_probability(threshold)
        
        m = num_eval_qubits
        n = self.n_state_qubits
        total_qubits = m + n + 1
        obj_qubit_in_sys = n
        
        # Base operator A and Grover operator Q
        full_a_op = build_full_risk_operator(self.discretizer, k_thresh)
        grover_q = build_grover_operator(full_a_op, n)
        grover_gate = grover_q.to_gate(label="Q")
        
        qc = QuantumCircuit(total_qubits, m)
        # 1. State preparation on system register [m, ..., total_qubits-1]
        system_qubits = list(range(m, total_qubits))
        qc.compose(full_a_op, qubits=system_qubits, inplace=True)
        
        # 2. Superposition on evaluation register [0, ..., m-1]
        eval_qubits = list(range(m))
        qc.h(eval_qubits)
        
        # 3. Controlled-Q^(2^j) powers
        for j in range(m):
            power = 2 ** j
            # Controlled grover gate
            ctrl_q = grover_gate.power(power).control(1)
            qc.append(ctrl_q, [eval_qubits[j]] + system_qubits)
            
        # 4. Inverse QFT on evaluation register
        qft_inv = QFTGate(m).inverse()
        qc.append(qft_inv, eval_qubits)
        
        # 5. Measure evaluation register
        qc.measure(eval_qubits, list(range(m)))
        
        circuit_depth = qc.depth()
        transpiled_qc = transpile(qc, self.aer_sim, optimization_level=1)
        result = self.aer_sim.run(transpiled_qc, shots=shots, seed_simulator=self.seed).result()
        counts = result.get_counts()
        
        # Most frequent measured bitstring gives the phase estimate
        top_bitstring = max(counts, key=counts.get)
        y_int = int(top_bitstring, 2)
        theta_est = (np.pi * y_int) / (2 ** m)
        p_est = float(np.sin(theta_est) ** 2)
        
        abs_err = float(abs(p_est - p_ref))
        rel_err = float(abs_err / max(p_ref, 1e-6))
        runtime = time.perf_counter() - start_time
        
        return {
            "algorithm": "Canonical QAE (Phase Estimation)",
            "backend": self.backend_name,
            "threshold": threshold,
            "k_threshold_bin": k_thresh,
            "num_qubits": total_qubits,
            "evaluation_qubits": m,
            "state_qubits": n,
            "ancilla_qubits": 1,
            "circuit_depth": circuit_depth,
            "shots": shots,
            "most_frequent_measurement": top_bitstring,
            "estimated_amplitude": float(np.sin(theta_est)),
            "estimated_probability": p_est,
            "classical_reference_probability": p_ref,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "runtime_seconds": float(runtime),
            "theoretical_speedup": "O(1/2^m) precision with 2^m Grover applications"
        }

    def run_statevector_exact(self, threshold: float) -> Dict[str, Any]:
        """
        Exact quantum statevector evaluation (zero sampling noise).
        Mathematically verifies the quantum amplitude projection.
        """
        start_time = time.perf_counter()
        k_thresh = self.discretizer.value_to_bin_index(threshold)
        p_ref = self.discretizer.get_classical_tail_probability(threshold)
        
        full_a_op = build_full_risk_operator(self.discretizer, k_thresh)
        sv = Statevector.from_instruction(full_a_op)
        probs = sv.probabilities_dict()
        
        # Target states where the objective qubit (highest bit in Qiskit bitstring) is '1'
        tail_prob = sum(v for k, v in probs.items() if k[0] == "1")
        
        runtime = time.perf_counter() - start_time
        return {
            "algorithm": "Exact Statevector Simulation",
            "backend": "Aer Statevector",
            "threshold": threshold,
            "k_threshold_bin": k_thresh,
            "num_qubits": self.n_state_qubits + 1,
            "state_qubits": self.n_state_qubits,
            "circuit_depth": full_a_op.depth(),
            "estimated_probability": float(tail_prob),
            "classical_reference_probability": p_ref,
            "absolute_error": float(abs(tail_prob - p_ref)),
            "relative_error": float(abs(tail_prob - p_ref) / max(p_ref, 1e-6)),
            "runtime_seconds": float(runtime)
        }
