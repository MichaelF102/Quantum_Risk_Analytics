"""
Mathematical and algorithmic educational callouts for the research platform.
"""
import streamlit as st


def explain_portfolio_return():
    with st.expander("📐 How It Works: Portfolio Return & Mathematical Foundation", expanded=False):
        st.markdown(r"""
        ### Portfolio Return Vectorization
        For a portfolio containing $N$ assets with capital allocations $w = [w_1, w_2, \dots, w_N]^T \in \mathbb{R}^N$
        and asset return vector $R = [R_1, R_2, \dots, R_N]^T \in \mathbb{R}^N$:
        
        $$R_p = w^T R = \sum_{i=1}^N w_i R_i \quad \text{subject to} \quad \sum_{i=1}^N w_i = 1$$
        
        The expected annualized return $\mu_p$ and portfolio variance $\sigma_p^2$ are computed using the covariance matrix $\Sigma$:
        
        $$\mu_p = w^T \mu, \qquad \sigma_p^2 = w^T \Sigma w$$
        
        The Sharpe Ratio measures the excess expected return per unit of total risk:
        
        $$\text{Sharpe Ratio} = \frac{\mu_p - r_f}{\sigma_p}$$
        """)


def explain_var():
    with st.expander("📐 How It Works: Value-at-Risk (VaR)", expanded=False):
        st.markdown(r"""
        ### Value-at-Risk ($\text{VaR}_\alpha$)
        Value-at-Risk quantifies the maximum expected loss over a specific time horizon $h$ at a statistical confidence level $\alpha \in (0, 1)$ (commonly $95\%$ or $99\%$):
        
        $$\text{VaR}_\alpha = - q_{1-\alpha}(R_p) = \inf \{ l \in \mathbb{R} : P(L > l) \le 1 - \alpha \}$$
        
        where $L = -R_p$ is the portfolio loss variable.
        
        * **Parametric Gaussian Model**: Assumes returns are normally distributed:
          $$\text{VaR}_\alpha = - \mu_p h + z_\alpha \sigma_p \sqrt{h}, \quad z_\alpha = \Phi^{-1}(\alpha)$$
        * **Limitation**: Real asset returns exhibit fat tails (leptokurtosis) and volatility clustering, which Gaussian parametric models systematically underestimate.
        """)


def explain_cvar():
    with st.expander("📐 How It Works: Conditional Value-at-Risk (CVaR / Expected Shortfall)", expanded=False):
        st.markdown(r"""
        ### Conditional Value-at-Risk ($\text{CVaR}_\alpha$)
        Conditional Value-at-Risk (also termed Expected Shortfall) is a **coherent risk measure** that calculates the expected loss *given that the loss exceeds the $\text{VaR}_\alpha$ threshold*:
        
        $$\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha] = \frac{1}{1 - \alpha} \int_\alpha^1 \text{VaR}_u \, du$$
        
        * **Coherence**: Unlike VaR, CVaR is sub-additive:
          $$\text{CVaR}_\alpha(X + Y) \le \text{CVaR}_\alpha(X) + \text{CVaR}_\alpha(Y)$$
          meaning portfolio diversification is always mathematically rewarded.
        * **Tail Sensitivity**: CVaR directly captures catastrophic black-swan tail behavior beyond the VaR cutoff.
        """)


def explain_monte_carlo():
    with st.expander("📐 How It Works: Classical Monte Carlo Risk Simulation", expanded=False):
        st.markdown(r"""
        ### Classical Monte Carlo Simulation
        Classical Monte Carlo estimates tail loss probabilities by generating $N$ pseudo-random realizations of correlated asset price paths:
        
        $$S_i(t + \Delta t) = S_i(t) \exp\left( \left(\mu_i - \frac{1}{2}\sigma_i^2\right)\Delta t + \sqrt{\Delta t} (L Z)_i \right)$$
        
        where $\Sigma = L L^T$ is the Cholesky decomposition of the covariance matrix, and $Z \sim \mathcal{N}(0, I_N)$.
        
        * **Central Limit Theorem Rate**: By the Law of Large Numbers and CLT, the standard error of the Monte Carlo estimate scales as:
          $$\epsilon_{MC} \sim \mathcal{O}\left(\frac{1}{\sqrt{N}}\right)$$
        * To gain 1 extra decimal digit of precision ($10\times$ error reduction), classical Monte Carlo requires $100\times$ more sample paths.
        """)


def explain_amplitude_estimation():
    with st.expander("📐 How It Works: Quantum Amplitude Estimation (QAE) & Tail Risk", expanded=False):
        st.markdown(r"""
        ### Quantum Amplitude Estimation (QAE)
        Quantum Amplitude Estimation combines quantum superposition and phase estimation / maximum likelihood to evaluate target probabilities encoded in quantum amplitudes:
        
        1. **State Preparation ($\mathcal{A}$)**: Encodes discretized loss probabilities $p_i$ into $n$ state qubits:
           $$\mathcal{A} |0\rangle^{\otimes n} |0\rangle = \sum_{i=0}^{2^n-1} \sqrt{p_i} |i\rangle |0\rangle$$
        2. **Risk Oracle ($\mathcal{O}$)**: Flips an ancilla flag qubit when loss state exceeds threshold $T$ ($i \ge k_{thresh}$):
           $$\mathcal{A}_{risk} |0\rangle^{\otimes (n+1)} = \sqrt{1 - a} |\psi_{safe}\rangle |0\rangle + \sqrt{a} |\psi_{tail}\rangle |1\rangle$$
           where the target probability is $a = P(L \ge T) = \sin^2(\theta)$.
        3. **Grover Operator ($Q$)**: Iterates the unitary reflection $Q = -\mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$. Each application rotates the state by $2\theta$ in the 2D subspace.
        4. **Quadratic Speedup**: QAE extracts $\theta$ with error:
           $$\epsilon_{QAE} \sim \mathcal{O}\left(\frac{1}{M}\right)$$
           where $M$ is the number of Grover oracle queries. This provides a fundamental quadratic speedup over classical Monte Carlo $\mathcal{O}(1/\sqrt{N})$.
        """)
