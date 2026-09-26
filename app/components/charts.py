"""
Interactive Plotly charts designed for the quantitative research terminal theme.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from src.utils.config import CONFIG

# Palette
BG_COLOR = "#0e1117"
CARD_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_AMBER = "#d29922"
ACCENT_PURPLE = "#bc8cff"
ACCENT_CYAN = "#39c5cf"


def get_terminal_layout(title: str = "", height: int = 420) -> dict:
    """Standard layout for dark terminal styling."""
    return dict(
        title=dict(
            text=f"<span style='color:{ACCENT_BLUE}; font-family:monospace; font-size:14px;'>// {title.upper()}</span>",
            x=0.01,
            y=0.96
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#11151c",
        height=height,
        margin=dict(l=45, r=25, t=45, b=40),
        font=dict(family="JetBrains Mono, SF Mono, Consolas, monospace", color=TEXT_COLOR, size=11),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor="#30363d",
            showline=True,
            linecolor="#30363d",
            tickfont=dict(color="#8b949e")
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor="#30363d",
            showline=True,
            linecolor="#30363d",
            tickfont=dict(color="#8b949e")
        ),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=10)
        )
    )


def plot_allocation_donut(asset_weights: Dict[str, float]) -> go.Figure:
    """Plot asset allocation donut chart."""
    labels = list(asset_weights.keys())
    values = list(asset_weights.values())
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            textinfo="label+percent",
            marker=dict(
                colors=[ACCENT_BLUE, ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_RED, "#e36209", "#8957e5"],
                line=dict(color="#0e1117", width=2)
            ),
            hoverinfo="label+percent+value"
        )
    ])
    layout = get_terminal_layout("Portfolio Asset Allocation", height=350)
    fig.update_layout(**layout)
    return fig


def plot_cumulative_returns(cum_returns: pd.Series) -> go.Figure:
    """Plot historical cumulative return series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values * 100.0,
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=2),
        name="Cumulative Return (%)",
        fill="tozeroy",
        fillcolor="rgba(57, 197, 207, 0.08)"
    ))
    layout = get_terminal_layout("Historical Cumulative Performance (%)", height=380)
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    return fig


def plot_rolling_volatility(rolling_vol: pd.DataFrame) -> go.Figure:
    """Plot rolling volatility series."""
    fig = go.Figure()
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_RED, ACCENT_CYAN, "#ff7b72", "#79c0ff"]
    for idx, col in enumerate(rolling_vol.columns):
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[col].values * 100.0,
            mode="lines",
            line=dict(color=colors[idx % len(colors)], width=1.5),
            name=str(col)
        ))
    layout = get_terminal_layout("21-Day Rolling Annualized Volatility (%)", height=380)
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Plot correlation matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="Viridis",
        zmin=-1.0,
        zmax=1.0,
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    layout = get_terminal_layout("Asset Correlation Matrix (Pearson)", height=400)
    fig.update_layout(**layout)
    return fig


def plot_loss_distribution_with_tail(
    losses: np.ndarray,
    var_threshold: float,
    cvar_threshold: float,
    confidence_level: float = 0.95,
    title: str = "Portfolio Loss Distribution & Tail Risk",
    is_percentage: bool = True
) -> go.Figure:
    """
    Plot loss distribution with VaR vertical line and CVaR tail highlighted.
    """
    scale = 100.0 if is_percentage else 1.0
    scaled_losses = losses * scale
    scaled_var = var_threshold * scale
    scaled_cvar = cvar_threshold * scale
    
    fig = go.Figure()
    
    # Histogram
    counts, bin_edges = np.histogram(scaled_losses, bins=60, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Body distribution
    body_mask = bin_centers < scaled_var
    tail_mask = bin_centers >= scaled_var
    
    fig.add_trace(go.Bar(
        x=bin_centers[body_mask],
        y=counts[body_mask],
        width=bin_edges[1] - bin_edges[0],
        marker=dict(color="#1f6feb", opacity=0.7),
        name="Loss Distribution"
    ))
    
    # Tail distribution (shaded red)
    if np.any(tail_mask):
        fig.add_trace(go.Bar(
            x=bin_centers[tail_mask],
            y=counts[tail_mask],
            width=bin_edges[1] - bin_edges[0],
            marker=dict(color=ACCENT_RED, opacity=0.85),
            name=f"Tail Loss (Top {int((1-confidence_level)*100)}%)"
        ))
        
    # Vertical line for VaR
    fig.add_vline(
        x=scaled_var,
        line_width=2.5,
        line_dash="dash",
        line_color=ACCENT_AMBER,
        annotation_text=f"VaR {int(confidence_level*100)}%: {scaled_var:.2f}{'%' if is_percentage else ''}",
        annotation_position="top left",
        annotation_font=dict(color=ACCENT_AMBER, size=11)
    )
    
    # Vertical line for CVaR
    fig.add_vline(
        x=scaled_cvar,
        line_width=2.5,
        line_dash="dot",
        line_color="#ff7b72",
        annotation_text=f"CVaR {int(confidence_level*100)}%: {scaled_cvar:.2f}{'%' if is_percentage else ''}",
        annotation_position="top right",
        annotation_font=dict(color="#ff7b72", size=11)
    )
    
    layout = get_terminal_layout(title, height=420)
    layout["xaxis"]["title"] = "Loss (%)" if is_percentage else "Loss ($)"
    layout["yaxis"]["title"] = "Probability Density"
    fig.update_layout(**layout)
    return fig


def plot_convergence_curves(df_conv: pd.DataFrame) -> go.Figure:
    """Plot Error vs Sample Count for Classical MC and Quantum QAE."""
    fig = go.Figure()
    
    # MC MAE
    fig.add_trace(go.Scatter(
        x=df_conv["Samples"],
        y=df_conv["MC_MAE"],
        mode="lines+markers",
        name="Classical MC MAE (O(1/√N))",
        line=dict(color=ACCENT_BLUE, width=2.5),
        marker=dict(size=7)
    ))
    
    # QAE MAE
    fig.add_trace(go.Scatter(
        x=df_conv["Samples"],
        y=df_conv["QAE_MAE"],
        mode="lines+markers",
        name="Quantum QAE MAE (O(1/M))",
        line=dict(color=ACCENT_GREEN, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond")
    ))
    
    layout = get_terminal_layout("Convergence Scaling: Absolute Error vs Evaluations", height=420)
    layout["xaxis"]["type"] = "log"
    layout["yaxis"]["type"] = "log"
    layout["xaxis"]["title"] = "Sample / Evaluation Count (Log Scale)"
    layout["yaxis"]["title"] = "Mean Absolute Error (Log Scale)"
    fig.update_layout(**layout)
    return fig


def plot_scalability_resources(df_scale: pd.DataFrame) -> go.Figure:
    """Plot scalability across asset count: Classical MC Runtime vs Quantum Circuit Depth."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_scale["Assets"],
        y=df_scale["Classical_MC_Time_Sec"] * 1000.0,
        name="Classical MC Time (ms)",
        marker=dict(color=ACCENT_BLUE, opacity=0.8),
        yaxis="y1"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_scale["Assets"],
        y=df_scale["Circuit_Depth_Grover"],
        mode="lines+markers",
        name="Grover Circuit Depth",
        line=dict(color=ACCENT_PURPLE, width=2.5),
        marker=dict(size=8),
        yaxis="y2"
    ))
    
    layout = get_terminal_layout("Scalability Benchmark: Portfolio Dimensions vs Computational Cost", height=420)
    layout["xaxis"]["title"] = "Number of Portfolio Assets"
    layout["yaxis"] = dict(title="MC Runtime (ms)", titlefont=dict(color=ACCENT_BLUE), tickfont=dict(color=ACCENT_BLUE), gridcolor=GRID_COLOR)
    layout["yaxis2"] = dict(title="Quantum Circuit Depth", titlefont=dict(color=ACCENT_PURPLE), tickfont=dict(color=ACCENT_PURPLE), overlaying="y", side="right", showgrid=False)
    fig.update_layout(**layout)
    return fig


def plot_stress_test_comparison(
    baseline_losses: np.ndarray,
    stressed_losses: np.ndarray,
    scenario_name: str,
    base_var: float,
    stressed_var: float
) -> go.Figure:
    """Plot before vs after stress scenario loss distributions."""
    fig = go.Figure()
    
    # Baseline
    b_counts, b_edges = np.histogram(baseline_losses * 100.0, bins=50, density=True)
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    fig.add_trace(go.Scatter(
        x=b_centers,
        y=b_counts,
        mode="lines",
        line=dict(color=ACCENT_BLUE, width=2),
        name="Baseline Distribution",
        fill="tozeroy",
        fillcolor="rgba(88, 166, 255, 0.15)"
    ))
    
    # Stressed
    s_counts, s_edges = np.histogram(stressed_losses * 100.0, bins=50, density=True)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    fig.add_trace(go.Scatter(
        x=s_centers,
        y=s_counts,
        mode="lines",
        line=dict(color=ACCENT_RED, width=2),
        name=f"Stressed ({scenario_name})",
        fill="tozeroy",
        fillcolor="rgba(248, 81, 73, 0.15)"
    ))
    
    # VaR lines
    fig.add_vline(x=base_var * 100.0, line_dash="dash", line_color=ACCENT_BLUE, annotation_text=f"Base VaR: {base_var*100:.2f}%")
    fig.add_vline(x=stressed_var * 100.0, line_dash="dash", line_color=ACCENT_RED, annotation_text=f"Stressed VaR: {stressed_var*100:.2f}%")
    
    layout = get_terminal_layout(f"Stress Impact: Baseline vs {scenario_name}", height=420)
    layout["xaxis"]["title"] = "Loss (%)"
    layout["yaxis"]["title"] = "Density"
    fig.update_layout(**layout)
    return fig
