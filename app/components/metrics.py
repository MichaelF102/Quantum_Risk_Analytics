"""
Terminal-style metric cards and KPI displays.
"""
from typing import Optional
import streamlit as st


def render_terminal_kpi(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    sublabel: Optional[str] = None,
    badge: Optional[str] = None
) -> None:
    """
    Render a sleek, dark-terminal styled KPI card.
    """
    badge_html = ""
    if badge:
        badge_html = f"<span style='background-color:#21262d; color:#58a6ff; font-family:monospace; font-size:10px; padding:2px 6px; border-radius:3px; border:1px solid #30363d; margin-left:6px;'>{badge}</span>"
        
    delta_html = ""
    if delta:
        color = "#3fb950" if delta_color == "green" else ("#f85149" if delta_color == "red" else "#8b949e")
        delta_html = f"<div style='font-family:monospace; font-size:12px; color:{color}; margin-top:4px;'>{delta}</div>"
        
    sublabel_html = ""
    if sublabel:
        sublabel_html = f"<div style='font-family:monospace; font-size:10px; color:#8b949e; margin-top:3px;'>{sublabel}</div>"

    card_html = f"""
    <div style='background-color:#161b22; border:1px solid #30363d; border-radius:6px; padding:12px 16px; margin-bottom:10px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div style='font-family:monospace; font-size:11px; text-transform:uppercase; color:#8b949e; letter-spacing:0.5px;'>
                {label}
            </div>
            {badge_html}
        </div>
        <div style='font-family:monospace; font-size:22px; font-weight:700; color:#f0f6fc; margin-top:6px; letter-spacing:-0.5px;'>
            {value}
        </div>
        {delta_html}
        {sublabel_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_status_pill(status: str, mode: str = "SIMULATED") -> None:
    """Render scientific status tag distinguishing Measured vs Simulated vs Theoretical."""
    color_map = {
        "MEASURED": ("#238636", "#3fb950"),
        "SIMULATED": ("#1f6feb", "#58a6ff"),
        "THEORETICAL": ("#8957e5", "#bc8cff"),
        "APPROXIMATE": ("#9e6a03", "#d29922")
    }
    bg, fg = color_map.get(mode.upper(), ("#30363d", "#c9d1d9"))
    
    html = f"""
    <div style='display:inline-block; font-family:monospace; font-size:11px; padding:3px 8px; border-radius:4px; background-color:{bg}33; border:1px solid {bg}; color:{fg}; margin-bottom:8px;'>
        STATUS: <strong>{mode.upper()}</strong> | {status}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
