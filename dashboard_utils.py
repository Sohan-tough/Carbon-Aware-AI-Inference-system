"""
dashboard_utils.py — Reusable Plotly chart builders and UI helpers.

All chart functions return ``go.Figure`` objects so they are trivially
embeddable in Streamlit with ``st.plotly_chart(fig, use_container_width=True)``.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import ENERGY_RULE_ENGINE, ENERGY_SMALL_MODEL, ENERGY_LARGE_MODEL

# ── Colour palette ─────────────────────────────────────────────────────────────
STAGE_COLOURS = {
    "Rule Engine": "#22c55e",   # green
    "RoBERTa":     "#3b82f6",   # blue
    "BERT":        "#f97316",   # orange
}

BG_COLOUR     = "#0f172a"   # slate-900
CARD_COLOUR   = "#1e293b"   # slate-800
TEXT_COLOUR   = "#f1f5f9"   # slate-100
GRID_COLOUR   = "#334155"   # slate-700


def _base_layout(title: str) -> dict:
    """Shared dark-theme layout for all charts."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOUR, size=15)),
        paper_bgcolor=CARD_COLOUR,
        plot_bgcolor=CARD_COLOUR,
        font=dict(color=TEXT_COLOUR, family="monospace"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


# ── Chart 1: Energy Comparison Bar Chart ──────────────────────────────────────
def energy_bar_chart(current_stage: str | None = None) -> go.Figure:
    """
    Horizontal bar chart comparing the three inference stages.
    Highlights *current_stage* with a brighter bar.
    """
    stages  = ["Rule Engine", "RoBERTa", "BERT"]
    energies = [ENERGY_RULE_ENGINE, ENERGY_SMALL_MODEL, ENERGY_LARGE_MODEL]
    colours = [STAGE_COLOURS[s] for s in stages]

    # Dim non-active bars
    if current_stage:
        opacity = [1.0 if s == current_stage else 0.35 for s in stages]
    else:
        opacity = [0.85, 0.85, 0.85]

    fig = go.Figure()
    for i, (stage, energy, colour, op) in enumerate(
        zip(stages, energies, colours, opacity)
    ):
        fig.add_trace(
            go.Bar(
                name=stage,
                x=[energy * 1e6],     # convert to µWh for readability
                y=[stage],
                orientation="h",
                marker=dict(color=colour, opacity=op, line=dict(width=0)),
                text=[f"{energy*1e6:.2f} µWh"],
                textposition="outside",
                hovertemplate=f"<b>{stage}</b><br>Energy: {energy:.6f} kWh<extra></extra>",
            )
        )

    layout = _base_layout("⚡ Energy Per Inference Stage (µWh)")
    layout.update(
        xaxis=dict(
            title="Energy (µWh)", gridcolor=GRID_COLOUR, showgrid=True, zeroline=False
        ),
        yaxis=dict(gridcolor=GRID_COLOUR, showgrid=False),
        showlegend=False,
        height=260,
    )
    fig.update_layout(**layout)
    return fig


# ── Chart 2: Inference Stage Distribution Pie Chart ───────────────────────────
def stage_distribution_pie(log_df: pd.DataFrame) -> go.Figure:
    """Pie chart of how often each stage was used across the session log."""
    if log_df.empty:
        counts = {"Rule Engine": 0, "RoBERTa": 0, "BERT": 0}
    else:
        counts = log_df["Stage"].value_counts().to_dict()
        for s in ["Rule Engine", "RoBERTa", "BERT"]:
            counts.setdefault(s, 0)

    labels  = list(counts.keys())
    values  = list(counts.values())
    colours = [STAGE_COLOURS[l] for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colours, line=dict(color=BG_COLOUR, width=2)),
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        )
    )
    layout = _base_layout("🥧 Inference Stage Distribution")
    layout.update(height=300, showlegend=True,
                  legend=dict(orientation="h", y=-0.1))
    fig.update_layout(**layout)
    return fig


# ── Chart 3: Cumulative Carbon Emissions Timeline ─────────────────────────────
def carbon_timeline(log_df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing cumulative CO₂ over the session.
    """
    fig = go.Figure()

    if not log_df.empty:
        df = log_df.copy()
        df["Cumulative CO₂ (g)"] = df["CO₂ (kg)"].cumsum() * 1000  # kg → g
        df["#"] = range(1, len(df) + 1)

        fig.add_trace(
            go.Scatter(
                x=df["#"],
                y=df["Cumulative CO₂ (g)"],
                mode="lines+markers",
                line=dict(color="#22c55e", width=2.5),
                marker=dict(size=7, color="#22c55e"),
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.12)",
                hovertemplate="Inference #%{x}<br>Cumulative CO₂: %{y:.4f} g<extra></extra>",
            )
        )
        
        # Adjust x-axis range based on number of inferences
        num_inferences = len(df)
        if num_inferences == 1:
            # For single inference, show range 0-3
            x_range = [0, 3]
        elif num_inferences <= 3:
            # For 2-3 inferences, show 0 to num+1
            x_range = [0, num_inferences + 1]
        else:
            # For more inferences, auto-scale with padding
            x_range = [0.5, num_inferences + 0.5]
        
        xaxis_config = dict(
            title="Inference #", 
            gridcolor=GRID_COLOUR, 
            dtick=1,
            range=x_range
        )
    else:
        # Empty placeholder
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines"))
        xaxis_config = dict(
            title="Inference #", 
            gridcolor=GRID_COLOUR, 
            dtick=1,
            range=[0, 5]
        )

    layout = _base_layout("🌍 Cumulative CO₂ Emissions This Session (grams)")
    layout.update(
        xaxis=xaxis_config,
        yaxis=dict(title="CO₂ (g)", gridcolor=GRID_COLOUR),
        height=280,
    )
    fig.update_layout(**layout)
    return fig


# ── Green Score Gauge ─────────────────────────────────────────────────────────
def green_score_gauge(score: int) -> go.Figure:
    """
    Gauge chart for the 0–100 Green Score.
    """
    colour = (
        "#22c55e" if score >= 80
        else "#eab308" if score >= 50
        else "#ef4444"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(color=colour, size=40), suffix=" / 100"),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=TEXT_COLOUR,
                    tickfont=dict(color=TEXT_COLOUR),
                ),
                bar=dict(color=colour),
                bgcolor=CARD_COLOUR,
                borderwidth=0,
                steps=[
                    dict(range=[0, 50],  color="#1e293b"),
                    dict(range=[50, 80], color="#1e293b"),
                    dict(range=[80, 100], color="#1e293b"),
                ],
                threshold=dict(
                    line=dict(color=colour, width=3),
                    thickness=0.8,
                    value=score,
                ),
            ),
            title=dict(text="🌿 Green Score", font=dict(color=TEXT_COLOUR, size=16)),
        )
    )
    fig.update_layout(
        paper_bgcolor=CARD_COLOUR,
        font=dict(color=TEXT_COLOUR),
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ── Helper: build inference log DataFrame ─────────────────────────────────────
def build_log_df(log: list[dict]) -> pd.DataFrame:
    """Convert the session inference log list into a display DataFrame."""
    if not log:
        return pd.DataFrame(columns=["Input", "Prediction", "Stage",
                                      "Confidence", "Energy (kWh)", "CO₂ (kg)"])
    rows = []
    for entry in log:
        rows.append({
            "Input":        entry["text"][:60] + ("…" if len(entry["text"]) > 60 else ""),
            "Prediction":   entry["label"],
            "Stage":        entry["stage"],
            "Confidence":   f"{entry['confidence']:.2%}",
            "Energy (kWh)": f"{entry['energy_kwh']:.6f}",
            "CO₂ (kg)":     entry["co2_kg"],
        })
    return pd.DataFrame(rows)
