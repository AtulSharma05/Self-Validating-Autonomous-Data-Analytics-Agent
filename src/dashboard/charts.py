"""
Interactive Plotly Chart Generators for the Self-Validating Analytics Dashboard (Feature 12).
"""

from __future__ import annotations

from typing import List, Optional
import plotly.graph_objects as go

from src.agent.schemas import FactorAttribution, VolumePriceDecomposition


def create_waterfall_chart(
    baseline_value: float,
    test_value: float,
    factors: List[FactorAttribution],
    metric_name: str = "Metric",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Constructs an interactive Plotly Waterfall chart demonstrating how the baseline metric
    flows to the current test period metric through individual contributing factors.
    """
    x_labels = ["Baseline"]
    y_values = [baseline_value]
    measures = ["absolute"]
    texts = [f"{baseline_value:,.2f}"]

    total_delta = test_value - baseline_value
    sum_factor_deltas = 0.0

    # Include top primary factors
    top_factors = factors[:6]
    for f in top_factors:
        x_labels.append(f"{f.dimension}: {f.category}")
        y_values.append(f.delta)
        measures.append("relative")
        texts.append(f"{f.delta:+,.2f} ({f.contribution_pct:+.1f}%)")
        sum_factor_deltas += f.delta

    # If there is a residual between top factors and total delta, show "Other Factors"
    residual = total_delta - sum_factor_deltas
    if abs(residual) > 1e-2 and len(factors) > len(top_factors):
        x_labels.append("Other Factors")
        y_values.append(residual)
        measures.append("relative")
        other_pct = (residual / total_delta * 100.0) if total_delta != 0 else 0.0
        texts.append(f"{residual:+,.2f} ({other_pct:+.1f}%)")

    # Final Total
    x_labels.append("Current Period")
    y_values.append(test_value)
    measures.append("total")
    texts.append(f"{test_value:,.2f}")

    fig = go.Figure(
        go.Waterfall(
            name=metric_name,
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=texts,
            y=y_values,
            connector={"line": {"color": "#94A3B8", "width": 1.5, "dash": "dot"}},
            increasing={"marker": {"color": "#10B981"}},   # Emerald green
            decreasing={"marker": {"color": "#EF4444"}},   # Rose red
            totals={"marker": {"color": "#3B82F6"}},       # Royal blue
        )
    )

    chart_title = title or f"{metric_name} Variance Waterfall Decomposition"
    fig.update_layout(
        title={"text": chart_title, "x": 0.02, "xanchor": "left"},
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={"tickangle": -25, "gridcolor": "rgba(148, 163, 184, 0.15)"},
        yaxis={"gridcolor": "rgba(148, 163, 184, 0.15)", "zeroline": True},
        font={"family": "system-ui, -apple-system, sans-serif"},
    )
    return fig


def create_volume_price_chart(
    vp: VolumePriceDecomposition,
    title: str = "Volume Effect vs. Price Effect Decomposition",
) -> go.Figure:
    """
    Renders a comparative bar chart showing physical volume impact vs. average price impact.
    """
    effects = ["Volume Effect", "Price Effect", "Total Delta"]
    values = [vp.volume_effect, vp.price_effect, vp.total_delta]
    colors = [
        "#10B981" if vp.volume_effect >= 0 else "#EF4444",
        "#10B981" if vp.price_effect >= 0 else "#EF4444",
        "#3B82F6" if vp.total_delta >= 0 else "#6366F1",
    ]
    texts = [
        f"{vp.volume_effect:+,.2f} ({vp.volume_contribution_pct:+.1f}%)",
        f"{vp.price_effect:+,.2f} ({vp.price_contribution_pct:+.1f}%)",
        f"{vp.total_delta:+,.2f}",
    ]

    fig = go.Figure(
        go.Bar(
            x=effects,
            y=values,
            marker_color=colors,
            text=texts,
            textposition="auto",
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        yaxis={"gridcolor": "rgba(148, 163, 184, 0.15)", "zeroline": True},
        font={"family": "system-ui, -apple-system, sans-serif"},
    )
    return fig


def create_factor_bar_chart(
    factors: List[FactorAttribution],
    dimension_name: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Renders a horizontal bar chart displaying category contributions to total metric delta.
    """
    sorted_factors = sorted(factors, key=lambda x: x.delta)
    categories = [f.category for f in sorted_factors]
    deltas = [f.delta for f in sorted_factors]
    colors = ["#10B981" if d >= 0 else "#EF4444" for d in deltas]
    texts = [f"{f.delta:+,.2f} ({f.contribution_pct:+.1f}%)" for f in sorted_factors]

    fig = go.Figure(
        go.Bar(
            y=categories,
            x=deltas,
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="auto",
        )
    )

    chart_title = title or f"Variance Breakdown by {dimension_name.replace('_', ' ').title()}"
    fig.update_layout(
        title={"text": chart_title, "x": 0.02, "xanchor": "left"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={"gridcolor": "rgba(148, 163, 184, 0.15)", "zeroline": True},
        font={"family": "system-ui, -apple-system, sans-serif"},
    )
    return fig

