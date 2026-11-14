"""
Unit tests for the dashboard Plotly visualization generators (Phase 8 / Feature 12).
"""

import plotly.graph_objects as go
import pytest

from src.agent.schemas import FactorAttribution, VolumePriceDecomposition
from src.dashboard.charts import (
    create_factor_bar_chart,
    create_volume_price_chart,
    create_waterfall_chart,
)


def test_create_waterfall_chart():
    factors = [
        FactorAttribution(
            dimension="region",
            category="West",
            baseline_value=1000.0,
            test_value=600.0,
            delta=-400.0,
            contribution_pct=80.0,
            is_primary=True,
        ),
        FactorAttribution(
            dimension="region",
            category="East",
            baseline_value=500.0,
            test_value=400.0,
            delta=-100.0,
            contribution_pct=20.0,
            is_primary=True,
        ),
    ]

    fig = create_waterfall_chart(
        baseline_value=1500.0,
        test_value=1000.0,
        factors=factors,
        metric_name="Revenue",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    wf = fig.data[0]
    assert "Baseline" in wf.x
    assert "Current Period" in wf.x
    assert wf.y[0] == 1500.0


def test_create_volume_price_chart():
    vp = VolumePriceDecomposition(
        baseline_volume=100.0,
        test_volume=80.0,
        baseline_avg_price=10.0,
        test_avg_price=9.0,
        volume_effect=-200.0,
        price_effect=-80.0,
        total_delta=-280.0,
        volume_contribution_pct=71.4,
        price_contribution_pct=28.6,
    )

    fig = create_volume_price_chart(vp)
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert "Volume Effect" in bar.x
    assert "Price Effect" in bar.x
    assert "Total Delta" in bar.x


def test_create_factor_bar_chart():
    factors = [
        FactorAttribution(
            dimension="category",
            category="Electronics",
            delta=-300.0,
            contribution_pct=60.0,
            is_primary=True,
        ),
        FactorAttribution(
            dimension="category",
            category="Apparel",
            delta=100.0,
            contribution_pct=-20.0,
            is_primary=True,
        ),
    ]

    fig = create_factor_bar_chart(factors, dimension_name="Product Category")
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert "Electronics" in bar.y
    assert "Apparel" in bar.y

