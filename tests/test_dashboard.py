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


def test_dashboard_renders_analysis_result_with_plan():
    """Regression test: verifies that dashboard does not crash with NameError: name 'plan' is not defined."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from src.agent.schemas import AnalysisObjective, AnalysisPlan, AnalysisType, PlanStep
    from src.api.session import session_manager
    from src.dashboard.app import SAMPLE_CSV

    app_path = Path(__file__).resolve().parent.parent / "src" / "dashboard" / "app.py"
    engine = session_manager.get_engine("dashboard_test_sess")
    engine.load_dataset(str(SAMPLE_CSV), table_name="sample_retail_sales")

    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["session_id"] = "dashboard_test_sess"
    at.session_state["active_table"] = "sample_retail_sales"
    at.session_state["analysis_result"] = {
        "session_id": "dashboard_test_sess",
        "user_question": "What is total count and sales?",
        "objective": AnalysisObjective(
            analysis_type=AnalysisType.DESCRIPTIVE,
            target_metric="revenue",
            target_table="sample_retail_sales",
        ),
        "plan": AnalysisPlan(
            reasoning="Multi-step test execution",
            steps=[
                PlanStep(step_id="step_1", description="Calculate order count", metric_col="order_id", agg_func="COUNT_DISTINCT", completed=True),
                PlanStep(step_id="step_2", description="Calculate total revenue", metric_col="revenue", agg_func="SUM", completed=True),
                PlanStep(step_id="step_3", description="Calculate Average Order Value (AOV)", metric_col="revenue", agg_func="AOV", completed=True),
            ],
        ),
        "step_results": {
            "step_1": {"scalar": 100.0, "dataframe": None, "success": True},
            "step_2": {"scalar": 5000.0, "dataframe": None, "success": True},
            "step_3": {"scalar": 50.0, "dataframe": None, "success": True},
        },
        "evidence_records": [],
        "correction_history": [],
    }
    at.run()
    assert not at.exception, f"Dashboard crashed during rendering: {at.exception}"


def test_dashboard_handles_missing_or_empty_plan_safely():
    """Verifies that dashboard safely renders when plan is None or steps list is empty."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from src.agent.schemas import AnalysisObjective, AnalysisPlan, AnalysisType, PlanStep
    from src.api.session import session_manager
    from src.dashboard.app import SAMPLE_CSV

    app_path = Path(__file__).resolve().parent.parent / "src" / "dashboard" / "app.py"
    engine = session_manager.get_engine("dashboard_test_empty_sess")
    engine.load_dataset(str(SAMPLE_CSV), table_name="sample_retail_sales")

    at = AppTest.from_file(str(app_path), default_timeout=10)
    at.session_state["session_id"] = "dashboard_test_empty_sess"
    at.session_state["active_table"] = "sample_retail_sales"
    at.session_state["analysis_result"] = {
        "session_id": "dashboard_test_empty_sess",
        "user_question": "What is total revenue?",
        "objective": AnalysisObjective(
            analysis_type=AnalysisType.DESCRIPTIVE,
            target_metric="revenue",
            target_table="sample_retail_sales",
        ),
        "plan": None,
        "step_results": {
            "step_1": {"scalar": 1000.0, "dataframe": None, "success": True},
        },
        "evidence_records": [],
        "correction_history": [],
    }
    at.run()
    assert not at.exception, f"Dashboard crashed with missing plan: {at.exception}"

    at2 = AppTest.from_file(str(app_path), default_timeout=10)
    at2.session_state["session_id"] = "dashboard_test_empty_sess"
    at2.session_state["active_table"] = "sample_retail_sales"
    at2.session_state["analysis_result"] = {
        "session_id": "dashboard_test_empty_sess",
        "user_question": "What is total revenue?",
        "objective": AnalysisObjective(
            analysis_type=AnalysisType.DESCRIPTIVE,
            target_metric="revenue",
            target_table="sample_retail_sales",
        ),
        "plan": AnalysisPlan(reasoning="Empty plan test", steps=[]),
        "step_results": {
            "step_1": {"scalar": 1000.0, "dataframe": None, "success": True},
        },
        "evidence_records": [],
        "correction_history": [],
    }
    at2.run()
    assert not at2.exception, f"Dashboard crashed with empty steps list: {at2.exception}"

