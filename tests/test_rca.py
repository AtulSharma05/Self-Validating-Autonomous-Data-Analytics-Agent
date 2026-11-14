"""
Unit and integration tests for the Diagnostic Root-Cause Analysis (RCA) Engine (Phase 5 / Feature 9).
"""

from pathlib import Path
from unittest.mock import MagicMock
import duckdb
import pytest

from src.agent.graph import create_analytics_graph
from src.agent.rca import RCAEngine
from src.agent.schemas import AnalysisObjective, AnalysisPlan, AnalysisType, PlanStep
from src.agent.state import AnalyticsAgentState
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


@pytest.fixture
def rca_con():
    """In-memory DuckDB connection loaded with synthetic retail sales."""
    engine = DataIngestionEngine(session_id="rca_test_sess")
    engine.load_dataset(str(DATA_PATH))
    yield engine.con
    engine.close()


def test_rca_variance_decomposition(rca_con):
    """Test dimensional variance decomposition across regions."""
    # Compare Q1 2023 vs Q2 2023
    m_base, m_test, delta, delta_pct, dim_factors, primary_factors = RCAEngine.decompose_variance(
        con=rca_con,
        table_name="sample_retail_sales",
        metric_col="revenue",
        dimensions=["region", "product_category"],
        baseline_where="order_date BETWEEN '2023-01-01' AND '2023-03-31'",
        test_where="order_date BETWEEN '2023-04-01' AND '2023-06-30'",
    )

    assert m_base > 0
    assert m_test > 0
    assert round(m_test - m_base, 2) == round(delta, 2)
    assert "region" in dim_factors
    assert "product_category" in dim_factors

    # Check sum of dimensional deltas equals overall delta
    region_deltas_sum = sum(f.delta for f in dim_factors["region"])
    assert region_deltas_sum == pytest.approx(delta, abs=0.5)

    category_deltas_sum = sum(f.delta for f in dim_factors["product_category"])
    assert category_deltas_sum == pytest.approx(delta, abs=0.5)

    # All primary factors must have |contribution_pct| >= 10%
    for pf in primary_factors:
        assert abs(pf.contribution_pct) >= 10.0
        assert pf.is_primary is True


def test_rca_volume_price_decomposition(rca_con):
    """
    Test Volume vs. Price decomposition formula:
        ΔRevenue = (Q_T - Q_B) * P_B + (P_T - P_B) * Q_T
    """
    vp = RCAEngine.decompose_volume_price(
        con=rca_con,
        table_name="sample_retail_sales",
        revenue_col="revenue",
        volume_col="units_sold",
        baseline_where="order_date BETWEEN '2023-01-01' AND '2023-06-30'",
        test_where="order_date BETWEEN '2023-07-01' AND '2023-12-31'",
    )

    assert vp is not None
    assert vp.baseline_volume > 0
    assert vp.test_volume > 0
    assert vp.baseline_avg_price > 0
    assert vp.test_avg_price > 0

    # Mathematical identity check: Volume Effect + Price Effect == Total Delta
    reconciled_delta = vp.volume_effect + vp.price_effect
    assert reconciled_delta == pytest.approx(vp.total_delta, abs=0.05)


def test_rca_full_analysis_report(rca_con):
    """Test complete RCA report generation and narrative synthesis."""
    report = RCAEngine.analyze(
        con=rca_con,
        table_name="sample_retail_sales",
        metric_col="revenue",
        dimensions=["region", "product_category"],
        baseline_where="order_date < '2023-07-01'",
        test_where="order_date >= '2023-07-01'",
        volume_col="units_sold",
    )

    assert report.target_metric == "revenue"
    assert report.volume_price is not None
    assert len(report.primary_factors) > 0
    assert "revenue" in report.narrative_summary.lower()
    assert "Decomposition (Volume vs Price)" in report.narrative_summary


def test_langgraph_diagnostic_rca_workflow(rca_con):
    """
    Integration test: verify that a DIAGNOSTIC question triggers the RCA node in LangGraph.
    """
    profile = AutomatedDataProfiler.profile_dataset(rca_con)

    mock_llm = MagicMock()
    code_resp = MagicMock()
    code_resp.content = """```sql
    SELECT CAST(SUM("revenue") AS DOUBLE) AS total_revenue
    FROM sample_retail_sales
    WHERE order_date BETWEEN '2023-01-01' AND '2023-06-30'
    ```"""
    mock_llm.invoke.return_value = code_resp

    graph = create_analytics_graph(rca_con, llm=mock_llm)

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DIAGNOSTIC,
        target_metric="SUM(revenue)",
        target_table="sample_retail_sales",
        time_column="order_date",
        baseline_period="order_date < '2023-07-01'",
        test_period="order_date >= '2023-07-01'",
        candidate_dimensions=["region", "product_category"]
    )

    plan = AnalysisPlan(
        reasoning="Examine revenue drop and attribute cause",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Calculate baseline revenue",
                metric_col="revenue",
                agg_func="SUM",
                where_clause="order_date < '2023-07-01'"
            )
        ]
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "rca_graph_sess",
        "user_question": "Why did sales drop in H2 compared to H1?",
        "profile": profile,
        "objective": objective,
        "plan": plan,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPLETED"
    assert final_state["is_diagnostic"] is True
    assert final_state.get("rca_report") is not None
    assert "Diagnostic Root-Cause Attribution" in final_state["final_summary"]

    # Verify evidence records include the RCA attribution step
    actions = [ev.action_type for ev in final_state["evidence_records"]]
    assert "ROOT_CAUSE_ANALYSIS" in actions

