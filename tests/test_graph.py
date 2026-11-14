"""
Unit and integration tests for the LangGraph State Machine (Phase 4).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.agent.graph import create_analytics_graph
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    PlanStep,
)
from src.agent.state import AnalyticsAgentState
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


@pytest.fixture
def test_env():
    engine = DataIngestionEngine(session_id="test_graph_sess")
    engine.load_dataset(str(DATA_PATH))
    profile = AutomatedDataProfiler.profile_dataset(engine.con)
    yield engine.con, profile
    engine.close()


def test_graph_compilation(test_env):
    con, profile = test_env
    graph = create_analytics_graph(con)
    assert graph is not None


def test_graph_execution_single_step(test_env):
    con, profile = test_env

    mock_llm = MagicMock()
    code_resp = MagicMock()
    code_resp.content = """```sql
    SELECT CAST(SUM("revenue") AS DOUBLE) AS total_revenue
    FROM sample_retail_sales
    ```"""
    mock_llm.invoke.return_value = code_resp

    graph = create_analytics_graph(con, llm=mock_llm)

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sample_retail_sales",
        time_column="order_date",
    )

    plan = AnalysisPlan(
        reasoning="Calculate overall total",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Compute aggregate revenue",
                metric_col="revenue",
                agg_func="SUM",
                where_clause="1=1"
            )
        ]
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "test_graph_sess",
        "user_question": "What is the total revenue?",
        "profile": profile,
        "objective": objective,
        "plan": plan,
        "max_retries": 2,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPLETED"
    assert len(final_state["evidence_records"]) == 1
    assert "Analytical Findings" in final_state["final_summary"]
    assert final_state["evidence_records"][0].validation_status == "VALIDATED_DETERMINISTICALLY"


def test_graph_self_correction_cycle(test_env):
    """
    Test that when the initial code fails (bad syntax / unknown column),
    the graph routes to correct_node and heals itself!
    """
    con, profile = test_env

    mock_llm = MagicMock()

    # 1st code attempt: Bad query (triggers syntax error)
    bad_code_resp = MagicMock()
    bad_code_resp.content = """```sql
    SELECT CAST(SUM("non_existent_column") AS DOUBLE) AS total
    FROM sample_retail_sales
    ```"""

    # 2nd code attempt (correction): Good query
    good_code_resp = MagicMock()
    good_code_resp.content = """```sql
    SELECT CAST(SUM("revenue") AS DOUBLE) AS total
    FROM sample_retail_sales
    ```"""

    mock_llm.invoke.side_effect = [bad_code_resp, good_code_resp]

    graph = create_analytics_graph(con, llm=mock_llm)

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sample_retail_sales",
    )

    plan = AnalysisPlan(
        reasoning="Compute single step",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Compute aggregate revenue",
                metric_col="revenue",
                agg_func="SUM",
                where_clause="1=1"
            )
        ]
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "test_correction_sess",
        "user_question": "Calculate overall revenue",
        "profile": profile,
        "objective": objective,
        "plan": plan,
        "max_retries": 3,
    }

    final_state = graph.invoke(initial_state)

    # Should have recovered via correction!
    assert final_state["status"] == "COMPLETED"
    assert len(final_state["evidence_records"]) == 1
    assert final_state["retry_count"] == 1
    assert "Analytical Findings" in final_state["final_summary"]


def test_graph_multi_step_flow(test_env):
    """
    Test multi-step execution where step 1 is scalar total and step 2 is regional breakdown.
    """
    con, profile = test_env

    mock_llm = MagicMock()

    step1_resp = MagicMock()
    step1_resp.content = """```sql
    SELECT CAST(SUM("revenue") AS DOUBLE) AS total
    FROM sample_retail_sales
    ```"""

    step2_resp = MagicMock()
    step2_resp.content = """```sql
    SELECT "region", CAST(SUM("revenue") AS DOUBLE) AS region_total
    FROM sample_retail_sales
    GROUP BY "region"
    ```"""

    mock_llm.invoke.side_effect = [step1_resp, step2_resp]

    graph = create_analytics_graph(con, llm=mock_llm)

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sample_retail_sales",
        candidate_dimensions=["region"]
    )

    plan = AnalysisPlan(
        reasoning="Compute total then breakdown",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Compute overall revenue",
                metric_col="revenue",
                agg_func="SUM",
                where_clause="1=1"
            ),
            PlanStep(
                step_id="step_2",
                description="Compute regional breakdown",
                metric_col="revenue",
                agg_func="SUM",
                group_col="region",
                where_clause="1=1"
            )
        ]
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "test_multistep_sess",
        "user_question": "Show revenue overall and by region",
        "profile": profile,
        "objective": objective,
        "plan": plan,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPLETED"
    assert len(final_state["evidence_records"]) == 2
    assert final_state["current_step_index"] == 1
    assert "step_1" in final_state["step_results"]
    assert "step_2" in final_state["step_results"]
