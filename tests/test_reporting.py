"""
Unit and integration tests for Provenance DAG and Evidence Confidence Scoring (Phase 6 / Features 10 & 11).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.agent.graph import create_analytics_graph
from src.agent.schemas import AnalysisObjective, AnalysisPlan, AnalysisType, ColumnProfile, DatasetProfile, FactorAttribution, PlanStep, RCAReport, TableProfile
from src.agent.state import AnalyticsAgentState
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler
from src.reporting.confidence import ConfidenceTier, ECSCalculator
from src.reporting.provenance import EvidenceArtifact, ProvenanceDAG

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


# -------------------------------------------------------------------------
# Feature 10: Provenance DAG Tests
# -------------------------------------------------------------------------

def test_evidence_artifact_checksum():
    q1 = "SELECT SUM(revenue) FROM sales"
    q2 = "SELECT AVG(revenue) FROM sales"

    c1 = EvidenceArtifact.compute_checksum(q1)
    c2 = EvidenceArtifact.compute_checksum(q1)
    c3 = EvidenceArtifact.compute_checksum(q2)

    assert c1 == c2  # Deterministic
    assert c1 != c3  # Different queries yield different hashes
    assert len(c1) == 16


def test_provenance_dag_lineage_and_structure():
    dag = ProvenanceDAG(session_id="test_dag")

    a1 = EvidenceArtifact(
        step_id="step_1",
        action_type="LOAD_DATA",
        query_sql="LOAD 'sales.csv'",
        result_summary="Loaded 1000 rows",
        checksum_hash="hash_step_1",
        validation_status="VALIDATED_DETERMINISTICALLY",
        execution_time_ms=12.5,
    )
    a2 = EvidenceArtifact(
        step_id="step_2",
        action_type="AGGREGATION",
        query_sql="SELECT SUM(revenue) FROM sales",
        result_summary="Total = 50000",
        checksum_hash="hash_step_2",
        validation_status="VALIDATED_DETERMINISTICALLY",
        execution_time_ms=4.2,
        parent_step_ids=["step_1"],
    )
    a3 = EvidenceArtifact(
        step_id="step_3",
        action_type="RCA",
        query_sql="RCA over region",
        result_summary="Top factor = West",
        checksum_hash="hash_step_3",
        validation_status="VALIDATED_DETERMINISTICALLY",
        execution_time_ms=8.1,
        parent_step_ids=["step_2"],
    )

    dag.add_artifact(a1)
    dag.add_artifact(a2)
    dag.add_artifact(a3)

    assert dag.verify_acyclic() is True
    assert dag.get_artifact("step_2").result_summary == "Total = 50000"

    # Lineage for step_3 should be: step_1 -> step_2 -> step_3
    lineage = dag.get_lineage("step_3")
    assert len(lineage) == 3
    assert [a.step_id for a in lineage] == ["step_1", "step_2", "step_3"]

    # Serialization
    dag_dict = dag.to_dict()
    assert dag_dict["node_count"] == 3
    assert dag_dict["edge_count"] == 2
    assert dag_dict["is_acyclic"] is True


# -------------------------------------------------------------------------
# Feature 11: ECS Confidence Scoring Tests
# -------------------------------------------------------------------------

def test_ecs_high_confidence_case():
    profile = DatasetProfile(
        session_id="test",
        total_tables=1,
        tables=[
            TableProfile(
                table_name="sales",
                row_count=1000,
                column_count=2,
                columns=[
                    ColumnProfile(name="revenue", data_type="DOUBLE", is_numeric=True, null_percentage=0.0),
                ]
            )
        ]
    )

    report = ECSCalculator.compute(
        checks_passed=4,
        checks_total=4,
        profile=profile,
        key_columns=["revenue"],
        unresolved_discrepancy=False,
    )

    assert report.score >= 75.0
    assert report.tier == ConfidenceTier.HIGH
    assert report.badge_color == "#10B981"
    assert "validation_score" in report.components


def test_ecs_low_confidence_with_contradiction_penalty():
    report = ECSCalculator.compute(
        checks_passed=1,
        checks_total=4,
        unresolved_discrepancy=True,
    )

    assert report.score < 50.0
    assert report.tier == ConfidenceTier.LOW
    assert report.badge_color == "#EF4444"
    assert report.components["contradiction_penalty"].raw_value == 1.0


def test_ecs_with_rca_explanatory_power():
    rca = RCAReport(
        target_metric="revenue",
        baseline_total=1000.0,
        test_total=800.0,
        total_delta=-200.0,
        delta_pct=-20.0,
        primary_factors=[
            FactorAttribution(
                dimension="region",
                category="West",
                delta=-180.0,
                contribution_pct=90.0,
                is_primary=True,
            )
        ]
    )

    report = ECSCalculator.compute(
        checks_passed=3,
        checks_total=3,
        rca_report=rca,
    )

    # 180 / 200 = 0.9 explanatory power
    assert report.components["attribution_power"].raw_value == pytest.approx(0.9, abs=0.05)
    assert report.score >= 75.0


# -------------------------------------------------------------------------
# LangGraph Integration Test with Provenance & ECS
# -------------------------------------------------------------------------

def test_langgraph_emits_provenance_and_ecs():
    engine = DataIngestionEngine(session_id="reporting_e2e")
    engine.load_dataset(str(DATA_PATH))
    profile = AutomatedDataProfiler.profile_dataset(engine.con)

    mock_llm = MagicMock()
    code_resp = MagicMock()
    code_resp.content = """```sql
    SELECT CAST(SUM("revenue") AS DOUBLE) AS total_revenue
    FROM sample_retail_sales
    ```"""
    mock_llm.invoke.return_value = code_resp

    graph = create_analytics_graph(engine.con, llm=mock_llm)

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sample_retail_sales",
    )
    plan = AnalysisPlan(
        reasoning="Overall total",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Compute revenue",
                metric_col="revenue",
                agg_func="SUM",
                where_clause="1=1"
            )
        ]
    )

    state: AnalyticsAgentState = {
        "session_id": "reporting_sess",
        "user_question": "What is total sales?",
        "profile": profile,
        "objective": objective,
        "plan": plan,
    }

    final_state = graph.invoke(state)

    assert final_state["status"] == "COMPLETED"

    # Verify ECS Report is present and computed
    ecs = final_state.get("ecs_report")
    assert ecs is not None
    assert 0.0 <= ecs.score <= 100.0
    assert ecs.tier in [ConfidenceTier.HIGH, ConfidenceTier.MODERATE, ConfidenceTier.LOW]
    assert "Evidence Confidence Score (ECS)" in final_state["final_summary"]

    # Verify Provenance DAG is present and acyclic
    prov = final_state.get("provenance_dag")
    assert prov is not None
    assert prov["node_count"] >= 1
    assert prov["is_acyclic"] is True

    engine.close()

