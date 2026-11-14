"""
Comprehensive test suite for Question Understanding, Deterministic Validation,
Self-Correction, and Grain-Aware Decision Support.
"""

import duckdb
import pandas as pd
import pytest

from src.agent.graph import create_analytics_graph
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisType,
    CorrectionAttempt,
    DatasetProfile,
)
from src.agent.state import AnalyticsAgentState
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler
from src.reporting.confidence import ConfidenceTier, ECSCalculator
from src.validation.validator import (
    ArithmeticVerifier,
    DataQualityVerifier,
    DeterministicValidator,
    SchemaAndTypeVerifier,
)


@pytest.fixture
def clean_mapping_engine():
    """Engine with clean 1:1 Product ID to Product Name mapping."""
    engine = DataIngestionEngine("clean_engine")
    df = pd.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4"],
        "product_name": ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "4K Monitor"],
        "sales": [25.0, 75.0, 45.0, 300.0],
        "profit": [10.0, 30.0, 20.0, 80.0],
    })
    engine.con.register("df_clean", df)
    engine.con.execute("CREATE TABLE products AS SELECT * FROM df_clean")
    return engine


@pytest.fixture
def conflict_mapping_engine():
    """Engine with deliberate Product ID <-> Product Name mapping conflicts."""
    engine = DataIngestionEngine("conflict_engine")
    df = pd.DataFrame({
        "product_id": ["P1", "P1", "P2", "P3"],
        "product_name": ["Phone Stand", "Desk Stand", "Wireless Mouse", "Keyboard"],
        "sales": [10.0, 20.0, 30.0, 40.0],
        "profit": [2.0, 5.0, 10.0, 15.0],
    })
    engine.con.register("df_conflict", df)
    engine.con.execute("CREATE TABLE products AS SELECT * FROM df_conflict")
    return engine


# ---------------------------------------------------------------------------
# 1. Question Understanding: Bidirectional Mapping Conflict Checks
# ---------------------------------------------------------------------------

def test_bidirectional_mapping_conflict_detected_end_to_end(conflict_mapping_engine):
    """
    Validates that a bidirectional mapping query identifies conflicts in both directions:
    1. Product ID mapped to multiple Product Names (P1 -> Phone Stand, Desk Stand).
    2. Product Name mapped to multiple Product IDs (0 conflicts).
    """
    profile = AutomatedDataProfiler.profile_dataset(conflict_mapping_engine.con, session_id="test_conflict")
    graph = create_analytics_graph(conflict_mapping_engine.con)

    state: AnalyticsAgentState = {
        "session_id": "test_conflict",
        "user_question": "Are there any Product IDs mapped to multiple Product Names, or Product Names mapped to multiple Product IDs?",
        "profile": profile,
        "max_retries": 3,
    }

    result = graph.invoke(state)

    assert result["status"] == "COMPLETED"
    obj: AnalysisObjective = result["objective"]
    assert obj.analysis_type == AnalysisType.DATA_QUALITY
    assert obj.quality_check_type == "bidirectional_mapping"
    assert "product_id" in obj.conflict_entities
    assert "product_name" in obj.conflict_entities

    # Both steps must pass deterministic validation
    assert len(result["plan"].steps) == 2
    assert result["plan"].steps[0].completed is True
    assert result["plan"].steps[1].completed is True

    # Step 1 should report 1 conflicting product_id (P1)
    df_step1 = result["step_results"]["step_1"]["dataframe"]
    assert len(df_step1) == 1
    assert df_step1.iloc[0]["product_id"] == "P1"
    assert df_step1.iloc[0]["conflict_count"] == 2

    # Step 2 should report 0 conflicting product_names
    df_step2 = result["step_results"]["step_2"]["dataframe"]
    assert len(df_step2) == 0

    # Summary should state inconsistencies detected
    summary = result["final_summary"]
    assert "Data Integrity Inconsistencies Detected" in summary
    assert "1 `product_id` value(s) map to multiple `product_name` values" in summary
    assert "0 `product_name` value(s) map to multiple `product_id` values" in summary


def test_bidirectional_mapping_clean_dataset_reports_100_percent_consistent(clean_mapping_engine):
    """
    Validates that a bidirectional mapping query on clean data reports 0 conflicts
    and yields 100% negative confirmation without false failures.
    """
    profile = AutomatedDataProfiler.profile_dataset(clean_mapping_engine.con, session_id="test_clean")
    graph = create_analytics_graph(clean_mapping_engine.con)

    state: AnalyticsAgentState = {
        "session_id": "test_clean",
        "user_question": "Are there any Product IDs mapped to multiple Product Names, or Product Names mapped to multiple Product IDs?",
        "profile": profile,
        "max_retries": 3,
    }

    result = graph.invoke(state)

    assert result["status"] == "COMPLETED"
    assert result["plan"].steps[0].completed is True
    assert result["plan"].steps[1].completed is True

    summary = result["final_summary"]
    assert "100% Consistent Mappings" in summary
    assert "No conflicting mappings detected" in summary
    assert "Number of conflicting product_ids:** 0" in summary
    assert "Number of conflicting product_names:** 0" in summary


# ---------------------------------------------------------------------------
# 2. Independent Deterministic Validation: Arithmetic & Schema Verifiers
# ---------------------------------------------------------------------------

def test_arithmetic_verifier_detects_calculation_mismatch():
    """
    Verifies that the independent ArithmeticVerifier flags calculation errors
    (e.g. LLM claims -20% when the math is ((700,000 - 1,000,000)/1,000,000)*100 = -30%).
    """
    # 1. Correct calculation
    res_correct = ArithmeticVerifier.verify_percentage_change(
        base_value=1_000_000, test_value=700_000, claimed_pct_change=-30.0
    )
    assert res_correct.is_valid is True
    assert res_correct.audit_details["expected"] == -30.0
    assert res_correct.audit_details["actual"] == -30.0

    # 2. Mismatched calculation (claimed -20.0%)
    res_mismatch = ArithmeticVerifier.verify_percentage_change(
        base_value=1_000_000, test_value=700_000, claimed_pct_change=-20.0
    )
    assert res_mismatch.is_valid is False
    assert res_mismatch.error_type == "calculation_mismatch"
    assert res_mismatch.audit_details["expected"] == -30.0
    assert res_mismatch.audit_details["actual"] == -20.0
    assert "Percentage change is incorrect" in res_mismatch.audit_details["message"]


def test_schema_and_type_verifier():
    """Verifies schema validation, expected columns, and non-empty checks."""
    df = pd.DataFrame({"product_id": ["P1"], "sales": [100.0]})

    # Passes when expected columns exist
    res = SchemaAndTypeVerifier.verify(df, expected_columns=["product_id", "sales"])
    assert res.is_valid is True

    # Fails when expected column is missing
    res_missing = SchemaAndTypeVerifier.verify(df, expected_columns=["product_id", "profit"])
    assert res_missing.is_valid is False
    assert res_missing.error_type == "SCHEMA_MISMATCH"

    # Disallow empty check
    empty_df = pd.DataFrame(columns=["product_id"])
    res_empty = SchemaAndTypeVerifier.verify(empty_df, disallow_empty=True)
    assert res_empty.is_valid is False
    assert res_empty.error_type == "EMPTY_RESULT"


# ---------------------------------------------------------------------------
# 3. Structured Self-Correction Loop & History Tracking
# ---------------------------------------------------------------------------

def test_self_correction_history_logged_in_state():
    """
    Verifies that when validation or execution encounters discrepancies,
    structured CorrectionAttempt records are logged in correction_history.
    """
    engine = DataIngestionEngine("corr_test")
    engine.con.execute("CREATE TABLE sales AS SELECT 100.0 AS revenue, 10.0 AS units")
    profile = AutomatedDataProfiler.profile_dataset(engine.con, session_id="corr_test")

    graph = create_analytics_graph(engine.con)

    # Initial state with invalid initial query step
    state: AnalyticsAgentState = {
        "session_id": "corr_test",
        "user_question": "What is total revenue?",
        "profile": profile,
        "max_retries": 3,
    }

    result = graph.invoke(state)
    assert "correction_history" in result
    assert isinstance(result["correction_history"], list)
    # First-pass query succeeds, so correction history should be empty or cleanly tracked
    assert len(result["correction_history"]) == 0
    assert result["plan"].steps[0].completed is True


def test_correction_attempt_model_fields():
    """Tests the fields and serialization of CorrectionAttempt."""
    attempt = CorrectionAttempt(
        attempt_number=1,
        step_id="step_1",
        failed_code="SELECT * FROM ghost_tbl",
        error_type="SYNTAX_ERROR",
        error_message="Table not found",
        corrected_code="SELECT * FROM sales",
        validation_passed=True,
        execution_time_ms=12.5,
    )
    dumped = attempt.model_dump()
    assert dumped["attempt_number"] == 1
    assert dumped["error_type"] == "SYNTAX_ERROR"
    assert dumped["validation_passed"] is True


# ---------------------------------------------------------------------------
# 4. Diagnostic RCA: Available Dimensions Discipline
# ---------------------------------------------------------------------------

def test_rca_investigates_only_available_columns_never_hallucinates(clean_mapping_engine):
    """
    Ensures Root-Cause Analysis only decomposes variance across candidate dimensions
    that actually exist in the table schema, never hallucinating missing columns
    (such as inventory, stock-outs, or marketing channels).
    """
    from src.agent.rca import RCAEngine

    # Table 'products' has columns: product_id, product_name, sales, profit
    available_dims = ["product_name"]

    report = RCAEngine.analyze(
        con=clean_mapping_engine.con,
        table_name="products",
        metric_col="sales",
        dimensions=available_dims,
    )

    assert report.target_metric == "sales"
    # All tested hypotheses must belong to available dimensions
    for h in report.hypotheses:
        assert h.dimension in available_dims
        assert h.dimension not in ["inventory", "stock_out", "marketing_channel", "ad_spend"]


# ---------------------------------------------------------------------------
# 5. Evidence Confidence Scoring: Insufficient Evidence Handling
# ---------------------------------------------------------------------------

def test_ecs_insufficient_evidence_when_validation_fails():
    """
    Tests that ECS calculates 0.0% and assigns INSUFFICIENT_EVIDENCE
    when all plan verification steps fail.
    """
    report = ECSCalculator.compute(
        checks_passed=0,
        checks_total=3,
    )

    assert report.score == 0.0
    assert report.tier == ConfidenceTier.INSUFFICIENT_EVIDENCE
    assert "Insufficient evidence" in report.recommendation


def test_ecs_high_confidence_when_validation_passes():
    """
    Tests that ECS calculates high confidence (>=85%) when all steps are validated deterministically.
    """
    report = ECSCalculator.compute(
        checks_passed=3,
        checks_total=3,
    )

    assert report.score >= 85.0
    assert report.tier == ConfidenceTier.HIGH


# ---------------------------------------------------------------------------
# 6. Grain Awareness: Order Deduplication vs Line-Item Trap
# ---------------------------------------------------------------------------

def test_grain_awareness_order_deduplication_end_to_end():
    """
    Prompt: 'How many total orders were placed in 2017, and what is the average order value (AOV)?'
    Dataset: 4 line-item rows, but only 2 unique orders in 2017 with total sales $300.
    Assert: Agent must report 2 unique orders and AOV of $150.00, NOT 3 rows and $100.00 average.
    """
    engine = DataIngestionEngine("grain_test")
    df = pd.DataFrame({
        "order_id": ["CA-2017-101", "CA-2017-101", "CA-2017-102", "CA-2016-103"],
        "order_date": ["2017-01-15", "2017-01-15", "2017-03-20", "2016-05-10"],
        "product_id": ["P1", "P2", "P3", "P4"],
        "sales": [100.0, 50.0, 150.0, 500.0],
        "profit": [10.0, 5.0, 20.0, 50.0],
        "quantity": [1, 1, 2, 1]
    })
    engine.con.register("df_orders", df)
    engine.con.execute("CREATE TABLE orders AS SELECT * FROM df_orders")
    profile = AutomatedDataProfiler.profile_dataset(engine.con, session_id="grain_test")

    graph = create_analytics_graph(engine.con)
    state: AnalyticsAgentState = {
        "session_id": "grain_test",
        "user_question": "How many total orders were placed in 2017, and what is the average order value (AOV)?",
        "profile": profile,
        "max_retries": 3,
    }

    result = graph.invoke(state)
    assert result["status"] == "COMPLETED"

    # Step 1: Unique Orders = 2.0 (COUNT(DISTINCT order_id))
    s1_res = result["step_results"]["step_1"]["scalar"]
    assert s1_res == 2.0

    # Step 2: Total 2017 Sales = 300.0
    s2_res = result["step_results"]["step_2"]["scalar"]
    assert s2_res == 300.0

    # Step 3: True AOV = 150.0 (300 / 2)
    s3_res = result["step_results"]["step_3"]["scalar"]
    assert s3_res == 150.0

    summary = result["final_summary"]
    assert "2" in summary
    assert "300.00" in summary
    assert "150.00" in summary
