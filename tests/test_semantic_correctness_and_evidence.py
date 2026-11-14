"""
Regression tests for Semantic Correctness, Evidence Schema, and Confidence Gating.

Covers:
  1. Total orders in a specified year (COUNT(DISTINCT order_id) with 2017 filter).
  2. Average Order Value (SUM(sales) / COUNT(DISTINCT order_id)).
  3. Multi-metric question requesting both total orders and AOV in 2017.
  4. Revenue vs. order count intent discrimination (no substitution).
  5. Profit vs. revenue intent discrimination (no substitution).
  6. Missing order ID in dataset schema (explanatory notice, no substitution).
  7. Missing date column when year filter is requested (explanatory notice).
  8. Entity deduplication over multi-row line items.
  9. Semantic coverage validator detects unauthorized substitutions.
 10. ClaimEvidence schema serialization, .to_dict() and safe rendering.
 11. ECS confidence gating when semantic coverage fails (score <= 20%, tier=LOW).
"""

import duckdb
import pytest
import pandas as pd

from src.agent.graph import create_analytics_graph
from src.agent.objective_extractor import ObjectiveExtractor
from src.agent.planner import Planner
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    ClaimEvidence,
    DatasetProfile,
    PlanStep,
    SemanticCoverageReport,
)
from src.ingestion.profiler import AutomatedDataProfiler
from src.reporting.confidence import ConfidenceTier, ECSCalculator
from src.validation.validator import DualPathVerifier, SemanticCoverageValidator


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def superstore_sample_con():
    """
    Creates a DuckDB in-memory database with representative retail data containing
    multiple line items per order across 2016 and 2017.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE sample_superstore (
            order_id VARCHAR,
            order_date DATE,
            sales DOUBLE,
            profit DOUBLE,
            quantity INT,
            category VARCHAR,
            region VARCHAR
        )
    """)
    # 2017 orders:
    # Order 1 (ORD-2017-01): 2 items ($100 + $200 = $300), profit ($20 + $40 = $60)
    # Order 2 (ORD-2017-02): 1 item ($300), profit ($50)
    # Order 3 (ORD-2017-03): 3 items ($50 + $50 + $100 = $200), profit ($10 + $10 + $20 = $40)
    # Total 2017: 6 line items, 3 unique orders, Total Sales = $800.0, AOV = 800 / 3 = $266.67, Total Profit = $150.0
    #
    # 2016 orders (should be filtered out by 2017 filter):
    # Order 4 (ORD-2016-01): 2 items ($500 + $500 = $1000)
    con.execute("""
        INSERT INTO sample_superstore VALUES
        ('ORD-2017-01', '2017-01-15', 100.0, 20.0, 1, 'Technology', 'East'),
        ('ORD-2017-01', '2017-01-15', 200.0, 40.0, 2, 'Technology', 'East'),
        ('ORD-2017-02', '2017-04-10', 300.0, 50.0, 1, 'Furniture', 'West'),
        ('ORD-2017-03', '2017-09-22',  50.0, 10.0, 1, 'Office Supplies', 'Central'),
        ('ORD-2017-03', '2017-09-22',  50.0, 10.0, 1, 'Office Supplies', 'Central'),
        ('ORD-2017-03', '2017-09-22', 100.0, 20.0, 2, 'Office Supplies', 'Central'),
        ('ORD-2016-01', '2016-03-01', 500.0, 100.0, 5, 'Technology', 'East'),
        ('ORD-2016-01', '2016-03-02', 500.0, 100.0, 5, 'Technology', 'East')
    """)
    return con


# -----------------------------------------------------------------------------
# Test 1: Total orders in a specified year
# -----------------------------------------------------------------------------
def test_total_orders_in_year_deduplicates_and_filters(superstore_sample_con):
    con = superstore_sample_con
    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_orders_2017")
    extractor = ObjectiveExtractor()

    question = "How many total orders were placed in 2017?"
    obj = extractor.extract(profile, question)

    assert obj.entity_grain == "order"
    assert "COUNT(DISTINCT" in obj.target_metric
    assert any("2017" in f for f in obj.filter_conditions)

    graph = create_analytics_graph(con)
    res = graph.invoke({
        "session_id": "test_orders_2017",
        "user_question": question,
        "profile": profile,
        "max_retries": 3,
    })

    assert res["status"] == "COMPLETED"
    # Ground truth: 3 unique orders in 2017 (not 6 line items)
    s1_res = res["step_results"]["step_1"]["scalar"]
    assert s1_res == 3.0
    assert "3" in res["final_summary"]


# -----------------------------------------------------------------------------
# Test 2: Average Order Value (AOV) uses correct formula and grain
# -----------------------------------------------------------------------------
def test_average_order_value_uses_order_grain(superstore_sample_con):
    con = superstore_sample_con
    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_aov_2017")
    extractor = ObjectiveExtractor()

    question = "What is the average order value in 2017?"
    obj = extractor.extract(profile, question)

    assert obj.entity_grain == "order"
    assert "AOV" in obj.target_metric or "COUNT(DISTINCT" in obj.target_metric

    graph = create_analytics_graph(con)
    res = graph.invoke({
        "session_id": "test_aov_2017",
        "user_question": question,
        "profile": profile,
        "max_retries": 3,
    })

    assert res["status"] == "COMPLETED"
    # Ground truth: Total 2017 sales = $800.0 / 3 orders = $266.67 (not 800/6 = $133.33)
    s_aov = res["step_results"]["step_3"]["scalar"]
    assert abs(s_aov - 266.66666667) < 0.01
    assert "266.67" in res["final_summary"]


# -----------------------------------------------------------------------------
# Test 3: Multi-metric question requesting both total orders and AOV in 2017
# -----------------------------------------------------------------------------
def test_multi_metric_orders_and_aov_generates_both_outputs(superstore_sample_con):
    con = superstore_sample_con
    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_multi_metric")

    question = "How many total orders were placed in 2017, and what is the average order value?"
    graph = create_analytics_graph(con)
    res = graph.invoke({
        "session_id": "test_multi_metric",
        "user_question": question,
        "profile": profile,
        "max_retries": 3,
    })

    assert res["status"] == "COMPLETED"
    plan = res["plan"]
    assert len(plan.steps) >= 2

    # Step 1 computes total unique orders (3)
    s1_val = res["step_results"]["step_1"]["scalar"]
    assert s1_val == 3.0

    # Step 2 computes total 2017 sales ($800.0)
    s2_val = res["step_results"]["step_2"]["scalar"]
    assert s2_val == 800.0

    # Step 3 computes AOV ($266.67)
    s3_val = res["step_results"]["step_3"]["scalar"]
    assert abs(s3_val - 266.66666667) < 0.01

    # Verify semantic coverage passed
    sem_cov: SemanticCoverageReport = res.get("semantic_coverage")
    assert sem_cov is not None
    assert sem_cov.is_covered is True
    assert "total_orders" in sem_cov.executed_metrics
    assert "average_order_value" in sem_cov.executed_metrics

    # Independent statuses
    assert res.get("execution_correctness") == "PASSED"
    assert res.get("calculation_correctness") == "PASSED"
    assert res.get("semantic_coverage_status") == "PASSED"
    assert res.get("overall_outcome") == "FULLY_VERIFIED"


# -----------------------------------------------------------------------------
# Test 4: Question requesting revenue vs order count (verifies no substitution)
# -----------------------------------------------------------------------------
def test_no_substitution_revenue_for_orders(superstore_sample_con):
    con = superstore_sample_con
    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_subst_orders")
    extractor = ObjectiveExtractor()

    question = "How many total orders were placed in 2017?"
    obj = extractor.extract(profile, question)

    # Must target order count, NOT sales or revenue
    assert "sales" not in obj.target_metric.lower() or "count(distinct" in obj.target_metric.lower()
    assert obj.entity_grain == "order"


# -----------------------------------------------------------------------------
# Test 5: Question requesting profit vs revenue (verifies no substitution)
# -----------------------------------------------------------------------------
def test_no_substitution_profit_for_revenue(superstore_sample_con):
    con = superstore_sample_con
    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_subst_profit")
    extractor = ObjectiveExtractor()

    question = "What is the total profit in 2017?"
    obj = extractor.extract(profile, question)

    assert "profit" in obj.target_metric.lower()
    assert "sales" not in obj.target_metric.lower()


# -----------------------------------------------------------------------------
# Test 6: Missing order ID in dataset schema (verifies explanatory notice)
# -----------------------------------------------------------------------------
def test_missing_order_id_produces_explanation_without_substitution():
    con = duckdb.connect(":memory:")
    # Table with sales and profit, but NO order_id column
    con.execute("""
        CREATE TABLE no_order_table (
            sales DOUBLE,
            profit DOUBLE,
            order_date DATE,
            region VARCHAR
        )
    """)
    con.execute("INSERT INTO no_order_table VALUES (100.0, 10.0, '2017-01-01', 'West')")

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_no_order_id")
    extractor = ObjectiveExtractor()

    question = "How many total orders were placed in 2017, and what is the average order value?"
    obj = extractor.extract(profile, question)

    assert obj.missing_columns_explanation is not None
    assert "order_id" in obj.missing_columns_explanation.lower()

    graph = create_analytics_graph(con)
    res = graph.invoke({
        "session_id": "test_no_order_id",
        "user_question": question,
        "profile": profile,
        "max_retries": 3,
    })

    assert res["status"] == "COMPLETED"
    assert "missing" in res["final_summary"].lower() or "unavailable" in res["final_summary"].lower()
    sem_cov = res.get("semantic_coverage")
    assert sem_cov.is_covered is False


# -----------------------------------------------------------------------------
# Test 7: Missing date column when year filter is requested
# -----------------------------------------------------------------------------
def test_missing_date_column_produces_explanation():
    con = duckdb.connect(":memory:")
    # Table with order_id and sales, but NO date column
    con.execute("""
        CREATE TABLE no_date_table (
            order_id VARCHAR,
            sales DOUBLE,
            region VARCHAR
        )
    """)
    con.execute("INSERT INTO no_date_table VALUES ('O1', 100.0, 'West')")

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_no_date")
    extractor = ObjectiveExtractor()

    question = "How many total orders were placed in 2017?"
    obj = extractor.extract(profile, question)

    assert obj.missing_columns_explanation is not None
    assert "date" in obj.missing_columns_explanation.lower() or "temporal" in obj.missing_columns_explanation.lower()


# -----------------------------------------------------------------------------
# Test 8: Duplicate line items for orders (deduplication verification)
# -----------------------------------------------------------------------------
def test_duplicate_line_items_deduplication():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE line_items (
            order_id VARCHAR,
            item_id VARCHAR,
            sales DOUBLE
        )
    """)
    # 5 line items across 2 orders
    con.execute("""
        INSERT INTO line_items VALUES
        ('ORD-1', 'ITEM-A', 50.0),
        ('ORD-1', 'ITEM-B', 50.0),
        ('ORD-1', 'ITEM-C', 50.0),
        ('ORD-2', 'ITEM-D', 100.0),
        ('ORD-2', 'ITEM-E', 100.0)
    """)

    res = DualPathVerifier.verify(con, "line_items", metric_col="order_id", agg_func="COUNT_DISTINCT")
    assert res.is_valid
    assert res.audit_details["path_a_sql"] == 2.0
    assert res.audit_details["path_b_pandas"] == 2.0


# -----------------------------------------------------------------------------
# Test 9: Semantic coverage failure when an unauthorized substitution occurs
# -----------------------------------------------------------------------------
def test_semantic_coverage_flags_unauthorized_substitution():
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="COUNT(DISTINCT order_id), AOV",
        target_table="sales",
        required_outputs=["Total unique orders", "Average Order Value"],
    )
    # Plan where agent replaced order count with SUM(sales)
    plan = AnalysisPlan(
        reasoning="Calculating sales revenue",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Calculate total sales revenue",
                metric_col="sales",
                agg_func="SUM",
                completed=True,
                code="SELECT SUM(sales) FROM sales",
            )
        ]
    )
    step_results = {"step_1": {"scalar": 733215.26}}

    question = "How many total orders were placed in 2017, and what is the average order value?"
    report = SemanticCoverageValidator.validate(
        user_question=question,
        objective=objective,
        plan=plan,
        step_results=step_results,
        executed_sqls=["SELECT SUM(sales) FROM sales"],
    )

    assert report.is_covered is False
    assert len(report.missing_metrics) >= 1
    assert len(report.unauthorized_substitutions) >= 1
    assert any("Total unique orders" in s["requested"] for s in report.unauthorized_substitutions)


# -----------------------------------------------------------------------------
# Test 10: ClaimEvidence schema serialization, .to_dict() and safe rendering
# -----------------------------------------------------------------------------
def test_claim_evidence_schema_and_to_dict_resilience():
    # Full fields
    claim = ClaimEvidence(
        claim_id="claim_step_1",
        dataset="superstore",
        claim_text="Total unique orders in 2017 is 1,687",
        metric_name="order_id",
        data_grain="order",
        relevant_filters=["order_date LIKE '%2017%'"],
        analytical_operation="COUNT(DISTINCT order_id)",
        generated_sql="SELECT COUNT(DISTINCT order_id) FROM superstore WHERE order_date LIKE '%2017%'",
        calculated_result=1687,
        supporting_statistics={"hash": "abcd1234ef567890"},
        checksum_hash="abcd1234ef567890",
        validation_status="VALIDATED_DETERMINISTICALLY",
        supporting_evidence=["Dual-path check confirmed"],
        contradictory_evidence=[],
        related_observations="Order deduplication applied",
    )

    d = claim.to_dict()
    assert d["claim_id"] == "claim_step_1"
    assert d["claim_text"] == "Total unique orders in 2017 is 1,687"
    assert d["metric_name"] == "order_id"
    assert d["data_grain"] == "order"
    assert d["checksum_hash"] == "abcd1234ef567890"
    assert d["validation_status"] == "VALIDATED_DETERMINISTICALLY"

    # Default / minimal fields (must not raise AttributeError)
    minimal_claim = ClaimEvidence(
        claim_id="c_min",
        dataset="data",
    )
    d_min = minimal_claim.to_dict()
    assert d_min["claim_id"] == "c_min"
    assert d_min["data_grain"] == "aggregate"
    assert d_min["checksum_hash"] == ""


# -----------------------------------------------------------------------------
# Test 11: ECS confidence gating when semantic coverage fails
# -----------------------------------------------------------------------------
def test_ecs_confidence_gating_on_semantic_failure():
    failed_sem_report = SemanticCoverageReport(
        is_covered=False,
        requested_metrics=["Total unique orders", "Average Order Value"],
        executed_metrics=["sales"],
        missing_metrics=["Total unique orders", "Average Order Value"],
        unauthorized_substitutions=[{"requested": "Total unique orders", "executed": "SUM(revenue)"}],
        details="Unauthorized metric substitution detected.",
    )

    # Even if 1/1 step passed with 100% execution validation:
    ecs = ECSCalculator.compute(
        checks_passed=1,
        checks_total=1,
        profile=None,
        key_columns=["sales"],
        semantic_coverage=failed_sem_report,
    )

    # Score MUST be gated at <= 20.0% and tier forced to LOW
    assert ecs.score <= 20.0
    assert ecs.tier == ConfidenceTier.LOW
    assert "semantic coverage" in ecs.recommendation.lower() or "unauthorized" in ecs.recommendation.lower()
