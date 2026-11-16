"""
Comprehensive regression test suite for Planner:
Objective-aligned planning, step constraints, and deterministic normalization.
"""

import json
from unittest.mock import MagicMock
import pytest

from src.agent.planner import Planner
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisType,
    ColumnProfile,
    DatasetProfile,
    PlanStep,
    TableProfile,
)


@pytest.fixture
def retail_profile():
    return DatasetProfile(
        session_id="test_planner",
        dataset_name="retail_test",
        row_count=1000,
        column_count=6,
        tables=[
            TableProfile(
                table_name="sales",
                row_count=1000,
                column_count=6,
                columns=[
                    ColumnProfile(name="order_id", data_type="VARCHAR", distinct_count=400, null_percentage=0.0),
                    ColumnProfile(
                        name="sales",
                        data_type="DOUBLE",
                        distinct_count=800,
                        null_percentage=0.0,
                        is_numeric=True,
                        min_value=1.0,
                        max_value=5000.0,
                        mean_value=250.0,
                    ),
                    ColumnProfile(
                        name="profit",
                        data_type="DOUBLE",
                        distinct_count=700,
                        null_percentage=0.0,
                        is_numeric=True,
                        min_value=-500.0,
                        max_value=2000.0,
                        mean_value=50.0,
                    ),
                    ColumnProfile(
                        name="region",
                        data_type="VARCHAR",
                        distinct_count=4,
                        null_percentage=0.0,
                        is_categorical=True,
                        sample_values=["East", "West", "Central", "South"],
                    ),
                    ColumnProfile(
                        name="ship_mode",
                        data_type="VARCHAR",
                        distinct_count=3,
                        null_percentage=0.0,
                        is_categorical=True,
                        sample_values=["Standard", "Second", "First"],
                    ),
                    ColumnProfile(
                        name="order_date",
                        data_type="DATE",
                        distinct_count=300,
                        null_percentage=0.0,
                        is_temporal=True,
                    ),
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Total orders + AOV: no dimension breakdowns
# ---------------------------------------------------------------------------
def test_total_orders_and_aov_no_dimension_breakdowns(retail_profile):
    """
    For 'How many total orders were placed in 2017, and what was the average order value?',
    the plan must contain calculations for unique orders and AOV with NO dimension breakdowns.
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric='COUNT(DISTINCT "order_id"), AOV',
        target_metrics=['COUNT(DISTINCT "order_id")', 'AOV'],
        target_table="sales",
        required_outputs=["Total unique orders", "Average Order Value"],
        candidate_dimensions=["ship_mode", "region", "segment", "state"],
        filter_conditions=["CAST(\"order_date\" AS VARCHAR) LIKE '%2017%'"],
        entity_grain="order",
    )

    plan = planner.create_template_plan(retail_profile, objective)

    # Must contain necessary calculations
    funcs = [s.agg_func for s in plan.steps]
    assert "COUNT_DISTINCT" in funcs, "Expected COUNT_DISTINCT for unique orders"
    assert "AOV" in funcs, "Expected AOV aggregation step"

    # No step should have a dimension breakdown
    for step in plan.steps:
        assert step.group_col is None, f"Step {step.step_id} has unexpected dimension breakdown: {step.group_col}"


# ---------------------------------------------------------------------------
# 2. Total revenue: no unrelated dimensions
# ---------------------------------------------------------------------------
def test_total_revenue_no_unrelated_dimensions(retail_profile):
    """
    For a simple scalar question like 'What is total revenue?', the plan must NOT
    contain unrequested breakdowns across candidate dimensions (e.g. ship_mode).
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="sales",
        target_metrics=["sales"],
        target_table="sales",
        required_outputs=["Total revenue"],
        candidate_dimensions=["ship_mode", "region"],
        filter_conditions=[],
    )

    plan = planner.create_template_plan(retail_profile, objective)

    # All steps must be scalar aggregations without grouping
    for step in plan.steps:
        assert step.group_col is None, f"Step {step.step_id} has unexpected grouping {step.group_col}"
        assert step.group_col != "ship_mode"
        assert step.group_col != "region"

    assert len(plan.steps) >= 1
    assert plan.steps[0].agg_func == "SUM"


# ---------------------------------------------------------------------------
# 3. Revenue by region: retain the requested region grouping
# ---------------------------------------------------------------------------
def test_revenue_by_region_retains_requested_grouping(retail_profile):
    """
    When grouping by a dimension (e.g. region) is explicitly requested,
    the plan must retain that grouping while excluding unrequested dimensions.
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="revenue by region",
        target_metrics=["revenue"],
        target_table="sales",
        required_outputs=["Revenue by region"],
        candidate_dimensions=["region", "ship_mode"],
        filter_conditions=[],
    )

    plan = planner.create_template_plan(retail_profile, objective)

    # Must retain region grouping
    grouped_steps = [s for s in plan.steps if s.group_col == "region"]
    assert len(grouped_steps) >= 1, "Expected at least one step grouped by requested dimension 'region'"

    # Must NOT group by unrequested ship_mode
    for step in plan.steps:
        assert step.group_col != "ship_mode"


# ---------------------------------------------------------------------------
# 4. Multiple requested metrics: preserve all necessary calculations
# ---------------------------------------------------------------------------
def test_multiple_requested_metrics_preserves_all_necessary_calculations(retail_profile):
    """
    When multiple metrics are requested (e.g. sales, profit, margin),
    all necessary calculations must be preserved.
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="sales, profit, profit_margin",
        target_metrics=["sales", "profit", "profit_margin"],
        target_table="sales",
        required_outputs=["Total sales", "Total profit", "Profit margin"],
        candidate_dimensions=["region"],
        filter_conditions=[],
    )

    plan = planner.create_template_plan(retail_profile, objective)

    assert len(plan.steps) == 3
    descs = " ".join(s.description.lower() for s in plan.steps)
    assert "sales" in descs
    assert "profit" in descs
    assert "margin" in descs


# ---------------------------------------------------------------------------
# 5. AOV: use SUM(sales) / COUNT(DISTINCT order_id), not AVG(sales)
# ---------------------------------------------------------------------------
def test_aov_uses_sum_div_count_distinct_not_avg(retail_profile):
    """
    AOV must use SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0),
    never AVG(sales), which calculates the line-item average.
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="AOV",
        target_metrics=["AOV"],
        target_table="sales",
        required_outputs=["Average Order Value"],
        candidate_dimensions=["region"],
        filter_conditions=[],
        entity_grain="order",
    )

    plan = planner.create_template_plan(retail_profile, objective)

    for step in plan.steps:
        if "aov" in step.description.lower() or "average order value" in step.description.lower():
            assert step.agg_func == "AOV"
            assert step.agg_func != "AVG"
            assert "sales" in step.metric_col.lower()

    # Verify COUNT_DISTINCT on order_id is present
    has_count_distinct = any(s.agg_func == "COUNT_DISTINCT" and "order_id" in s.metric_col.lower() for s in plan.steps)
    assert has_count_distinct, "Expected COUNT_DISTINCT on order_id"

    # Verify SUM on sales is present
    has_sum_sales = any(s.agg_func == "SUM" and "sales" in s.metric_col.lower() for s in plan.steps)
    assert has_sum_sales, "Expected SUM on sales"


# ---------------------------------------------------------------------------
# 6. LLM returning six steps: normalize to the necessary steps
# ---------------------------------------------------------------------------
def test_llm_returning_six_steps_normalizes_to_necessary_steps(retail_profile):
    """
    When the LLM generates six steps including unrequested breakdowns across
    candidate dimensions, the planner normalizes the plan down to 2-4 necessary steps,
    pruning unrequested exploratory dimensions and renumbering step IDs.
    """
    mock_llm = MagicMock()
    six_step_response = {
        "reasoning": "Calculate orders, sales, AOV, and exploratory breakdowns by ship_mode, segment, and state.",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Calculate total unique orders using COUNT(DISTINCT order_id)",
                "metric_col": "order_id",
                "agg_func": "COUNT_DISTINCT",
                "group_col": None,
                "where_clause": "CAST(order_date AS VARCHAR) LIKE '%2017%'",
            },
            {
                "step_id": "step_2",
                "description": "Calculate total sales volume for sales",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": None,
                "where_clause": "CAST(order_date AS VARCHAR) LIKE '%2017%'",
            },
            {
                "step_id": "step_3",
                "description": "Calculate Average Order Value (AOV) as total sales divided by unique order_id",
                "metric_col": "sales",
                "agg_func": "AOV",
                "group_col": None,
                "where_clause": "CAST(order_date AS VARCHAR) LIKE '%2017%'",
            },
            {
                "step_id": "step_4",
                "description": "Breakdown Average Order Value grouped by ship_mode",
                "metric_col": "sales",
                "agg_func": "AOV",
                "group_col": "ship_mode",
                "where_clause": "CAST(order_date AS VARCHAR) LIKE '%2017%'",
            },
            {
                "step_id": "step_5",
                "description": "Breakdown orders grouped by segment",
                "metric_col": "order_id",
                "agg_func": "COUNT_DISTINCT",
                "group_col": "segment",
                "where_clause": "CAST(order_date AS VARCHAR) LIKE '%2017%'",
            },
            {
                "step_id": "step_6",
                "description": "Breakdown sales volume grouped by state",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": "state",
                "where_clause": "CAST(order_date AS VARCHAR) LIKE '%2017%'",
            },
        ],
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(six_step_response)
    mock_llm.invoke.return_value = mock_response

    planner = Planner(llm=mock_llm)
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric='COUNT(DISTINCT "order_id"), AOV',
        target_metrics=['COUNT(DISTINCT "order_id")', 'AOV'],
        target_table="sales",
        required_outputs=["Total unique orders", "Average Order Value"],
        candidate_dimensions=["ship_mode", "segment", "state"],
        filter_conditions=["CAST(\"order_date\" AS VARCHAR) LIKE '%2017%'"],
        entity_grain="order",
    )

    plan = planner.create_plan(retail_profile, objective)

    # Must be normalized to 2 to 4 steps (specifically 3 necessary steps)
    assert 2 <= len(plan.steps) <= 4, f"Expected 2-4 steps, got {len(plan.steps)}"
    assert len(plan.steps) == 3, f"Expected exactly 3 steps, got {len(plan.steps)}"

    # Step IDs must be sequential
    assert [s.step_id for s in plan.steps] == ["step_1", "step_2", "step_3"]

    # All unrequested dimension breakdowns must be pruned
    for s in plan.steps:
        assert s.group_col is None, f"Step {s.step_id} still has group_col {s.group_col}"

    # Both required outputs must be preserved
    assert any(s.agg_func == "COUNT_DISTINCT" for s in plan.steps)
    assert any(s.agg_func == "AOV" for s in plan.steps)


# ---------------------------------------------------------------------------
# 7. Missing required metric: reject or fall back rather than silently dropping it
# ---------------------------------------------------------------------------
def test_missing_required_metric_falls_back_to_deterministic(retail_profile):
    """
    If the LLM returns steps that omit a required metric (e.g. omits unique orders and AOV),
    the planner detects the coverage gap, rejects the invalid plan, and safely falls back
    to the deterministic template planner.
    """
    mock_llm = MagicMock()
    bad_response = {
        "reasoning": "Compute sales only and omit other metrics",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Calculate total sales",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": None,
                "where_clause": "1=1",
            }
        ],
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(bad_response)
    mock_llm.invoke.return_value = mock_response

    planner = Planner(llm=mock_llm)
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric='COUNT(DISTINCT "order_id"), AOV',
        target_metrics=['COUNT(DISTINCT "order_id")', 'AOV'],
        target_table="sales",
        required_outputs=["Total unique orders", "Average Order Value"],
        candidate_dimensions=["ship_mode"],
        filter_conditions=[],
        entity_grain="order",
    )

    plan = planner.create_plan(retail_profile, objective)

    # Must fall back to deterministic template plan and include both required metrics
    funcs = [s.agg_func for s in plan.steps]
    assert "COUNT_DISTINCT" in funcs, "Fallback plan must cover unique orders"
    assert "AOV" in funcs, "Fallback plan must cover AOV"
    assert len(plan.steps) >= 2


# ---------------------------------------------------------------------------
# 8. Issue 1: Ranking questions authorize only requested entity, not all candidate dimensions
# ---------------------------------------------------------------------------
def test_ranking_authorizes_only_requested_entity_and_rejects_unrelated_dimensions(retail_profile):
    """
    For a ranking question such as 'Which product generated the highest profit?',
    only the explicitly requested ranking entity ('product_name') should be authorized.
    Unrelated candidate dimensions ('region', 'ship_mode', 'customer_name') must NOT be authorized.
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="highest profit by product",
        target_metrics=["profit"],
        target_table="sales",
        required_outputs=["Highest profit product"],
        candidate_dimensions=["product_name", "region", "ship_mode", "customer_name"],
        filter_conditions=[],
        entity_grain="product",
    )

    # 1. Direct authorization check via _is_grouping_requested
    assert planner._is_grouping_requested(objective, "product_name") is True
    assert planner._is_grouping_requested(objective, "region") is False
    assert planner._is_grouping_requested(objective, "ship_mode") is False
    assert planner._is_grouping_requested(objective, "customer_name") is False

    # 2. Template planner check: must group by product_name and not unrelated dimensions
    plan = planner.create_template_plan(retail_profile, objective)
    grouped_steps = [s for s in plan.steps if s.group_col is not None]
    assert len(grouped_steps) >= 1
    for s in grouped_steps:
        assert s.group_col == "product_name"
        assert s.group_col != "region"
        assert s.group_col != "ship_mode"

    # 3. LLM plan normalization check: prune steps grouping by unauthorized dimensions
    mock_llm = MagicMock()
    ranking_response = {
        "reasoning": "Identify highest profit product and exploratory breakdowns",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Calculate profit grouped by product_name to identify the highest product",
                "metric_col": "profit",
                "agg_func": "SUM",
                "group_col": "product_name",
                "where_clause": "1=1",
            },
            {
                "step_id": "step_2",
                "description": "Compute overall aggregate profit for verification",
                "metric_col": "profit",
                "agg_func": "SUM",
                "group_col": None,
                "where_clause": "1=1",
            },
            {
                "step_id": "step_3",
                "description": "Breakdown profit grouped by region",
                "metric_col": "profit",
                "agg_func": "SUM",
                "group_col": "region",
                "where_clause": "1=1",
            },
            {
                "step_id": "step_4",
                "description": "Breakdown profit grouped by ship_mode",
                "metric_col": "profit",
                "agg_func": "SUM",
                "group_col": "ship_mode",
                "where_clause": "1=1",
            },
        ],
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(ranking_response)
    mock_llm.invoke.return_value = mock_response

    llm_planner = Planner(llm=mock_llm)
    norm_plan = llm_planner.create_plan(retail_profile, objective)

    # Must keep product_name and overall aggregate, but prune region and ship_mode
    step_groups = [s.group_col for s in norm_plan.steps if s.group_col is not None]
    assert "product_name" in step_groups
    assert "region" not in step_groups
    assert "ship_mode" not in step_groups


# ---------------------------------------------------------------------------
# 9. Issue 2: Diagnostic mode authorizes legitimate RCA dimensions without blanket authorization
# ---------------------------------------------------------------------------
def test_diagnostic_authorizes_rca_dimensions_without_authorizing_all_schema_dimensions(retail_profile):
    """
    In DIAGNOSTIC mode, legitimate RCA exploratory dimensions (primary/secondary or filtered)
    are permitted, but candidate dimensions are not automatically blanket-authorized.
    """
    planner = Planner()
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DIAGNOSTIC,
        target_metric="revenue",
        target_metrics=["revenue"],
        target_table="sales",
        required_outputs=["Decompose revenue drop"],
        candidate_dimensions=["region", "category", "ship_mode", "postal_code", "segment"],
        filter_conditions=["region = 'West'"],
    )

    # region is in candidate_dimensions[:2] AND in filter_conditions -> authorized
    assert planner._is_grouping_requested(objective, "region") is True
    # category is in candidate_dimensions[:2] -> authorized
    assert planner._is_grouping_requested(objective, "category") is True
    # ship_mode, postal_code, segment are beyond candidate_dimensions[:2] and not in filters -> NOT authorized
    assert planner._is_grouping_requested(objective, "ship_mode") is False
    assert planner._is_grouping_requested(objective, "postal_code") is False
    assert planner._is_grouping_requested(objective, "segment") is False

    # LLM plan normalization should retain authorized diagnostic dimensions and prune unauthorized ones
    mock_llm = MagicMock()
    diagnostic_response = {
        "reasoning": "Diagnostic decomposition of revenue",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Calculate overall total revenue",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": None,
                "where_clause": "1=1",
            },
            {
                "step_id": "step_2",
                "description": "Decompose revenue across region",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": "region",
                "where_clause": "1=1",
            },
            {
                "step_id": "step_3",
                "description": "Decompose revenue across category",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": "category",
                "where_clause": "1=1",
            },
            {
                "step_id": "step_4",
                "description": "Decompose revenue across postal_code",
                "metric_col": "sales",
                "agg_func": "SUM",
                "group_col": "postal_code",
                "where_clause": "1=1",
            },
        ],
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(diagnostic_response)
    mock_llm.invoke.return_value = mock_response

    llm_planner = Planner(llm=mock_llm)
    norm_plan = llm_planner.create_plan(retail_profile, objective)

    groups = [s.group_col for s in norm_plan.steps if s.group_col is not None]
    assert "region" in groups
    assert "category" in groups
    assert "postal_code" not in groups, "Unauthorized diagnostic dimension postal_code must be pruned"


# ---------------------------------------------------------------------------
# 10. Truncation logic: Never silently discard essential calculations
# ---------------------------------------------------------------------------
def test_truncation_logic_never_discards_essential_calculations(retail_profile):
    """
    When an LLM generates more than 4 steps, essential calculations must never be
    silently discarded. If essential calculations alone exceed 4 steps, it must
    fall back safely to the deterministic template planner.
    """
    mock_llm = MagicMock()
    # 5 essential steps for 5 requested metrics
    five_metric_response = {
        "reasoning": "Compute 5 requested metrics",
        "steps": [
            {"step_id": "step_1", "description": "Calculate total sales", "metric_col": "sales", "agg_func": "SUM", "group_col": None, "where_clause": "1=1"},
            {"step_id": "step_2", "description": "Calculate total profit", "metric_col": "profit", "agg_func": "SUM", "group_col": None, "where_clause": "1=1"},
            {"step_id": "step_3", "description": "Calculate profit margin", "metric_col": "profit", "agg_func": "AVG", "group_col": None, "where_clause": "1=1"},
            {"step_id": "step_4", "description": "Calculate total unique orders using COUNT(DISTINCT order_id)", "metric_col": "order_id", "agg_func": "COUNT_DISTINCT", "group_col": None, "where_clause": "1=1"},
            {"step_id": "step_5", "description": "Calculate Average Order Value (AOV)", "metric_col": "sales", "agg_func": "AOV", "group_col": None, "where_clause": "1=1"},
        ],
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(five_metric_response)
    mock_llm.invoke.return_value = mock_response

    planner = Planner(llm=mock_llm)
    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="sales, profit, margin, orders, aov",
        target_metrics=["sales", "profit", "profit_margin", "order", "aov"],
        target_table="sales",
        required_outputs=["Total sales", "Total profit", "Profit margin", "Total unique orders", "Average Order Value"],
        candidate_dimensions=["region"],
        filter_conditions=[],
        entity_grain="order",
    )

    plan = planner.create_plan(retail_profile, objective)

    # Must fall back to template plan rather than silently truncating/dropping the 5th metric
    assert plan is not None
    assert len(plan.steps) >= 2
    # Verify no essential metric was silently dropped from the plan
    funcs = [s.agg_func for s in plan.steps]
    assert "SUM" in funcs
    assert "COUNT_DISTINCT" in funcs
    assert "AOV" in funcs

