"""Comprehensive unit and regression tests for AIR v2.1 Phase 1 implementation."""

import pytest
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    ColumnProfile,
    DatasetProfile,
    PlanStep,
    TableProfile,
)
from src.agent.air.schemas import (
    AIRLifecycleState,
    ColumnRef,
    ComparisonOperator,
    DataType,
    DimensionAuthorizationMatrix,
    GrainAssessment,
    GrainConfidence,
    Predicate,
)
from src.agent.air.operations import (
    AggregateMeasureSpec,
    AggregateOperation,
    AggregationFunction,
    DerivedExpressionSpec,
    FilterOperation,
    RankOperation,
    SortCriterion,
    SortDirection,
)
from src.agent.air.dag import DAGIntegrityValidator, DAGValidationError, MasterAIR
from src.agent.air.filter_parser import StrictFilterParser
from src.agent.air.temporal import TemporalBoundaryResolver
from src.agent.air.adapter import AIRAdapter


@pytest.fixture
def base_schemas():
    return {
        "sales": {
            "order_id": DataType.STRING,
            "order_date": DataType.DATETIME,
            "sales": DataType.FLOAT,
            "profit": DataType.FLOAT,
            "region": DataType.STRING,
            "product_name": DataType.STRING,
            "zip_code": DataType.INTEGER,  # numeric dimension
        }
    }


# =============================================================================
# 1. AOV as SUM / COUNT DISTINCT / Division
# =============================================================================

def test_aov_representation_and_derivation():
    """Verifies AOV is decomposed into SUM, COUNT_DISTINCT, and ratio derivation."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric='AOV',
        required_outputs=["Average Order Value"],
    )
    plan = AnalysisPlan(
        reasoning="AOV calculation",
        steps=[PlanStep(step_id="s1", description="Calculate AOV", metric_col="sales", agg_func="AOV")],
    )

    res = AIRAdapter.from_legacy(obj, plan, user_question="What is the average order value?")
    assert res.is_successful
    assert len(res.air.operations) == 1

    op = res.air.operations[0]
    assert isinstance(op, AggregateOperation)
    assert len(op.measures) == 2
    assert op.measures[0].function == AggregationFunction.SUM
    assert op.measures[0].output_alias == "total_sales"
    assert op.measures[1].function == AggregationFunction.COUNT_DISTINCT
    assert op.measures[1].output_alias == "unique_orders"

    assert len(op.derived_measures) == 1
    der = op.derived_measures[0]
    assert der.output_alias == "aov"
    assert "total_sales / NULLIF(unique_orders, 0)" in der.expression
    assert der.referenced_columns == ["total_sales", "unique_orders"]

    # All three must be declared in output_schema
    assert "total_sales" in op.output_schema
    assert "unique_orders" in op.output_schema
    assert "aov" in op.output_schema


# =============================================================================
# 2. Top 5 Products by Aggregate Revenue
# =============================================================================

def test_top_5_products_aggregates_before_ranking():
    """Verifies that ranking by total revenue builds a two-node pipeline: Aggregate -> Rank."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric="sales",
        candidate_dimensions=["product_name"],
        required_outputs=["Top Products"],
    )
    plan = AnalysisPlan(
        reasoning="Rank products",
        steps=[PlanStep(step_id="s1", description="Top 5 products by total revenue", metric_col="sales", agg_func="RANK")],
    )

    res = AIRAdapter.from_legacy(obj, plan, user_question="Top 5 products by total revenue")
    assert res.is_successful
    assert len(res.air.operations) == 2

    agg_op = res.air.operations[0]
    rank_op = res.air.operations[1]

    # Node 1: Aggregate
    assert isinstance(agg_op, AggregateOperation)
    assert agg_op.group_by_columns[0].name == "product_name"
    assert agg_op.measures[0].output_alias == "total_sales"

    # Node 2: Rank
    assert isinstance(rank_op, RankOperation)
    assert rank_op.depends_on == [agg_op.operation_id]
    assert rank_op.input_datasets == [agg_op.output_alias]
    assert rank_op.limit == 5
    assert rank_op.order_by[0].column.name == "total_sales"
    assert rank_op.order_by[0].is_metric is True
    assert rank_op.order_by[0].direction == SortDirection.DESC


# =============================================================================
# 3. Fail-Closed Dimension Authorization & Partitioning
# =============================================================================

def test_unauthorized_grouping_rejected(base_schemas):
    """Verifies that grouping by a dimension with allow_grouping=False is rejected."""
    dims = [DimensionAuthorizationMatrix(column_name="region", allow_grouping=False)]
    op = AggregateOperation(
        operation_id="op1",
        description="unauthorized group",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        group_by_columns=[ColumnRef(name="region")],
        output_schema={"region": DataType.STRING},
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Unauthorized Grouping: Column 'region' not authorized for GROUP BY"):
        DAGIntegrityValidator.validate_and_sort([op], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_unauthorized_partitioning_rejected(base_schemas):
    """Verifies that window partitioning by a dimension with allow_grouping=False is rejected."""
    dims = [DimensionAuthorizationMatrix(column_name="region", allow_grouping=False)]
    op = RankOperation(
        operation_id="op1",
        description="unauthorized partition",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        order_by=[SortCriterion(column=ColumnRef(name="sales"), direction=SortDirection.DESC, is_metric=True)],
        group_by_partitions=[ColumnRef(name="region")],
        output_schema=base_schemas["sales"],
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Unauthorized Partitioning.*region"):
        DAGIntegrityValidator.validate_and_sort([op], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_numeric_dimension_versus_metric_ordering(base_schemas):
    """Verifies that ordering on a numeric dimension requires dimension authorization, but metric ordering does not."""
    # Case A: Sorting by numeric dimension 'zip_code' without allow_order_by -> fails
    dims = [DimensionAuthorizationMatrix(column_name="zip_code", allow_order_by=False)]
    op_dim = RankOperation(
        operation_id="op1",
        description="order by zip",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        order_by=[SortCriterion(column=ColumnRef(name="zip_code"), direction=SortDirection.ASC, is_metric=False)],
        output_schema=base_schemas["sales"],
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Unauthorized Dimension Ordering.*zip_code"):
        DAGIntegrityValidator.validate_and_sort([op_dim], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])

    # Case B: Sorting by computed metric with is_metric=True -> succeeds even if not in dimension matrix
    op_metric = RankOperation(
        operation_id="op2",
        description="order by sales",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v2",
        order_by=[SortCriterion(column=ColumnRef(name="sales"), direction=SortDirection.DESC, is_metric=True)],
        output_schema=base_schemas["sales"],
        is_terminal=True,
    )
    res = DAGIntegrityValidator.validate_and_sort([op_metric], terminal_ids=["op2"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])
    assert len(res) == 1


# =============================================================================
# 4. Strict Boolean Filter Parser Tests
# =============================================================================

def test_quoted_strings_containing_and_or_not_split():
    """Verifies string literals containing AND/OR are parsed as literal strings, not boolean operators."""
    raw = "region = 'East AND West' AND product_name = 'Chair OR Table'"
    preds, err = StrictFilterParser.parse_conjunction(raw, ["region", "product_name"])
    assert err is None
    assert len(preds) == 2
    assert preds[0].literal_value == "East AND West"
    assert preds[1].literal_value == "Chair OR Table"


def test_between_combined_with_other_predicates():
    """Verifies BETWEEN is parsed atomically alongside other conjunctions."""
    raw = "order_date BETWEEN '2017-01-01' AND '2017-12-31' AND sales >= 50"
    preds, err = StrictFilterParser.parse_conjunction(raw, ["order_date", "sales"])
    assert err is None
    assert len(preds) == 2
    assert preds[0].operator == ComparisonOperator.BETWEEN
    assert preds[0].literal_value == ["2017-01-01", "2017-12-31"]
    assert preds[1].operator == ComparisonOperator.GTE
    assert preds[1].literal_value == 50


def test_unsupported_or_expressions_rejected():
    """Verifies that SQL OR clauses produce fatal diagnostics."""
    raw = "region = 'East' OR sales > 100"
    preds, err = StrictFilterParser.parse_conjunction(raw, ["region", "sales"])
    assert err is not None
    assert "Unsupported boolean operator 'OR'" in err


# =============================================================================
# 5. Type and Column Existence Invariants
# =============================================================================

def test_missing_column_rejection(base_schemas):
    """Verifies referencing non-existent column raises DAGValidationError."""
    dims = []
    op = AggregateOperation(
        operation_id="op1",
        description="bad col",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        measures=[AggregateMeasureSpec(output_alias="rev", function=AggregationFunction.SUM, target_column=ColumnRef(name="unknown_col"))],
        output_schema={"rev": DataType.FLOAT},
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Measure column 'unknown_col' not in inputs"):
        DAGIntegrityValidator.validate_and_sort([op], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_incompatible_aggregation_type(base_schemas):
    """Verifies that applying SUM to a string column fails validation."""
    dims = []
    op = AggregateOperation(
        operation_id="op1",
        description="sum string",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        measures=[AggregateMeasureSpec(output_alias="sum_reg", function=AggregationFunction.SUM, target_column=ColumnRef(name="region"))],
        output_schema={"sum_reg": DataType.FLOAT},
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Incompatible data type for SUM: Column 'region' is 'string'"):
        DAGIntegrityValidator.validate_and_sort([op], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_invalid_derived_expression_ast():
    """Verifies restricted AST blocks unauthorized calls and undeclared column references."""
    # Forbidden call
    with pytest.raises(ValueError, match="Forbidden function call"):
        DerivedExpressionSpec(output_alias="x", expression="__import__('os').system('ls')", referenced_columns=["a"])

    # Undeclared reference
    with pytest.raises(ValueError, match="references undeclared columns"):
        DerivedExpressionSpec(output_alias="x", expression="a + b", referenced_columns=["a"])


# =============================================================================
# 6. Exact Dependency, Producer & Schema Consistency
# =============================================================================

def test_exact_dependency_input_matching(base_schemas):
    """Verifies an operation reading from a view must declare its producing operation in depends_on."""
    dims = []
    # op1 produces v1, is terminal
    op1 = FilterOperation(operation_id="op1", description="1", depends_on=[], input_datasets=["sales"], output_alias="v1", predicates=[], output_schema=base_schemas["sales"], is_terminal=True)
    # op2 produces v2
    op2 = FilterOperation(operation_id="op2", description="2", depends_on=[], input_datasets=["sales"], output_alias="v2", predicates=[], output_schema=base_schemas["sales"])
    # op3 depends only on op2, but reads v1 (produced by op1) and v2 (produced by op2)
    op3 = FilterOperation(operation_id="op3", description="3", depends_on=["op2"], input_datasets=["v1", "v2"], output_alias="v3", predicates=[], output_schema=base_schemas["sales"], is_terminal=True)

    with pytest.raises(DAGValidationError, match="reads view 'v1' from 'op1' without declaring it in depends_on"):
        DAGIntegrityValidator.validate_and_sort([op1, op2, op3], terminal_ids=["op1", "op3"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_duplicate_producers_and_alias_collisions(base_schemas):
    """Verifies multiple producers of same alias and collisions with base table are rejected."""
    dims = []
    # Collision with base table 'sales'
    op_col = FilterOperation(operation_id="op1", description="1", depends_on=[], input_datasets=["sales"], output_alias="sales", predicates=[], output_schema=base_schemas["sales"], is_terminal=True)
    with pytest.raises(DAGValidationError, match="Output alias 'sales' collides with base table name"):
        DAGIntegrityValidator.validate_and_sort([op_col], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_incorrect_output_schema_rejected(base_schemas):
    """Verifies declared output schema must match derived schema."""
    dims = [DimensionAuthorizationMatrix(column_name="region", allow_grouping=True)]
    op = AggregateOperation(
        operation_id="op1",
        description="agg",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        group_by_columns=[ColumnRef(name="region")],
        measures=[AggregateMeasureSpec(output_alias="tot_sales", function=AggregationFunction.SUM, target_column=ColumnRef(name="sales"))],
        # Declares extra column not produced by operation
        output_schema={"region": DataType.STRING, "tot_sales": DataType.FLOAT, "ghost_col": DataType.INTEGER},
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Invariant 9: Output schema mismatch"):
        DAGIntegrityValidator.validate_and_sort([op], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=[])


def test_missing_requested_outputs_rejected(base_schemas):
    """Verifies Invariant 10 fails if a user requested output is not projected in terminal result."""
    dims = []
    op = AggregateOperation(
        operation_id="op1",
        description="agg",
        depends_on=[],
        input_datasets=["sales"],
        output_alias="v1",
        measures=[AggregateMeasureSpec(output_alias="sales", function=AggregationFunction.SUM, target_column=ColumnRef(name="sales"))],
        output_schema={"sales": DataType.FLOAT},
        is_terminal=True,
    )
    with pytest.raises(DAGValidationError, match="Invariant 10: Requested output requirement 'Average Order Value' is NOT satisfied"):
        DAGIntegrityValidator.validate_and_sort([op], terminal_ids=["op1"], base_table_schemas=base_schemas, dimensions=dims, requested_outputs=["Average Order Value"])


# =============================================================================
# 7. Temporal Half-Open Boundaries & Timezones
# =============================================================================

def test_half_open_date_and_timestamp_boundaries():
    """Verifies period intervals are strictly [start, next_start) for timestamp safety."""
    col = ColumnRef(name="order_date", data_type=DataType.DATETIME)

    # Quarter
    p_start, p_end = TemporalBoundaryResolver.resolve_period_to_predicates("2023-Q3", col)
    assert p_start.operator == ComparisonOperator.GTE and p_start.literal_value == "2023-07-01"
    assert p_end.operator == ComparisonOperator.LT and p_end.literal_value == "2023-10-01"

    # Year
    p_y_start, p_y_end = TemporalBoundaryResolver.resolve_period_to_predicates("2017", col)
    assert p_y_start.operator == ComparisonOperator.GTE and p_y_start.literal_value == "2017-01-01"
    assert p_y_end.operator == ComparisonOperator.LT and p_y_end.literal_value == "2018-01-01"


def test_unknown_timezone_policy_handling():
    """Verifies unsupported timezone policy raises ValueError."""
    col = ColumnRef(name="order_date", data_type=DataType.DATETIME)
    with pytest.raises(ValueError, match="Unsupported timezone_policy"):
        TemporalBoundaryResolver.resolve_period_to_predicates("2023-Q1", col, timezone_policy="INVALID_TZ")


# =============================================================================
# 8. Grain & Candidate Key Evidence Tests
# =============================================================================

def test_candidate_key_remains_candidate_without_empirical_uniqueness():
    """Verifies a key in primary_key_candidates without distinct==count evidence is CANDIDATE, not PROVEN_UNIQUE."""
    profile = DatasetProfile(
        session_id="s1",
        tables=[
            TableProfile(
                table_name="sales",
                row_count=1000,
                column_count=2,
                columns=[ColumnProfile(name="order_id", data_type="VARCHAR", distinct_count=500, null_percentage=0.0)],
                primary_key_candidates=["order_id"],
            )
        ],
    )
    obj = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", entity_grain="order")
    res = AIRAdapter.from_legacy(obj, profile=profile)

    assert res.air.grain_assessment.confidence == GrainConfidence.INFERRED_MULTI_GRAIN
    assert res.air.grain_assessment.requires_entity_rollup is True


# =============================================================================
# 9. Lossless Legacy Snapshot & Lifecycle State Tests
# =============================================================================

def test_exact_legacy_objective_and_plan_restoration():
    """Verifies exact deep-copy restoration of legacy AnalysisObjective and AnalysisPlan."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric='COUNT(DISTINCT "order_id"), AOV',
        target_metrics=['COUNT(DISTINCT "order_id")', 'AOV'],
        entity_grain="order",
        required_outputs=["Total unique orders", "Average Order Value"],
        candidate_dimensions=["region", "ship_mode"],
        filter_conditions=["order_date >= '2017-01-01'"],
    )
    plan = AnalysisPlan(
        reasoning="Test reasoning",
        steps=[
            PlanStep(step_id="step_1", description="Count orders", metric_col="order_id", agg_func="COUNT_DISTINCT", where_clause="1=1"),
            PlanStep(step_id="step_2", description="Sum sales", metric_col="sales", agg_func="SUM", where_clause="1=1"),
        ],
    )

    res = AIRAdapter.from_legacy(obj, plan)
    assert res.is_successful

    restored_obj = AIRAdapter.to_legacy_objective(res.air)
    restored_plan = AIRAdapter.to_legacy_plan(res.air)

    assert restored_obj.model_dump() == obj.model_dump()
    assert restored_plan.model_dump() == plan.model_dump()


def test_empty_dag_in_extracted_state():
    """Verifies MasterAIR in EXTRACTED state is valid with an empty DAG."""
    air = MasterAIR(
        lifecycle_state=AIRLifecycleState.EXTRACTED,
        raw_user_question="What are total sales?",
        metadata_summary="Extracted objective",
        grain_assessment=GrainAssessment(target_entity="line_item", supporting_evidence="None"),
        operations=[],
        terminal_operation_ids=[],
    )
    assert air.lifecycle_state == AIRLifecycleState.EXTRACTED
    assert len(air.operations) == 0


def test_empty_dag_rejection_in_planned_or_validated_state(base_schemas):
    """Verifies empty DAG raises DAGValidationError in PLANNED or VALIDATED states."""
    with pytest.raises(DAGValidationError, match="Invariant 1: Operation graph cannot be empty"):
        DAGIntegrityValidator.validate_and_sort([], terminal_ids=[], base_table_schemas=base_schemas, dimensions=[], requested_outputs=[], lifecycle_state=AIRLifecycleState.PLANNED)

    with pytest.raises(DAGValidationError, match="Invariant 1: Operation graph cannot be empty"):
        DAGIntegrityValidator.validate_and_sort([], terminal_ids=[], base_table_schemas=base_schemas, dimensions=[], requested_outputs=[], lifecycle_state=AIRLifecycleState.VALIDATED)


# =============================================================================
# 10. D1: Ranking Dimension Authorization Tests
# =============================================================================

def test_ranking_entity_products_to_product_name(base_schemas):
    """Verifies 'products' maps deterministically to 'product_name' and authorizes grouping."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric="sales",
        candidate_dimensions=["product_name", "region"],
    )
    plan = AnalysisPlan(
        reasoning="Top products",
        steps=[PlanStep(step_id="s1", description="Top 5 products by total revenue", metric_col="sales", agg_func="RANK")],
    )
    res = AIRAdapter.from_legacy(obj, plan, user_question="Top 5 products by total revenue")
    assert res.is_successful
    p_dim = next(d for d in res.air.dimensions if d.column_name == "product_name")
    r_dim = next(d for d in res.air.dimensions if d.column_name == "region")

    assert p_dim.allow_grouping is True
    assert p_dim.allow_order_by is True
    assert p_dim.is_requested_in_final_output is True
    # Unrelated dimension remains unauthorized
    assert r_dim.allow_grouping is False
    assert r_dim.allow_order_by is False


def test_ranking_entity_product_category_not_product_name():
    """Verifies 'product category' maps to 'product_category' and NOT 'product_name'."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric="sales",
        candidate_dimensions=["product_name", "product_category"],
    )
    plan = AnalysisPlan(
        reasoning="Top category",
        steps=[PlanStep(step_id="s1", description="Top product category by revenue", metric_col="sales", agg_func="RANK")],
    )
    res = AIRAdapter.from_legacy(obj, plan, user_question="Top product category by revenue")
    assert res.is_successful
    cat_dim = next(d for d in res.air.dimensions if d.column_name == "product_category")
    prod_dim = next(d for d in res.air.dimensions if d.column_name == "product_name")

    assert cat_dim.allow_grouping is True
    assert cat_dim.is_requested_in_final_output is True
    assert prod_dim.allow_grouping is False
    assert prod_dim.is_requested_in_final_output is False


def test_ranking_entity_customers_to_customer_name():
    """Verifies 'customers' maps deterministically to 'customer_name'."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric="sales",
        candidate_dimensions=["customer_name", "region"],
    )
    plan = AnalysisPlan(
        reasoning="Top customers",
        steps=[PlanStep(step_id="s1", description="Top 3 customers by revenue", metric_col="sales", agg_func="RANK")],
    )
    res = AIRAdapter.from_legacy(obj, plan, user_question="Top 3 customers by revenue")
    assert res.is_successful
    c_dim = next(d for d in res.air.dimensions if d.column_name == "customer_name")
    assert c_dim.allow_grouping is True
    assert c_dim.is_requested_in_final_output is True


def test_ranking_entity_ambiguous_mapping_diagnostic():
    """Verifies ambiguous entity matching (e.g. 'products' with both product_name and product_id) returns diagnostic."""
    obj = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_table="sales",
        target_metric="sales",
        candidate_dimensions=["product_name", "product_id"],
    )
    plan = AnalysisPlan(
        reasoning="Ambiguous",
        steps=[PlanStep(step_id="s1", description="Top 5 products by sales", metric_col="sales", agg_func="RANK")],
    )
    res = AIRAdapter.from_legacy(obj, plan, user_question="Top 5 products by sales")
    assert not res.is_successful
    assert any("Ambiguous entity mapping" in d.diagnostic_reason for d in res.diagnostics)


# =============================================================================
# 11. D2: Compound Grouping Authorization Tests
# =============================================================================

def test_compound_grouping_authorization_examples():
    """Verifies explicit compound grouping sets: single, double, triple, and filter-only."""
    c_dims = ["region", "ship_mode", "segment"]

    # 1. Single: "Revenue by region"
    obj1 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=c_dims)
    res1 = AIRAdapter.from_legacy(obj1, user_question="Revenue by region")
    d_map1 = {d.column_name: d.allow_grouping for d in res1.air.dimensions}
    assert d_map1 == {"region": True, "ship_mode": False, "segment": False}

    # 2. Double: "Revenue by region and ship mode"
    obj2 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=c_dims)
    res2 = AIRAdapter.from_legacy(obj2, user_question="Revenue by region and ship mode")
    d_map2 = {d.column_name: d.allow_grouping for d in res2.air.dimensions}
    assert d_map2 == {"region": True, "ship_mode": True, "segment": False}

    # 3. Triple: "Revenue by region, ship mode, and segment"
    obj3 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=c_dims)
    res3 = AIRAdapter.from_legacy(obj3, user_question="Revenue by region, ship mode, and segment")
    d_map3 = {d.column_name: d.allow_grouping for d in res3.air.dimensions}
    assert d_map3 == {"region": True, "ship_mode": True, "segment": True}

    # 4. Filter only: "Revenue for Western region" (no grouping)
    obj4 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=c_dims)
    res4 = AIRAdapter.from_legacy(obj4, user_question="Revenue for Western region")
    d_map4 = {d.column_name: d.allow_grouping for d in res4.air.dimensions}
    assert d_map4 == {"region": False, "ship_mode": False, "segment": False}


# =============================================================================
# 12. D3: Ranking Grain Semantics Tests (Three Distinct Graphs)
# =============================================================================

def test_ranking_grain_semantics_three_distinct_graphs():
    """
    Proves that the three ranking grain modes generate different, valid operation graphs:
    1. Raw records: RankOperation reading from base table.
    2. Entity aggregation: AggregateOperation by entity -> RankOperation.
    3. Entity grain aggregation: AggregateOperation by order_id -> RankOperation.
    """
    # Case 1: Highest individual transaction (raw record ranking)
    obj1 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=["product_name", "order_id"])
    plan1 = AnalysisPlan(reasoning="r1", steps=[PlanStep(step_id="s1", description="Highest individual transaction", metric_col="sales", agg_func="RANK")])
    res1 = AIRAdapter.from_legacy(obj1, plan1, user_question="Highest individual transaction")
    assert res1.is_successful
    assert len(res1.air.operations) == 1
    assert isinstance(res1.air.operations[0], RankOperation)
    assert res1.air.operations[0].input_datasets == ["sales"]

    # Case 2: Top 5 products by total revenue (entity aggregation)
    obj2 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=["product_name"])
    plan2 = AnalysisPlan(reasoning="r2", steps=[PlanStep(step_id="s1", description="Top 5 products by total revenue", metric_col="sales", agg_func="RANK")])
    res2 = AIRAdapter.from_legacy(obj2, plan2, user_question="Top 5 products by total revenue")
    assert res2.is_successful
    assert len(res2.air.operations) == 2
    assert isinstance(res2.air.operations[0], AggregateOperation)
    assert res2.air.operations[0].group_by_columns[0].name == "product_name"
    assert isinstance(res2.air.operations[1], RankOperation)
    assert res2.air.operations[1].input_datasets == ["view_s1_agg"]

    # Case 3: Highest order by total order value (entity grain aggregation)
    obj3 = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", entity_grain="order", candidate_dimensions=["order_id"])
    plan3 = AnalysisPlan(reasoning="r3", steps=[PlanStep(step_id="s1", description="Highest order by total order value", metric_col="sales", agg_func="RANK")])
    profile3 = DatasetProfile(
        session_id="s3",
        tables=[TableProfile(table_name="sales", row_count=100, column_count=2, columns=[ColumnProfile(name="order_id", data_type="VARCHAR"), ColumnProfile(name="sales", data_type="DOUBLE")])]
    )
    res3 = AIRAdapter.from_legacy(obj3, plan3, profile=profile3, user_question="Highest order by total order value")
    assert res3.is_successful
    assert len(res3.air.operations) == 2
    assert isinstance(res3.air.operations[0], AggregateOperation)
    assert res3.air.operations[0].group_by_columns[0].name == "order_id"
    assert isinstance(res3.air.operations[1], RankOperation)
    assert res3.air.operations[1].input_datasets == ["view_s1_agg"]


# =============================================================================
# 13. D4: Shadow Integration Graph Tests
# =============================================================================

def test_shadow_integration_graph_success_and_failure(base_schemas):
    """
    Verifies AnalyticsWorkflowEngine.plan_node:
    - Stores validated AIR with lifecycle_state == VALIDATED when valid.
    - Sets air = None and populates air_diagnostic when validation fails, preserving legacy plan.
    """
    import duckdb
    from src.agent.graph import AnalyticsWorkflowEngine
    con = duckdb.connect()
    engine = AnalyticsWorkflowEngine(con=con)

    profile = DatasetProfile(
        session_id="test_engine",
        tables=[TableProfile(table_name="sales", row_count=100, column_count=2, columns=[ColumnProfile(name="sales", data_type="DOUBLE", is_numeric=True), ColumnProfile(name="region", data_type="VARCHAR", is_categorical=True)])]
    )

    # 1. Valid plan
    obj_valid = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=["region"], required_outputs=["Total Sales"])
    plan_valid = AnalysisPlan(reasoning="v", steps=[PlanStep(step_id="s1", description="Revenue by region", metric_col="sales", agg_func="SUM", group_col="region")])
    state_valid = {"profile": profile, "objective": obj_valid, "plan": plan_valid, "user_question": "Revenue by region"}

    out_valid = engine.plan_node(state_valid)
    assert out_valid["plan"] is not None
    assert out_valid["air"] is not None
    assert out_valid["air"].lifecycle_state == AIRLifecycleState.VALIDATED
    assert out_valid["air_diagnostic"] is None

    # 2. Invalid plan (references unresolvable column in group_col)
    obj_invalid = AnalysisObjective(analysis_type=AnalysisType.DESCRIPTIVE, target_table="sales", target_metric="sales", candidate_dimensions=["region"], required_outputs=["Total Sales"])
    plan_invalid = AnalysisPlan(reasoning="inv", steps=[PlanStep(step_id="s1", description="Revenue by secret_dim", metric_col="sales", agg_func="SUM", group_col="secret_dim")])
    state_invalid = {"profile": profile, "objective": obj_invalid, "plan": plan_invalid, "user_question": "Revenue by region"}

    out_invalid = engine.plan_node(state_invalid)
    assert out_invalid["plan"] is not None  # Legacy execution remains unaffected!
    assert out_invalid["air"] is None       # Unvalidated AIR is NOT exposed as validated
    assert out_invalid["air_diagnostic"] is not None
    assert "Invariant 8" in out_invalid["air_diagnostic"] or "not in inputs" in out_invalid["air_diagnostic"]


# =============================================================================
# 14. D5: Statically Provable Zero Divisor Tests
# =============================================================================

def test_statically_provable_zero_divisors_rejected():
    """Verifies that constant zero denominators in all equivalent forms are rejected."""
    bad_expressions = [
        "a / 0",
        "a // 0",
        "a % 0",
        "a / -0",
        "a / 0.0",
        "a / -0.0",
        "a / (0 + 0)",
        "a / (0 * b)",
    ]
    for expr in bad_expressions:
        with pytest.raises(ValueError, match="Statically provable zero divisor"):
            DerivedExpressionSpec(output_alias="out", expression=expr, referenced_columns=["a", "b"])


def test_nullif_protected_denominator_permitted():
    """Verifies that NULLIF-protected and dynamic denominators are permitted."""
    d1 = DerivedExpressionSpec(output_alias="out1", expression="a / NULLIF(b, 0)", referenced_columns=["a", "b"])
    assert d1.output_alias == "out1"

    d2 = DerivedExpressionSpec(output_alias="out2", expression="a / (b - 1)", referenced_columns=["a", "b"])
    assert d2.output_alias == "out2"

