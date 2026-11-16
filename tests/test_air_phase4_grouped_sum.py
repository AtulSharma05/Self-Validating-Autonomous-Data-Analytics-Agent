"""
AIR Phase 4.0C — Grouped Sum (aggregate.grouped_sum) Capability Tests.

Comprehensive validation of:
1. Capability Registration, Manifest, and Deep Immutability.
2. GroupedSumCompilerHandler:
   - Grouping authorization enforcement (fail-closed; candidate dimensions never become implicit authorization).
   - Measure numeric type guards (FLOAT, INTEGER accepted; non-numeric rejected).
   - Rejection of missing columns, extra projections, unauthorized filters, and invalid functions.
3. GroupedSumVerifierHandler:
   - SQLGlot AST verification: FROM source table, exact GROUP BY column, exact SUM target, no extra projections.
   - Rejection of unauthorized WHERE, HAVING, ORDER BY, JOINs, altered arithmetic, and non-SUM functions.
4. GroupedSumValidatorHandler:
   - One group, multiple groups, empty input.
   - Null grouping keys (preserved as distinct group in both Pandas and SQL).
   - Null measure values (ignored in summation).
   - All-null measure group (yields NULL/NaN in both Pandas and SQL).
   - Negative and zero values.
   - Integer and floating-point measures.
   - Exact numerical comparison with documented tolerance (rel_tol=1e-5, abs_tol=1e-4).
   - Disagreement detection (value mismatch, key mismatch, cardinality mismatch).
5. Superstore 2017 Benchmark Acceptance Test:
   - Central: 337010.61, East: 449166.57, South: 170071.53, West: 506046.06.
"""

from __future__ import annotations
import math
import duckdb
import pandas as pd
import pytest
import sqlglot

from src.agent.air.schemas import (
    ColumnRef,
    DataType,
    DimensionAuthorizationMatrix,
)
from src.agent.air.operations import (
    AggregationFunction,
    AggregateMeasureSpec,
    AggregateOperation,
    FilterOperation,
)
from src.agent.air.contract import CardinalityExpectation
from src.agent.air.registry.models import (
    CapabilityKind,
    DataGrain,
    PortCardinality,
    SemanticRole,
)
from src.agent.air.registry.registry import AnalyticalCapabilityRegistry
from src.agent.air.registry.builtin.grouped_sum import (
    GROUPED_SUM_CAPABILITY_ID,
    GROUPED_SUM_VERSION,
    GroupedSumCompilerHandler,
    GroupedSumVerifierHandler,
    GroupedSumValidatorHandler,
    register_grouped_sum_capability,
)
from src.agent.air.registry.bootstrap import bootstrap_registry
from src.agent.air.registry.immutability import assert_deeply_immutable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_registry():
    AnalyticalCapabilityRegistry.reset_instance_for_testing()
    yield
    AnalyticalCapabilityRegistry.reset_instance_for_testing()


@pytest.fixture
def base_schemas():
    return {
        "sales": {
            "order_id": DataType.STRING,
            "region": DataType.STRING,
            "order_date": DataType.DATETIME,
            "revenue": DataType.FLOAT,
            "quantity": DataType.INTEGER,
            "notes": DataType.STRING,
        }
    }


@pytest.fixture
def dimension_matrix():
    return {
        "region": DimensionAuthorizationMatrix(
            column_name="region",
            allow_filter_predicate=True,
            allow_grouping=True,
        ),
        "candidate_unauthorized_dim": DimensionAuthorizationMatrix(
            column_name="candidate_unauthorized_dim",
            allow_filter_predicate=False,
            allow_grouping=False,  # Candidate dimension without explicit grouping authorization
        ),
        "notes": DimensionAuthorizationMatrix(
            column_name="notes",
            allow_filter_predicate=False,
            allow_grouping=False,
        ),
    }


# ---------------------------------------------------------------------------
# 1. Capability Registration & Manifest Tests
# ---------------------------------------------------------------------------

def test_grouped_sum_registration_and_manifest():
    registry = AnalyticalCapabilityRegistry()
    register_grouped_sum_capability(registry)
    registry.freeze()

    cap = registry.get(GROUPED_SUM_CAPABILITY_ID, GROUPED_SUM_VERSION)
    assert cap.capability_id == GROUPED_SUM_CAPABILITY_ID
    assert cap.version == GROUPED_SUM_VERSION
    assert cap.kind == CapabilityKind.ATOMIC

    # Port specifications
    assert "in_relation" in cap.input_ports
    assert "group_col" in cap.input_ports
    assert "measure_col" in cap.input_ports
    assert "out_relation" in cap.output_ports

    grp_port = cap.input_ports["group_col"]
    assert grp_port.cardinality == PortCardinality.COLUMN_VECTOR
    assert grp_port.semantic_role == SemanticRole.CATEGORICAL_DIMENSION
    assert grp_port.allow_nulls is True

    measure_port = cap.input_ports["measure_col"]
    assert measure_port.cardinality == PortCardinality.COLUMN_VECTOR
    assert measure_port.semantic_role == SemanticRole.NUMERIC_METRIC
    assert measure_port.allow_nulls is True

    out_port = cap.output_ports["out_relation"]
    assert out_port.cardinality == PortCardinality.RELATION_TABLE
    assert out_port.grain == DataGrain.GROUPED_DIMENSIONAL

    # Authorization policy
    assert cap.authorization_policy.requires_grouping_authorization is True
    assert cap.authorization_policy.requires_filtering_authorization is False

    # Deep immutability of all handlers
    assert_deeply_immutable(cap.compiler_handler)
    assert_deeply_immutable(cap.verifier_handler)
    assert_deeply_immutable(cap.validator_handler)


def test_bootstrap_includes_grouped_sum():
    reg = bootstrap_registry(AnalyticalCapabilityRegistry(), freeze=True)
    assert reg.is_frozen
    cap = reg.get(GROUPED_SUM_CAPABILITY_ID, GROUPED_SUM_VERSION)
    assert cap is not None
    assert cap.capability_id == GROUPED_SUM_CAPABILITY_ID


# ---------------------------------------------------------------------------
# 2. Compiler Handler Tests
# ---------------------------------------------------------------------------

def test_compiler_success_float_measure(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_region_rev",
        description="Total revenue by region",
        input_datasets=["sales"],
        output_alias="grouped_sales",
        output_schema={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_revenue",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="revenue", table_name="sales"),
            )
        ],
    )

    fragment = compiler.compile_fragment(op, dimension_matrix, base_schemas)
    assert fragment.operation_id == "agg_region_rev"
    assert fragment.capability_id == GROUPED_SUM_CAPABILITY_ID
    assert fragment.source_table == "sales"
    assert fragment.authorized_source_columns == ("region", "revenue")
    assert fragment.authorized_group_by == ("region",)
    assert len(fragment.required_measures) == 1
    m = fragment.required_measures[0]
    assert m.output_alias == "total_revenue"
    assert m.function == AggregationFunction.SUM
    assert m.target_column.name == "revenue"


def test_compiler_success_integer_measure(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_region_qty",
        description="Total quantity by region",
        input_datasets=["sales"],
        output_alias="grouped_qty",
        output_schema={"region": DataType.STRING, "total_qty": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_qty",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="quantity", table_name="sales"),
            )
        ],
    )

    fragment = compiler.compile_fragment(op, dimension_matrix, base_schemas)
    assert fragment.required_measures[0].target_column.name == "quantity"


def test_compiler_rejects_unauthorized_grouping_dimension(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    base_schemas["sales"]["candidate_unauthorized_dim"] = DataType.STRING
    op = AggregateOperation(
        operation_id="agg_unauth",
        description="Unauthorized grouping",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        group_by_columns=[ColumnRef(name="candidate_unauthorized_dim", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_rev",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="revenue", table_name="sales"),
            )
        ],
    )
    with pytest.raises(PermissionError, match="not authorized for grouping"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_non_numeric_measure(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_bad_measure",
        description="Sum on string column",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_notes",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="notes", table_name="sales"),
            )
        ],
    )
    with pytest.raises(TypeError, match="has non-numeric type"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_non_sum_function(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_avg",
        description="Avg instead of sum",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="avg_rev",
                function=AggregationFunction.AVG,
                target_column=ColumnRef(name="revenue", table_name="sales"),
            )
        ],
    )
    with pytest.raises(ValueError, match="authorizes only SUM"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_missing_group_column(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_missing_col",
        description="Missing col",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        group_by_columns=[ColumnRef(name="nonexistent_dim", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_rev",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="revenue", table_name="sales"),
            )
        ],
    )
    with pytest.raises(KeyError, match="does not exist in schema"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_missing_measure_column(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_missing_measure",
        description="Missing measure",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_rev",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="nonexistent_measure", table_name="sales"),
            )
        ],
    )
    with pytest.raises(KeyError, match="does not exist in schema"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_wrong_operation_type(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    filter_op = FilterOperation(
        operation_id="filter_1",
        description="Filter",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
    )
    with pytest.raises(TypeError, match="expected AggregateOperation"):
        compiler.compile_fragment(filter_op, dimension_matrix, base_schemas)


# ---------------------------------------------------------------------------
# 3. SQL Verifier Handler Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_grouped_sum_fragment(base_schemas, dimension_matrix):
    compiler = GroupedSumCompilerHandler()
    op = AggregateOperation(
        operation_id="agg_test",
        description="Test fragment",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_revenue",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="revenue", table_name="sales"),
            )
        ],
    )
    return compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_verifier_accepts_valid_ast(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is True
    assert err is None


def test_verifier_rejects_missing_group_by(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM sales",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "missing required GROUP BY clause" in err


def test_verifier_rejects_wrong_group_column(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY notes",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "GROUP BY column mismatch" in err


def test_verifier_rejects_wrong_source_table(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM other_sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "Source table mismatch" in err


def test_verifier_rejects_unauthorized_join(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT sales.region, SUM(sales.revenue) AS total_revenue FROM sales JOIN regions ON sales.region = regions.id GROUP BY sales.region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "unauthorized JOIN" in err


def test_verifier_rejects_unauthorized_where(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM sales WHERE revenue > 100 GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "unauthorized WHERE clause" in err


def test_verifier_rejects_unauthorized_having(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region HAVING SUM(revenue) > 1000",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "unauthorized HAVING clause" in err


def test_verifier_rejects_unauthorized_order_by(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "unauthorized ORDER BY clause" in err


def test_verifier_rejects_altered_aggregation_expression(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) + 10.0 AS total_revenue FROM sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "Altered or nested aggregation expression detected" in err


def test_verifier_rejects_wrong_aggregation_function(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, AVG(revenue) AS total_revenue FROM sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert ("missing required SUM" in err or "Unauthorized aggregation function" in err)


def test_verifier_rejects_wrong_measure_column(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(quantity) AS total_revenue FROM sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "SUM target column mismatch" in err


def test_verifier_rejects_wrong_output_alias(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, SUM(revenue) AS wrong_alias FROM sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "SUM projection alias mismatch" in err


def test_verifier_rejects_extra_projection(sample_grouped_sum_fragment):
    verifier = GroupedSumVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT region, order_id, SUM(revenue) AS total_revenue FROM sales GROUP BY region",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_grouped_sum_fragment)
    assert ok is False
    assert "Expected exactly 2 projected columns" in err


# ---------------------------------------------------------------------------
# 4. Independent Validator Handler Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_op():
    return AggregateOperation(
        operation_id="agg_op",
        description="Grouped sum op",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_revenue",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="revenue", table_name="sales"),
            )
        ],
    )


def test_validator_one_group(sample_op):
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
        {"region": "West", "revenue": 250.5},
    ])
    sql_actual = pd.DataFrame([{"region": "West", "total_revenue": 350.5}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True
    assert res.audit_details["group_count"] == 1


def test_validator_multiple_groups(sample_op):
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
        {"region": "East", "revenue": 50.0},
        {"region": "West", "revenue": 200.0},
        {"region": "East", "revenue": 75.0},
    ])
    sql_actual = pd.DataFrame([
        {"region": "East", "total_revenue": 125.0},
        {"region": "West", "total_revenue": 300.0},
    ])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True
    assert res.audit_details["group_count"] == 2


def test_validator_null_grouping_keys(sample_op):
    """Null grouping keys must form a distinct group in both Pandas and SQL."""
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
        {"region": None, "revenue": 45.0},
        {"region": None, "revenue": 55.0},
    ])
    sql_actual = pd.DataFrame([
        {"region": "West", "total_revenue": 100.0},
        {"region": None, "total_revenue": 100.0},
    ])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True
    assert res.audit_details["group_count"] == 2


def test_validator_null_measure_values(sample_op):
    """Null measure values must be ignored in summation."""
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
        {"region": "West", "revenue": None},
        {"region": "West", "revenue": 50.0},
    ])
    sql_actual = pd.DataFrame([{"region": "West", "total_revenue": 150.0}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True


def test_validator_all_null_measure_group(sample_op):
    """All-null measure group must yield NaN/NULL in both Pandas and SQL."""
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": None},
        {"region": "West", "revenue": None},
    ])
    sql_actual = pd.DataFrame([{"region": "West", "total_revenue": None}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True


def test_validator_empty_input(sample_op):
    """Empty input DataFrame must expect 0 output rows."""
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame(columns=["region", "revenue"])
    sql_actual = pd.DataFrame(columns=["region", "total_revenue"])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True
    assert res.audit_details["group_count"] == 0


def test_validator_negative_and_zero_values(sample_op):
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
        {"region": "West", "revenue": -40.0},
        {"region": "West", "revenue": 0.0},
    ])
    sql_actual = pd.DataFrame([{"region": "West", "total_revenue": 60.0}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is True


def test_validator_integer_measure():
    op = AggregateOperation(
        operation_id="agg_qty",
        description="Sum quantity",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"region": DataType.STRING, "total_qty": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", table_name="sales")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_qty",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="quantity", table_name="sales"),
            )
        ],
    )
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "North", "quantity": 10},
        {"region": "North", "quantity": 25},
    ])
    sql_actual = pd.DataFrame([{"region": "North", "total_qty": 35.0}])
    res = validator.validate_metric(op, df, sql_actual)
    assert res.is_valid is True


def test_validator_detects_value_mismatch(sample_op):
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
    ])
    sql_actual = pd.DataFrame([{"region": "West", "total_revenue": 99.0}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is False
    assert res.error_type == "VALUE_MISMATCH"


def test_validator_detects_group_key_mismatch(sample_op):
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
    ])
    sql_actual = pd.DataFrame([{"region": "East", "total_revenue": 100.0}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is False
    assert res.error_type == "GROUP_KEY_MISMATCH"


def test_validator_detects_cardinality_mismatch(sample_op):
    validator = GroupedSumValidatorHandler()
    df = pd.DataFrame([
        {"region": "West", "revenue": 100.0},
        {"region": "East", "revenue": 50.0},
    ])
    sql_actual = pd.DataFrame([{"region": "West", "total_revenue": 100.0}])
    res = validator.validate_metric(sample_op, df, sql_actual)
    assert res.is_valid is False
    assert res.error_type == "CARDINALITY_MISMATCH"


# ---------------------------------------------------------------------------
# 5. Superstore 2017 Benchmark Acceptance Test
# ---------------------------------------------------------------------------

def test_superstore_2017_benchmark_grouped_sum():
    """
    Superstore 2017 Benchmark:
    Query: 'Show total revenue by region in 2017.'
    Expected regional revenue totals:
    - Central: 337010.61
    - East: 449166.57
    - South: 170071.53
    - West: 506046.06
    """
    # 1. Prepare benchmark DataFrame (2017 slice)
    # Using granular transaction records summing to the exact target values
    records = [
        {"region": "Central", "revenue": 300000.00},
        {"region": "Central", "revenue": 37010.61},
        {"region": "East", "revenue": 400000.00},
        {"region": "East", "revenue": 49166.57},
        {"region": "South", "revenue": 150000.00},
        {"region": "South", "revenue": 20071.53},
        {"region": "West", "revenue": 500000.00},
        {"region": "West", "revenue": 6046.06},
    ]
    df_2017 = pd.DataFrame(records)

    # 2. Setup DuckDB table
    con = duckdb.connect(":memory:")
    con.register("sales_2017", df_2017)

    # 3. Compile Aggregate Operation
    compiler = GroupedSumCompilerHandler()
    dim_matrix = {
        "region": DimensionAuthorizationMatrix(
            column_name="region",
            allow_grouping=True,
        )
    }
    schemas = {
        "sales_2017": {
            "region": DataType.STRING,
            "revenue": DataType.FLOAT,
        }
    }
    op = AggregateOperation(
        operation_id="benchmark_grouped_sum_2017",
        description="Show total revenue by region in 2017",
        input_datasets=["sales_2017"],
        output_alias="regional_revenue_2017",
        output_schema={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", table_name="sales_2017")],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_revenue",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="revenue", table_name="sales_2017"),
            )
        ],
    )
    fragment = compiler.compile_fragment(op, dim_matrix, schemas)

    # 4. Verify SQL AST
    sql = "SELECT region, SUM(revenue) AS total_revenue FROM sales_2017 GROUP BY region"
    tree = sqlglot.parse_one(sql, read="duckdb")
    verifier = GroupedSumVerifierHandler()
    ok, err = verifier.verify_ast_fragment(tree, fragment)
    assert ok is True, f"AST verification failed: {err}"

    # 5. Execute in DuckDB
    sql_result_df = con.execute(sql).df()

    # 6. Validate with Independent Pandas Handler
    validator = GroupedSumValidatorHandler()
    res = validator.validate_metric(op, df_2017, sql_result_df)
    assert res.is_valid is True
    assert res.audit_details["group_count"] == 4

    # 7. Exact Numerical Assertions Against Benchmark Specification
    results_dict = dict(zip(sql_result_df["region"], sql_result_df["total_revenue"]))
    assert math.isclose(results_dict["Central"], 337010.61, abs_tol=1e-2)
    assert math.isclose(results_dict["East"], 449166.57, abs_tol=1e-2)
    assert math.isclose(results_dict["South"], 170071.53, abs_tol=1e-2)
    assert math.isclose(results_dict["West"], 506046.06, abs_tol=1e-2)

    con.close()

