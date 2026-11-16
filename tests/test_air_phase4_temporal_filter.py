"""
AIR Phase 4.0B — Temporal Filter (filter.temporal_range) Capability Tests.

Comprehensive validation of:
1. Capability Registration, Manifest, and Deep Immutability.
2. TemporalFilterCompilerHandler: Schema, dimension authorization, and interval validation.
3. TemporalFilterVerifierHandler: SQLGlot AST WHERE clause verification, strict half-open interval checks.
4. TemporalFilterValidatorHandler: Strict half-open semantics [start, end), sub-microsecond boundary checks,
   NULL exclusion, timezone awareness checks, and independent Pandas ground-truth validation.
5. End-to-end compilation, AST verification, DuckDB execution, and Pandas validation.
"""

from __future__ import annotations
import datetime
from types import MappingProxyType
import duckdb
import pandas as pd
import pytest
import sqlglot

from src.agent.air.schemas import (
    ColumnRef,
    ComparisonOperator,
    DataType,
    DimensionAuthorizationMatrix,
    Predicate,
)
from src.agent.air.operations import (
    FilterOperation,
    AggregateOperation,
)
from src.agent.air.contract import (
    CardinalityExpectation,
    PredicateOperator,
    TypedPredicate,
)
from src.agent.air.registry.models import (
    CapabilityKind,
    DataGrain,
    PortCardinality,
    SemanticRole,
)
from src.agent.air.registry.registry import AnalyticalCapabilityRegistry
from src.agent.air.registry.builtin.temporal_filter import (
    TEMPORAL_FILTER_CAPABILITY_ID,
    TEMPORAL_FILTER_VERSION,
    TemporalFilterCompilerHandler,
    TemporalFilterVerifierHandler,
    TemporalFilterValidatorHandler,
    register_temporal_filter_capability,
    parse_and_validate_temporal_boundary,
)
from src.agent.air.registry.ports import (
    BaseTableColumnBinding,
    validate_port_binding,
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
            "order_date": DataType.DATETIME,
            "ship_date": DataType.DATE,
            "revenue": DataType.FLOAT,
            "region": DataType.STRING,
            "quantity": DataType.INTEGER,
        }
    }


@pytest.fixture
def dimension_matrix():
    return {
        "order_date": DimensionAuthorizationMatrix(
            column_name="order_date",
            allow_filter_predicate=True,
            allow_grouping=False,
        ),
        "ship_date": DimensionAuthorizationMatrix(
            column_name="ship_date",
            allow_filter_predicate=True,
            allow_grouping=False,
        ),
        "region": DimensionAuthorizationMatrix(
            column_name="region",
            allow_filter_predicate=True,
            allow_grouping=True,
        ),
        "unauthorized_date": DimensionAuthorizationMatrix(
            column_name="unauthorized_date",
            allow_filter_predicate=False,
            allow_grouping=False,
        ),
    }


# ---------------------------------------------------------------------------
# 1. Capability Registration & Manifest Tests
# ---------------------------------------------------------------------------

def test_temporal_filter_registration_and_manifest():
    registry = AnalyticalCapabilityRegistry()
    register_temporal_filter_capability(registry)
    registry.freeze()

    cap = registry.get(TEMPORAL_FILTER_CAPABILITY_ID, TEMPORAL_FILTER_VERSION)
    assert cap.capability_id == TEMPORAL_FILTER_CAPABILITY_ID
    assert cap.version == TEMPORAL_FILTER_VERSION
    assert cap.kind == CapabilityKind.ATOMIC

    # Port checks
    assert "in_relation" in cap.input_ports
    assert "time_col" in cap.input_ports
    assert "out_relation" in cap.output_ports

    in_port = cap.input_ports["in_relation"]
    assert in_port.payload_type.is_relation is True
    assert in_port.cardinality == PortCardinality.RELATION_TABLE

    time_port = cap.input_ports["time_col"]
    assert time_port.payload_type.is_relation is False
    assert time_port.payload_type.primitive_type == DataType.DATETIME
    assert time_port.semantic_role == SemanticRole.TEMPORAL_BOUNDARY

    out_port = cap.output_ports["out_relation"]
    assert out_port.payload_type.is_relation is True
    assert out_port.cardinality == PortCardinality.RELATION_TABLE

    # Authorization policy
    assert cap.authorization_policy.requires_filtering_authorization is True

    # Immutability validation
    assert_deeply_immutable(cap.compiler_handler)
    assert_deeply_immutable(cap.verifier_handler)
    assert_deeply_immutable(cap.validator_handler)


def test_bootstrap_registry():
    reg = bootstrap_registry(AnalyticalCapabilityRegistry(), freeze=True)
    assert reg.is_frozen
    cap = reg.get(TEMPORAL_FILTER_CAPABILITY_ID, TEMPORAL_FILTER_VERSION)
    assert cap is not None


# ---------------------------------------------------------------------------
# 2. Compiler Handler Tests
# ---------------------------------------------------------------------------

def test_compiler_success_between_predicate(base_schemas, dimension_matrix):
    compiler = TemporalFilterCompilerHandler()
    op = FilterOperation(
        operation_id="filter_2017",
        description="Filter 2017 sales",
        input_datasets=["sales"],
        output_alias="sales_2017",
        output_schema={"order_id": DataType.STRING, "order_date": DataType.DATETIME},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )

    fragment = compiler.compile_fragment(op, dimension_matrix, base_schemas)
    assert fragment.operation_id == "filter_2017"
    assert fragment.capability_id == TEMPORAL_FILTER_CAPABILITY_ID
    assert fragment.source_table == "sales"
    assert fragment.authorized_source_columns == ("order_date",)
    assert len(fragment.required_filters) == 1
    req = fragment.required_filters[0]
    assert req.operator == PredicateOperator.BETWEEN_HALF_OPEN
    assert req.literal_value == "2017-01-01 00:00:00"
    assert req.secondary_literal == "2018-01-01 00:00:00"
    assert req.is_temporal is True


def test_compiler_success_paired_predicates(base_schemas, dimension_matrix):
    compiler = TemporalFilterCompilerHandler()
    op = FilterOperation(
        operation_id="filter_2017",
        description="Filter 2017 sales via paired bounds",
        input_datasets=["sales"],
        output_alias="sales_2017",
        output_schema={"order_id": DataType.STRING, "order_date": DataType.DATETIME},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.GTE,
                literal_value="2017-01-01",
            ),
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.LT,
                literal_value="2018-01-01",
            ),
        ],
    )

    fragment = compiler.compile_fragment(op, dimension_matrix, base_schemas)
    assert len(fragment.required_filters) == 1
    req = fragment.required_filters[0]
    assert req.literal_value == "2017-01-01"
    assert req.secondary_literal == "2018-01-01"


def test_compiler_rejects_wrong_operation_type(base_schemas, dimension_matrix):
    compiler = TemporalFilterCompilerHandler()
    agg_op = AggregateOperation(
        operation_id="agg_1",
        description="Aggregate",
        input_datasets=["sales"],
        output_alias="agg_out",
        output_schema={},
    )
    with pytest.raises(TypeError, match="expected FilterOperation"):
        compiler.compile_fragment(agg_op, dimension_matrix, base_schemas)


def test_compiler_rejects_non_temporal_column(base_schemas, dimension_matrix):
    compiler = TemporalFilterCompilerHandler()
    op = FilterOperation(
        operation_id="filter_bad",
        description="Filter on string column",
        input_datasets=["sales"],
        output_alias="sales_filtered",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="region", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01", "2018-01-01"],
            )
        ],
    )
    with pytest.raises(TypeError, match="Port contract authorizes only"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_unauthorized_dimension(base_schemas, dimension_matrix):
    compiler = TemporalFilterCompilerHandler()
    base_schemas["sales"]["unauthorized_date"] = DataType.DATETIME
    op = FilterOperation(
        operation_id="filter_unauth",
        description="Filter unauthorized column",
        input_datasets=["sales"],
        output_alias="sales_filtered",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="unauthorized_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01", "2018-01-01"],
            )
        ],
    )
    with pytest.raises(PermissionError, match="not authorized for filter predicates"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


def test_compiler_rejects_contradictory_interval(base_schemas, dimension_matrix):
    compiler = TemporalFilterCompilerHandler()
    op = FilterOperation(
        operation_id="filter_contra",
        description="Contradictory interval",
        input_datasets=["sales"],
        output_alias="sales_filtered",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2018-01-01", "2017-01-01"],
            )
        ],
    )
    with pytest.raises(ValueError, match="Contradictory temporal range"):
        compiler.compile_fragment(op, dimension_matrix, base_schemas)


# ---------------------------------------------------------------------------
# 3. AST Verifier Handler Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fragment():
    return TemporalFilterCompilerHandler().compile_fragment(
        FilterOperation(
            operation_id="filter_2017",
            description="Filter 2017 sales",
            input_datasets=["sales"],
            output_alias="sales_2017",
            output_schema={"order_id": DataType.STRING},
            predicates=[
                Predicate(
                    column=ColumnRef(name="order_date", table_name="sales"),
                    operator=ComparisonOperator.BETWEEN,
                    literal_value=["2017-01-01", "2018-01-01"],
                )
            ],
        ),
        {
            "order_date": DimensionAuthorizationMatrix(
                column_name="order_date",
                allow_filter_predicate=True,
            )
        },
        {"sales": {"order_date": DataType.DATETIME, "order_id": DataType.STRING}},
    )


def test_verifier_accepts_valid_ast(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is True
    assert err is None


def test_verifier_accepts_parenthesized_ast(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE (order_date >= '2017-01-01') AND (order_date < '2018-01-01')",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is True
    assert err is None


def test_verifier_rejects_missing_where_clause(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one("SELECT order_id FROM sales", read="duckdb")
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is False
    assert "missing a WHERE clause" in err


def test_verifier_rejects_inclusive_upper_bound_lte(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE order_date >= '2017-01-01' AND order_date <= '2018-01-01'",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is False
    assert "Upper boundary uses inclusive (<= '2018-01-01')" in err


def test_verifier_rejects_inclusive_between(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is False
    assert "BETWEEN operator is inclusive of both bounds" in err


def test_verifier_rejects_strictly_greater_lower_bound(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE order_date > '2017-01-01' AND order_date < '2018-01-01'",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is False
    assert "Lower boundary uses strictly greater (>)" in err


def test_verifier_rejects_weakened_or_condition(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE (order_date >= '2017-01-01' OR region = 'West') AND order_date < '2018-01-01'",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is False
    assert "weakened OR condition" in err


def test_verifier_rejects_unauthorized_extra_predicate(sample_fragment):
    verifier = TemporalFilterVerifierHandler()
    ast = sqlglot.parse_one(
        "SELECT order_id FROM sales WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01' AND region = 'East'",
        read="duckdb",
    )
    ok, err = verifier.verify_ast_fragment(ast, sample_fragment)
    assert ok is False
    assert "Unauthorized filter predicate on column(s)" in err


# ---------------------------------------------------------------------------
# 4. Independent Validator Handler Tests (Boundary & Timezone Semantics)
# ---------------------------------------------------------------------------

@pytest.fixture
def boundary_df():
    """
    Carefully curated dataset testing:
    - Exact lower bound (2017-01-01 00:00:00.000000) -> INCLUDED
    - 1 microsecond before start (2016-12-31 23:59:59.999999) -> EXCLUDED
    - Mid-interval timestamp (2017-06-15 12:00:00.000000) -> INCLUDED
    - 1 microsecond before end (2017-12-31 23:59:59.999999) -> INCLUDED
    - Exact upper bound (2018-01-01 00:00:00.000000) -> EXCLUDED
    - 1 microsecond after end (2018-01-01 00:00:00.000001) -> EXCLUDED
    - Far out of bounds (2015-01-01, 2020-01-01) -> EXCLUDED
    - NULL / NaT timestamp -> EXCLUDED
    """
    records = [
        {"order_id": "R1_EXACT_START", "order_date": pd.Timestamp("2017-01-01 00:00:00.000000")},
        {"order_id": "R2_MICRO_BEFORE_START", "order_date": pd.Timestamp("2016-12-31 23:59:59.999999")},
        {"order_id": "R3_MID_INTERVAL", "order_date": pd.Timestamp("2017-06-15 12:00:00.000000")},
        {"order_id": "R4_MICRO_BEFORE_END", "order_date": pd.Timestamp("2017-12-31 23:59:59.999999")},
        {"order_id": "R5_EXACT_END", "order_date": pd.Timestamp("2018-01-01 00:00:00.000000")},
        {"order_id": "R6_MICRO_AFTER_END", "order_date": pd.Timestamp("2018-01-01 00:00:00.000001")},
        {"order_id": "R7_FAR_PAST", "order_date": pd.Timestamp("2015-05-01 10:00:00.000000")},
        {"order_id": "R8_FAR_FUTURE", "order_date": pd.Timestamp("2020-05-01 10:00:00.000000")},
        {"order_id": "R9_NULL", "order_date": pd.NaT},
    ]
    return pd.DataFrame(records)


def test_independent_validator_exact_boundaries(boundary_df):
    validator = TemporalFilterValidatorHandler()
    op = FilterOperation(
        operation_id="test_filter",
        description="Filter 2017",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"order_id": DataType.STRING},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )

    # Expected matching records are R1, R3, R4 (exactly 3 rows)
    expected_ids = {"R1_EXACT_START", "R3_MID_INTERVAL", "R4_MICRO_BEFORE_END"}

    # Mock SQL result matching expected exactly
    matching_slice = boundary_df[boundary_df["order_id"].isin(expected_ids)].copy()

    res = validator.validate_metric(op, boundary_df, matching_slice)
    assert res.is_valid is True
    assert res.audit_details["expected_rows"] == 3


def test_independent_validator_rejects_inclusion_of_exact_upper_bound(boundary_df):
    validator = TemporalFilterValidatorHandler()
    op = FilterOperation(
        operation_id="test_filter",
        description="Filter 2017",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"order_id": DataType.STRING},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )

    # Erroneous SQL result that included exact end R5 (4 rows instead of 3)
    erroneous_ids = {"R1_EXACT_START", "R3_MID_INTERVAL", "R4_MICRO_BEFORE_END", "R5_EXACT_END"}
    erroneous_slice = boundary_df[boundary_df["order_id"].isin(erroneous_ids)].copy()

    res = validator.validate_metric(op, boundary_df, erroneous_slice)
    assert res.is_valid is False
    assert res.error_type == "CARDINALITY_MISMATCH"
    assert "Expected 3 rows, got 4 rows" in res.error_message


def test_independent_validator_rejects_row_membership_mismatch(boundary_df):
    validator = TemporalFilterValidatorHandler()
    op = FilterOperation(
        operation_id="test_filter",
        description="Filter 2017",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"order_id": DataType.STRING},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )

    # 3 rows, but R5 included instead of R1 (wrong members)
    wrong_ids = {"R5_EXACT_END", "R3_MID_INTERVAL", "R4_MICRO_BEFORE_END"}
    wrong_slice = boundary_df[boundary_df["order_id"].isin(wrong_ids)].copy()

    res = validator.validate_metric(op, boundary_df, wrong_slice)
    assert res.is_valid is False
    assert res.error_type == "ROW_MEMBERSHIP_MISMATCH"


def test_independent_validator_timezone_aware_matching():
    validator = TemporalFilterValidatorHandler()
    # UTC-aware dataframe
    df = pd.DataFrame([
        {"id": "A", "dt": pd.Timestamp("2017-05-01 00:00:00+00:00")},
        {"id": "B", "dt": pd.Timestamp("2018-05-01 00:00:00+00:00")},
    ])
    op = FilterOperation(
        operation_id="tz_filter",
        description="Filter UTC",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="dt", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01T00:00:00+00:00", "2018-01-01T00:00:00+00:00"],
            )
        ],
    )

    actual_slice = df[df["id"] == "A"].copy()
    res = validator.validate_metric(op, df, actual_slice)
    assert res.is_valid is True
    assert res.audit_details["expected_rows"] == 1


def test_independent_validator_timezone_mismatch_fails_closed():
    validator = TemporalFilterValidatorHandler()
    # Naive dataframe vs UTC-aware predicate bounds
    df = pd.DataFrame([
        {"id": "A", "dt": pd.Timestamp("2017-05-01 00:00:00")},
    ])
    op = FilterOperation(
        operation_id="tz_filter",
        description="Filter UTC on naive df",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="dt", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01T00:00:00+00:00", "2018-01-01T00:00:00+00:00"],
            )
        ],
    )

    res = validator.validate_metric(op, df, df)
    assert res.is_valid is False
    assert res.error_type == "TIMEZONE_MISMATCH"
    assert "Cannot compare timezone-naive and timezone-aware" in res.error_message


def test_independent_validator_rejects_string_column_fail_closed():
    validator = TemporalFilterValidatorHandler()
    df = pd.DataFrame([
        {"id": "A", "dt_str": "2017-05-01"},
    ])
    op = FilterOperation(
        operation_id="str_filter",
        description="Filter on string col",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="dt_str", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01", "2018-01-01"],
            )
        ],
    )

    res = validator.validate_metric(op, df, df)
    assert res.is_valid is False
    assert res.error_type == "TYPE_MISMATCH"
    assert "rejected fail-closed without lossy conversion" in res.error_message


# ---------------------------------------------------------------------------
# 5. End-to-End DuckDB Execution and Validation Test
# ---------------------------------------------------------------------------

def test_e2e_duckdb_execution_and_pandas_validation(boundary_df, dimension_matrix, base_schemas):
    # 1. Setup DuckDB table
    con = duckdb.connect(":memory:")
    con.register("sales", boundary_df)

    # 2. Compile Filter Operation
    compiler = TemporalFilterCompilerHandler()
    op = FilterOperation(
        operation_id="op_filter_2017",
        description="Extract 2017 records",
        input_datasets=["sales"],
        output_alias="sales_2017",
        output_schema={"order_id": DataType.STRING, "order_date": DataType.DATETIME},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )
    fragment = compiler.compile_fragment(op, dimension_matrix, base_schemas)

    # 3. Verify Generated SQL AST
    sql = (
        "SELECT order_id, order_date "
        "FROM sales "
        "WHERE order_date >= '2017-01-01 00:00:00' AND order_date < '2018-01-01 00:00:00'"
    )
    tree = sqlglot.parse_one(sql, read="duckdb")
    verifier = TemporalFilterVerifierHandler()
    ok, err = verifier.verify_ast_fragment(tree, fragment)
    assert ok is True, f"AST verification failed: {err}"

    # 4. Execute Query in DuckDB
    sql_result_df = con.execute(sql).df()

    # 5. Independent Pandas Validation
    validator = TemporalFilterValidatorHandler()
    res = validator.validate_metric(op, boundary_df, sql_result_df)
    assert res.is_valid is True
    assert res.audit_details["expected_rows"] == 3
    assert set(sql_result_df["order_id"]) == {"R1_EXACT_START", "R3_MID_INTERVAL", "R4_MICRO_BEFORE_END"}

    con.close()


# ---------------------------------------------------------------------------
# 6. Item 1 & 2 Targeted Verification Tests
# ---------------------------------------------------------------------------

def test_temporal_port_type_consistency_declared_in_contract():
    """Item 2: Verifies registry contract accurately represents accepted types."""
    registry = AnalyticalCapabilityRegistry()
    register_temporal_filter_capability(registry)
    registry.freeze()

    cap = registry.get(TEMPORAL_FILTER_CAPABILITY_ID, TEMPORAL_FILTER_VERSION)
    time_port = cap.input_ports["time_col"]

    # Accurately declares accepted primitive types
    assert time_port.payload_type.allowed_primitive_types == (DataType.DATETIME, DataType.DATE)
    assert time_port.accepts_data_type(DataType.DATETIME) is True
    assert time_port.accepts_data_type(DataType.DATE) is True

    # STRING and numeric inputs fail closed
    assert time_port.accepts_data_type(DataType.STRING) is False
    assert time_port.accepts_data_type(DataType.INTEGER) is False
    assert time_port.accepts_data_type(DataType.FLOAT) is False
    assert time_port.accepts_data_type(DataType.BOOLEAN) is False

    # Binding validation against port contract
    valid_dt_binding = BaseTableColumnBinding(
        consumer_port_name="time_col",
        table_name="sales",
        column_name="order_date",
        expected_data_type=DataType.DATETIME,
    )
    assert validate_port_binding(time_port, valid_dt_binding) is True

    valid_date_binding = BaseTableColumnBinding(
        consumer_port_name="time_col",
        table_name="sales",
        column_name="ship_date",
        expected_data_type=DataType.DATE,
    )
    assert validate_port_binding(time_port, valid_date_binding) is True

    invalid_str_binding = BaseTableColumnBinding(
        consumer_port_name="time_col",
        table_name="sales",
        column_name="region",
        expected_data_type=DataType.STRING,
    )
    assert validate_port_binding(time_port, invalid_str_binding) is False


def test_compiler_accepts_date_only_when_declared_by_contract(base_schemas, dimension_matrix):
    """Item 2: Proves DATE input is accepted only when declared by contract."""
    # Compiler declaring only DATETIME
    compiler_dt_only = TemporalFilterCompilerHandler(accepted_types=(DataType.DATETIME,))
    
    op_date = FilterOperation(
        operation_id="filter_date",
        description="Filter on date column",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"order_id": DataType.STRING},
        predicates=[
            Predicate(
                column=ColumnRef(name="ship_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01", "2018-01-01"],
            )
        ],
    )
    # ship_date is DATE, so compiler_dt_only must reject it fail closed
    with pytest.raises(TypeError, match="Port contract authorizes only: \\('datetime',\\)"):
        compiler_dt_only.compile_fragment(op_date, dimension_matrix, base_schemas)

    # Compiler declaring only DATE
    compiler_date_only = TemporalFilterCompilerHandler(accepted_types=(DataType.DATE,))
    op_dt = FilterOperation(
        operation_id="filter_dt",
        description="Filter on datetime column",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"order_id": DataType.STRING},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )
    # order_date is DATETIME, so compiler_date_only must reject it fail closed
    with pytest.raises(TypeError, match="Port contract authorizes only: \\('date',\\)"):
        compiler_date_only.compile_fragment(op_dt, dimension_matrix, base_schemas)

    # When declared, DATE is accepted
    fragment_date = compiler_date_only.compile_fragment(op_date, dimension_matrix, base_schemas)
    assert fragment_date.authorized_source_columns == ("ship_date",)


def test_contract_and_compiler_cannot_disagree_about_accepted_types():
    """Item 2: Proves contract and compiler cannot disagree about accepted types."""
    registry = AnalyticalCapabilityRegistry()
    # Attempting to register capability with mismatched contract and compiler
    # register_temporal_filter_capability enforces strict equality between port and compiler
    register_temporal_filter_capability(registry, accepted_types=(DataType.DATETIME,))
    cap = registry.get(TEMPORAL_FILTER_CAPABILITY_ID, TEMPORAL_FILTER_VERSION)
    time_port = cap.input_ports["time_col"]
    assert time_port.payload_type.allowed_primitive_types == cap.compiler_handler.accepted_types


def test_dst_ambiguous_local_timestamp_fails_closed():
    """Item 1: Proves ambiguous local timestamps during DST fall back fail closed."""
    # 2024-11-03 01:30:00 occurred twice in America/New_York (EDT UTC-4 and EST UTC-5)
    with pytest.raises(ValueError, match="Cannot infer dst time"):
        parse_and_validate_temporal_boundary("2024-11-03 01:30:00", target_tz="America/New_York")


def test_dst_nonexistent_local_timestamp_fails_closed():
    """Item 1: Proves nonexistent local timestamps during DST spring forward fail closed."""
    # 2024-03-10 02:30:00 does not exist in America/New_York due to 1-hour forward jump
    with pytest.raises(ValueError, match="nonexistent time due to daylight savings time"):
        parse_and_validate_temporal_boundary("2024-03-10 02:30:00", target_tz="America/New_York")


def test_explicit_timezone_interpretation_and_no_silent_localization():
    """Item 1: Naive timestamps remain naive; silent localization is prohibited."""
    # Naive boundary remains strictly naive
    ts_naive = parse_and_validate_temporal_boundary("2017-01-01 00:00:00")
    assert ts_naive.tz is None

    # Fixed offset boundary remains aware
    ts_utc = parse_and_validate_temporal_boundary("2017-01-01T00:00:00Z")
    assert ts_utc.tz is not None

    # Comparing naive df against aware boundaries fails closed with TIMEZONE_MISMATCH
    validator = TemporalFilterValidatorHandler()
    df_naive = pd.DataFrame([{"id": "1", "dt": pd.Timestamp("2017-06-01 00:00:00")}])
    op_aware = FilterOperation(
        operation_id="tz_aware_op",
        description="Filter aware",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={},
        predicates=[
            Predicate(
                column=ColumnRef(name="dt", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01T00:00:00+00:00", "2018-01-01T00:00:00+00:00"],
            )
        ],
    )
    res = validator.validate_metric(op_aware, df_naive, df_naive)
    assert res.is_valid is False
    assert res.error_type == "TIMEZONE_MISMATCH"
    assert "Silent localization or timezone inference is strictly prohibited" in res.error_message


def test_compiler_verifier_pandas_boundary_agreement(dimension_matrix, base_schemas):
    """Item 1: Verifies exact agreement between compiler, verifier, and pandas."""
    compiler = TemporalFilterCompilerHandler()
    verifier = TemporalFilterVerifierHandler()
    validator = TemporalFilterValidatorHandler()

    op = FilterOperation(
        operation_id="op_agree",
        description="Agree test",
        input_datasets=["sales"],
        output_alias="out",
        output_schema={"order_id": DataType.STRING},
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", table_name="sales"),
                operator=ComparisonOperator.BETWEEN,
                literal_value=["2017-01-01 00:00:00", "2018-01-01 00:00:00"],
            )
        ],
    )
    fragment = compiler.compile_fragment(op, dimension_matrix, base_schemas)
    req = fragment.required_filters[0]

    # Verify AST matches exact emitted literals
    sql_exact = (
        f"SELECT order_id FROM sales "
        f"WHERE order_date >= '{req.literal_value}' AND order_date < '{req.secondary_literal}'"
    )
    ast = sqlglot.parse_one(sql_exact, read="duckdb")
    ok, err = verifier.verify_ast_fragment(ast, fragment)
    assert ok is True

    # Pandas ground truth evaluates exact boundaries
    df = pd.DataFrame([
        {"order_id": "LOWER_BOUND", "order_date": pd.Timestamp("2017-01-01 00:00:00")},
        {"order_id": "UPPER_BOUND", "order_date": pd.Timestamp("2018-01-01 00:00:00")},
    ])
    sql_actual = df[df["order_id"] == "LOWER_BOUND"].copy()
    res = validator.validate_metric(op, df, sql_actual)
    assert res.is_valid is True
    assert res.audit_details["expected_rows"] == 1


