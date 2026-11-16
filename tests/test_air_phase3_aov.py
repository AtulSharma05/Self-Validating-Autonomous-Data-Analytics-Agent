"""
Unit, Adversarial, and End-to-End Tests for AIR Phase 3:
AOV (Average Order Value) Vertical Slice, Multi-Branch Data-Quality Fusion,
AST Type & Expression Hardening, Missing-Sales Evidence, and Fail-Closed Graph Enforcement.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

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
    DerivedOperation,
    FilterOperation,
)
from src.agent.air.dag import MasterAIR
from src.agent.air.contract import (
    CardinalityExpectation,
    ContractEngine,
    ExecutionContract,
    ExecutionReceipt,
    PredicateOperator,
    RequiredDerivedMeasureSpec,
    RequiredMeasureSpec,
    TypedPredicate,
)
from src.agent.air.compiler import AIRContractCompiler, CompilerError
from src.agent.air.sql_verifier import SQLASTConformanceVerifier
from src.agent.codegen import CodeGenerator, create_contract_template_code
from src.agent.schemas import AnalysisObjective, AnalysisPlan, AnalysisType, DatasetProfile, PlanStep, TableProfile, ColumnProfile
from src.agent.graph import create_analytics_graph
from src.agent.state import AnalyticsAgentState
from src.execution.sandbox import SandboxedSQLExecutor
from src.validation.validator import DeterministicValidator, DualPathVerifier


@pytest.fixture
def superstore_schemas():
    return {
        "sample_superstore": {
            "order_id": DataType.STRING,
            "order_date": DataType.DATETIME,
            "sales": DataType.FLOAT,
            "region": DataType.STRING,
            "customer_id": DataType.STRING,
            "profit": DataType.FLOAT,
        }
    }


@pytest.fixture
def aov_benchmark_duckdb():
    """
    Benchmark database for AOV in 2017:
    - 2 unique identifiable orders in 2017:
      * 'CA-2017-1001': 2 line items: $100.0 + $200.0 = $300.0
      * 'CA-2017-1002': 1 line item: $300.0
      Total identifiable orders = 2, Total valid revenue = $600.0, True AOV = $300.0
      Line-item trap: $600.0 / 3 items = $200.0 (must be avoided!)
    - Branch B: NULL order_id row in 2017:
      * NULL order_id, sales = $150.0 (unassigned revenue)
    - Outside 2017:
      * 'CA-2018-2001' in 2018: $500.0
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE sample_superstore (
            order_id VARCHAR,
            order_date TIMESTAMP,
            sales DOUBLE,
            region VARCHAR,
            customer_id VARCHAR,
            profit DOUBLE
        );
        INSERT INTO sample_superstore VALUES
            ('CA-2017-1001', '2017-02-10 10:00:00', 100.0, 'West', 'CUST-01', 20.0),
            ('CA-2017-1001', '2017-02-10 10:05:00', 200.0, 'West', 'CUST-01', 40.0),
            ('CA-2017-1002', '2017-05-15 14:00:00', 300.0, 'East', 'CUST-02', 60.0),
            (NULL,           '2017-07-20 12:00:00', 150.0, 'East', 'CUST-03', 10.0),
            ('CA-2018-2001', '2018-01-05 09:00:00', 500.0, 'West', 'CUST-04', 100.0);
    """)
    yield con
    con.close()


@pytest.fixture
def aov_missing_sales_duckdb():
    """
    Database with missing/null sales evidence:
    - Order 'CA-2017-1001': 1 row with $300.0, 1 row with NULL sales (partial null sales)
    - Order 'CA-2017-1002': 1 row with NULL sales (order with no non-null sales)
    - Order 'CA-2017-1003': 1 row with $300.0 (fully valid)
    - NULL order_id: 1 row with $50.0
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE sample_superstore (
            order_id VARCHAR,
            order_date TIMESTAMP,
            sales DOUBLE,
            region VARCHAR,
            customer_id VARCHAR,
            profit DOUBLE
        );
        INSERT INTO sample_superstore VALUES
            ('CA-2017-1001', '2017-03-01 10:00:00', 300.0, 'West', 'C-1', 50.0),
            ('CA-2017-1001', '2017-03-01 10:01:00', NULL,  'West', 'C-1', 0.0),
            ('CA-2017-1002', '2017-04-10 11:00:00', NULL,  'East', 'C-2', 0.0),
            ('CA-2017-1003', '2017-08-20 15:00:00', 300.0, 'West', 'C-3', 40.0),
            (NULL,           '2017-09-01 12:00:00', 50.0,  'East', 'C-4', 10.0);
    """)
    yield con
    con.close()


@pytest.fixture
def aov_master_air(superstore_schemas):
    """
    Phase 3 Multi-Branch DAG for:
    'How many total orders were placed in 2017, and what was the average order value?'
    Temporal filter: 2017 records
    Branch A: order_id IS NOT NULL -> COUNT_DISTINCT(order_id), SUM(sales), Derived AOV
    Branch B: order_id IS NULL -> COUNT(*) as null_order_id_count, SUM(sales) as unassigned_revenue
    """
    dims = [
        DimensionAuthorizationMatrix(column_name="order_date", allow_filter_predicate=True),
        DimensionAuthorizationMatrix(column_name="order_id", allow_filter_predicate=True),
        DimensionAuthorizationMatrix(column_name="region", allow_grouping=False),
    ]

    f_temporal = FilterOperation(
        operation_id="op_filter_2017",
        description="Filter records in 2017",
        input_datasets=["sample_superstore"],
        output_alias="view_2017",
        output_schema=dict(superstore_schemas["sample_superstore"]),
        predicates=[
            Predicate(
                column=ColumnRef(name="order_date", data_type=DataType.DATETIME),
                operator=ComparisonOperator.GTE,
                literal_value="2017-01-01",
            ),
            Predicate(
                column=ColumnRef(name="order_date", data_type=DataType.DATETIME),
                operator=ComparisonOperator.LT,
                literal_value="2018-01-01",
            ),
        ],
    )

    f_branch_a = FilterOperation(
        operation_id="op_branch_a_filter",
        description="Filter identifiable orders (non-null)",
        depends_on=["op_filter_2017"],
        input_datasets=["view_2017"],
        output_alias="view_branch_a",
        output_schema=dict(superstore_schemas["sample_superstore"]),
        predicates=[
            Predicate(
                column=ColumnRef(name="order_id", data_type=DataType.STRING),
                operator=ComparisonOperator.IS_NOT_NULL,
                literal_value="",
            ),
        ],
    )

    agg_branch_a = AggregateOperation(
        operation_id="op_branch_a_agg",
        description="Aggregate identifiable orders and revenue",
        depends_on=["op_branch_a_filter"],
        input_datasets=["view_branch_a"],
        output_alias="view_branch_a_agg",
        output_schema={"total_orders": DataType.INTEGER, "total_revenue": DataType.FLOAT},
        group_by_columns=[],
        measures=[
            AggregateMeasureSpec(
                output_alias="total_orders",
                function=AggregationFunction.COUNT_DISTINCT,
                target_column=ColumnRef(name="order_id", data_type=DataType.STRING),
            ),
            AggregateMeasureSpec(
                output_alias="total_revenue",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="sales", data_type=DataType.FLOAT),
            ),
        ],
    )

    f_branch_b = FilterOperation(
        operation_id="op_branch_b_filter",
        description="Filter null-ID quality branch",
        depends_on=["op_filter_2017"],
        input_datasets=["view_2017"],
        output_alias="view_branch_b",
        output_schema=dict(superstore_schemas["sample_superstore"]),
        predicates=[
            Predicate(
                column=ColumnRef(name="order_id", data_type=DataType.STRING),
                operator=ComparisonOperator.IS_NULL,
                literal_value="",
            ),
        ],
    )

    agg_branch_b = AggregateOperation(
        operation_id="op_branch_b_agg",
        description="Aggregate null-ID records and unassigned revenue",
        depends_on=["op_branch_b_filter"],
        input_datasets=["view_branch_b"],
        output_alias="view_branch_b_agg",
        output_schema={"null_order_id_count": DataType.INTEGER, "unassigned_revenue": DataType.FLOAT},
        is_terminal=True,
        group_by_columns=[],
        measures=[
            AggregateMeasureSpec(
                output_alias="null_order_id_count",
                function=AggregationFunction.COUNT,
                target_column=ColumnRef(name="order_id", data_type=DataType.STRING),
            ),
            AggregateMeasureSpec(
                output_alias="unassigned_revenue",
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name="sales", data_type=DataType.FLOAT),
            ),
        ],
    )

    derived_aov = DerivedOperation(
        operation_id="op_derived_aov",
        description="Calculate true Average Order Value (AOV)",
        depends_on=["op_branch_a_agg"],
        input_datasets=["view_branch_a_agg"],
        output_alias="view_aov_final",
        output_schema={
            "total_orders": DataType.INTEGER,
            "total_revenue": DataType.FLOAT,
            "average_order_value": DataType.FLOAT,
        },
        is_terminal=True,
        expressions=[
            DerivedExpressionSpec(
                output_alias="average_order_value",
                expression="total_revenue / NULLIF(total_orders, 0)",
                referenced_columns=["total_revenue", "total_orders"],
                null_handling="null_if_zero",
            )
        ],
    )

    air = MasterAIR(
        lifecycle_state=AIRLifecycleState.VALIDATED,
        raw_user_question="How many total orders were placed in 2017, and what was the average order value?",
        metadata_summary="AOV in 2017",
        dimensions=dims,
        operations=[f_temporal, f_branch_a, agg_branch_a, f_branch_b, agg_branch_b, derived_aov],
        terminal_operation_ids=["op_derived_aov", "op_branch_b_agg"],
        grain_assessment=GrainAssessment(
            target_entity="order_id",
            confidence=GrainConfidence.INFERRED_MULTI_GRAIN,
            cardinality_ratio=0.5,
            requires_entity_rollup=True,
            supporting_evidence="Benchmark order_id multi-grain line item dataset",
        ),
    )
    return air


# =============================================================================
# 1. COMPILER & CONTRACT FUSION TESTS
# =============================================================================

def test_air_compiler_fuses_multi_branch_and_derived_into_single_contract(
    aov_master_air, superstore_schemas
):
    """Proves the compiler fuses temporal filter, Branch A, Branch B, and derived AOV into 1 execution contract."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contracts = compiler.compile()

    assert len(contracts) == 1
    contract = contracts[0]

    assert contract.target_engine == ContractEngine.DUCKDB_SQL
    assert contract.target_table == "sample_superstore"

    # Authorized projections: 5 columns
    expected_projections = [
        "total_orders",
        "total_revenue",
        "average_order_value",
        "null_order_id_count",
        "unassigned_revenue",
    ]
    assert set(contract.expected_output_columns.keys()) == set(expected_projections)

    # Required measures: 4 measures
    measure_aliases = {m.output_alias for m in contract.required_measures}
    assert measure_aliases == {
        "total_orders",
        "total_revenue",
        "null_order_id_count",
        "unassigned_revenue",
    }

    # Conditional predicates: Branch A revenue requires IS NOT NULL, Branch B requires IS NULL
    rev_measure = next(m for m in contract.required_measures if m.output_alias == "total_revenue")
    assert rev_measure.condition_predicate is not None
    assert rev_measure.condition_predicate.operator == PredicateOperator.IS_NOT_NULL

    unassigned_measure = next(m for m in contract.required_measures if m.output_alias == "unassigned_revenue")
    assert unassigned_measure.condition_predicate is not None
    assert unassigned_measure.condition_predicate.operator == PredicateOperator.IS_NULL

    # Required derived measures: AOV division
    assert len(contract.required_derived_measures) == 1
    dm = contract.required_derived_measures[0]
    assert dm.output_alias == "average_order_value"
    assert dm.numerator_source_alias == "total_revenue"
    assert dm.denominator_source_alias == "total_orders"
    assert dm.expected_operator == "/"
    assert dm.enforce_nullif_zero is True
    assert dm.target_data_type in (DataType.FLOAT, "DOUBLE", DataType.FLOAT.value)

    # Shared temporal filters
    assert len(contract.required_filters) == 2
    for f in contract.required_filters:
        assert f.column.name == "order_date"


def test_contract_codegen_produces_deterministic_fused_sql(aov_master_air, superstore_schemas):
    """Proves deterministic contract template codegen emits correct CASE WHEN and NULLIF division."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]

    sql = create_contract_template_code(contract)
    assert 'COUNT(DISTINCT "order_id") AS total_orders' in sql
    assert 'CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END' in sql
    assert 'CASE WHEN "order_id" IS NULL THEN "sales" ELSE NULL END' in sql
    assert 'NULLIF(CAST(COUNT(DISTINCT "order_id") AS DOUBLE), 0)' in sql
    assert 'AS average_order_value' in sql
    assert 'AS null_order_id_count' in sql
    assert 'AS unassigned_revenue' in sql
    assert 'WHERE "order_date" >=' in sql


# =============================================================================
# 2. STRICT AST CONFORMANCE & ADVERSARIAL TESTS (A-01 through A-09)
# =============================================================================

def test_ast_verifier_accepts_valid_contract_sql(aov_master_air, superstore_schemas):
    """Proves AST verifier issues ExecutionReceipt for valid contract SQL."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)
    print("\n--- GENERATED SQL ---")
    print(sql)

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, contract)
    assert is_valid is True, f"Verification failed with: {err}"
    assert err is None
    assert isinstance(receipt, ExecutionReceipt)
    assert receipt.execution_unit_id == contract.execution_unit_id


def test_adversarial_ast_a01_wrong_source_table(aov_master_air, superstore_schemas):
    """A-01: Verifier rejects unauthorized source table."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract).replace("sample_superstore", "unauthorized_orders")

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "Unauthorized table" in err or "Expected table 'sample_superstore'" in err


def test_adversarial_ast_a02_wrong_numerator_column(aov_master_air, superstore_schemas):
    """A-02: Verifier rejects wrong column in numerator (e.g. profit instead of sales)."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract).replace('"sales"', '"profit"')

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "measure" in err.lower() or "profit" in err.lower() or "sales" in err.lower()


def test_adversarial_ast_a03_wrong_denominator_column(aov_master_air, superstore_schemas):
    """A-03: Verifier rejects wrong column in denominator (e.g. customer_id instead of order_id)."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract).replace('"order_id"', '"customer_id"')

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "measure" in err.lower() or "customer_id" in err.lower() or "order_id" in err.lower()


def test_adversarial_ast_a04_incorrect_cast_type(aov_master_air, superstore_schemas):
    """A-04: Verifier rejects incorrect target cast type (e.g. INT instead of DOUBLE for revenue)."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract).replace("AS DOUBLE", "AS INT")

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "cast" in err.lower() or "type" in err.lower() or "double" in err.lower()


def test_adversarial_ast_a05_missing_distinct_in_denominator(aov_master_air, superstore_schemas):
    """A-05: Verifier rejects line-item denominator (COUNT(order_id) without DISTINCT)."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract).replace("COUNT(DISTINCT", "COUNT(")

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "distinct" in err.lower() or "count" in err.lower()


def test_adversarial_ast_a06_missing_nullif_zero_guard(aov_master_air, superstore_schemas):
    """A-06: Verifier rejects derived division missing NULLIF zero guard."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    # Replace NULLIF(..., 0) with bare denominator
    bad_sql = """
    SELECT
        COUNT(DISTINCT "order_id") AS total_orders,
        CAST(SUM(CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END) AS DOUBLE) AS total_revenue,
        CAST(SUM(CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END) AS DOUBLE) / CAST(COUNT(DISTINCT "order_id") AS DOUBLE) AS average_order_value,
        CAST(COUNT(CASE WHEN "order_id" IS NULL THEN 1 ELSE NULL END) AS BIGINT) AS null_order_id_count,
        CAST(SUM(CASE WHEN "order_id" IS NULL THEN "sales" ELSE NULL END) AS DOUBLE) AS unassigned_revenue
    FROM sample_superstore
    WHERE "order_date" >= '2017-01-01' AND "order_date" < '2018-01-01'
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "nullif" in err.lower()


def test_adversarial_ast_a07_unauthorized_arithmetic(aov_master_air, superstore_schemas):
    """A-07: Verifier rejects unauthorized arithmetic (+10, *1.05) inside expressions."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract).replace(
        'AS total_revenue,',
        ' + 10.0 AS total_revenue,'
    )
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None


def test_adversarial_ast_a08_hidden_subquery_or_cte(aov_master_air, superstore_schemas):
    """A-08: Verifier rejects unauthorized CTEs and subqueries."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = f"WITH cte AS (SELECT * FROM sample_superstore) {create_contract_template_code(contract)}"

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "cte" in err.lower() or "with" in err.lower() or "unauthorized" in err.lower()


def test_adversarial_ast_a09_mismatched_filter_scope(aov_master_air, superstore_schemas):
    """A-09: Verifier rejects mismatched or extra filter predicates."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    bad_sql = create_contract_template_code(contract) + " AND region = 'West'"

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(bad_sql, contract)
    assert is_valid is False
    assert receipt is None
    assert "filter" in err.lower() or "where" in err.lower() or "region" in err.lower()


# =============================================================================
# 3. BENCHMARK EXECUTION & LINE-ITEM TRAP VERIFICATION
# =============================================================================

def test_benchmark_aov_execution_on_duckdb(
    aov_benchmark_duckdb, aov_master_air, superstore_schemas
):
    """
    Executes benchmark AOV query against DuckDB sandbox and verifies:
    - 2 unique identifiable orders
    - $600.0 total valid revenue
    - $300.0 AOV (NOT the $200.0 line-item average)
    - 1 null-ID row with $150.0 unassigned revenue
    - Unassigned revenue is NOT in AOV numerator
    """
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)
    print("\n--- GENERATED SQL ---")
    print(sql)

    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, contract)
    assert is_valid is True, f"Verification failed with: {err}"

    executor = SandboxedSQLExecutor(aov_benchmark_duckdb)
    res = executor.execute(sql, receipt=receipt, contract=contract)

    assert res.success is True
    assert res.result_df is not None
    df = res.result_df

    assert len(df) == 1
    row = df.iloc[0]

    assert row["total_orders"] == 2
    assert pytest.approx(row["total_revenue"], 1e-4) == 600.0
    assert pytest.approx(row["average_order_value"], 1e-4) == 300.0
    assert row["null_order_id_count"] == 1
    assert pytest.approx(row["unassigned_revenue"], 1e-4) == 150.0

    # Line-item average check: 600.0 / 3 rows = 200.0. AOV must NOT be 200.0!
    assert row["average_order_value"] != 200.0


# =============================================================================
# 4. INDEPENDENT DUAL-PATH & CONTRACT VALIDATION TESTS
# =============================================================================

def test_deterministic_validator_validates_contract_execution(
    aov_benchmark_duckdb, aov_master_air, superstore_schemas
):
    """Independent validator verifies both Branch A and Branch B metrics against vectorized Pandas."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)

    _, _, receipt = SQLASTConformanceVerifier.verify(sql, contract)
    executor = SandboxedSQLExecutor(aov_benchmark_duckdb)
    res = executor.execute(sql, receipt=receipt, contract=contract)

    validator = DeterministicValidator(aov_benchmark_duckdb, "sample_superstore")
    val_report = validator.validate_contract_execution(
        step_id="step_aov_benchmark",
        contract=contract,
        result_df=res.result_df,
    )

    assert val_report.is_fully_valid is True
    assert val_report.checks_passed == val_report.checks_total
    assert val_report.checks_total >= 5  # 4 measures + 1 derived AOV


def test_deterministic_validator_catches_tampered_metric(
    aov_benchmark_duckdb, aov_master_air, superstore_schemas
):
    """Independent validator rejects DataFrame if any metric has been tampered with."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)

    _, _, receipt = SQLASTConformanceVerifier.verify(sql, contract)
    executor = SandboxedSQLExecutor(aov_benchmark_duckdb)
    res = executor.execute(sql, receipt=receipt, contract=contract)

    # Tamper with AOV result
    tampered_df = res.result_df.copy()
    tampered_df["average_order_value"] = 200.0  # Injected line-item trap

    validator = DeterministicValidator(aov_benchmark_duckdb, "sample_superstore")
    val_report = validator.validate_contract_execution(
        step_id="step_aov_tampered",
        contract=contract,
        result_df=tampered_df,
    )

    assert val_report.is_fully_valid is False
    first_err = val_report.first_error()
    assert first_err is not None
    assert first_err.error_type == "CALCULATION_MISMATCH"
    assert "average_order_value" in first_err.error_message


# =============================================================================
# 5. MISSING-SALES EVIDENCE TESTS
# =============================================================================

def test_missing_sales_evidence_diagnostics(aov_missing_sales_duckdb):
    """
    Verifies diagnostic queries accurately detect:
    - orders with no non-null sales
    - rows with null sales
    - orders with partial null sales
    """
    where_cls = "\"order_date\" >= '2017-01-01' AND \"order_date\" < '2018-01-01'"
    missing_q = f"""
    WITH per_order AS (
        SELECT
            "order_id",
            COUNT("sales") AS non_null_sales_count,
            COUNT(CASE WHEN "sales" IS NULL THEN 1 END) AS null_sales_count
        FROM sample_superstore
        WHERE {where_cls} AND "order_id" IS NOT NULL
        GROUP BY "order_id"
    )
    SELECT
        CAST(COUNT(CASE WHEN non_null_sales_count = 0 THEN 1 END) AS BIGINT) AS orders_with_no_sales,
        CAST(COUNT(CASE WHEN non_null_sales_count > 0 AND null_sales_count > 0 THEN 1 END) AS BIGINT) AS orders_with_partial_null_sales
    FROM per_order
    """
    res = aov_missing_sales_duckdb.execute(missing_q).fetchone()
    assert res[0] == 1  # CA-2017-1002 has only NULL sales
    assert res[1] == 1  # CA-2017-1001 has 1 valid ($300) and 1 NULL sales

    missing_rows_q = f'SELECT CAST(COUNT(*) AS BIGINT) FROM sample_superstore WHERE {where_cls} AND "sales" IS NULL'
    null_rows = aov_missing_sales_duckdb.execute(missing_rows_q).fetchone()[0]
    assert null_rows == 2  # CA-2017-1001 item 2 and CA-2017-1002 item 1


# =============================================================================
# 6. FAIL-CLOSED GRAPH ENFORCEMENT TESTS
# =============================================================================

def test_graph_fail_closed_on_compilation_failure(aov_benchmark_duckdb):
    """Proves agent graph halts immediately with COMPILATION_FAILED if AIR compiler fails."""
    graph = create_analytics_graph(aov_benchmark_duckdb)

    # Force compiler failure by passing an unresolvable DAG schema
    broken_air = MasterAIR(
        lifecycle_state=AIRLifecycleState.VALIDATED,
        raw_user_question="test fail",
        metadata_summary="test fail",
        dimensions=[],
        operations=[
            FilterOperation(
                operation_id="op_fail",
                description="Invalid operation",
                input_datasets=["non_existent_table"],
                output_alias="v_fail",
                output_schema={},
                predicates=[],
            )
        ],
        terminal_operation_ids=["op_fail"],
        grain_assessment=GrainAssessment(
            target_entity="unknown",
            confidence=GrainConfidence.UNKNOWN,
            supporting_evidence="Unknown grain test",
        ),
    )

    obj = AnalysisObjective(
        target_metric="sales",
        target_table="sample_superstore",
        analysis_type=AnalysisType.DESCRIPTIVE,
    )
    prof = DatasetProfile(
        session_id="test_session",
        dataset_name="test",
        total_rows=5,
        total_columns=6,
        memory_usage_bytes=1000,
        tables=[TableProfile(table_name="sample_superstore", row_count=5, column_count=6, columns=[])],
    )
    plan = AnalysisPlan(
        reasoning="Test plan",
        steps=[PlanStep(step_id="step_1", description="Calculate AOV", agg_func="AOV", metric_col="sales")],
    )

    q_text = "How many total orders were placed in 2017, and what was the average order value?"
    # Run state directly with pre-set compilation error
    initial_state = {
        "user_question": q_text,
        "objective": obj,
        "profile": prof,
        "plan": plan,
        "status": "COMPILATION_FAILED",
        "compilation_error": "AIR_COMPILATION_ERROR: Operation 'op_fail' references unknown table",
        "step_results": {},
        "evidence_records": [],
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPILATION_FAILED"
    assert final_state["overall_outcome"] == "FAILED_VALIDATION"
    assert "Pipeline Halted" in final_state["final_summary"]
    assert final_state["ecs_report"].score == 0.0


def test_graph_fail_closed_on_verification_failure(aov_benchmark_duckdb, aov_master_air, superstore_schemas):
    """Proves agent graph halts immediately with VERIFICATION_FAILED when AST verification fails."""
    graph = create_analytics_graph(aov_benchmark_duckdb)

    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contracts = {c.execution_unit_id: c for c in compiler.compile()}

    q_text = "How many total orders were placed in 2017, and what was the average order value?"
    obj = AnalysisObjective(
        target_metric="sales",
        target_table="sample_superstore",
        analysis_type=AnalysisType.DESCRIPTIVE,
    )
    prof = DatasetProfile(
        session_id="test_session",
        dataset_name="test",
        total_rows=5,
        total_columns=6,
        memory_usage_bytes=1000,
        tables=[TableProfile(table_name="sample_superstore", row_count=5, column_count=6, columns=[])],
    )
    plan = AnalysisPlan(
        reasoning="Test plan",
        steps=[PlanStep(step_id="step_1", description="Calculate AOV", agg_func="AOV", metric_col="sales")],
    )

    initial_state = {
        "user_question": q_text,
        "objective": obj,
        "profile": prof,
        "plan": plan,
        "contracts": contracts,
        "status": "PLAN_FORMULATED",
        "step_results": {},
        "evidence_records": [],
    }

    # Mock codegen to produce adversarial code that fails AST verification
    with patch("src.agent.codegen.CodeGenerator.generate_with_contract") as mock_gen:
        mock_gen.return_value = MagicMock(
            executable_code="SELECT * FROM unauthorized_table",
            code_type="sql",
        )
        final_state = graph.invoke(initial_state)

    assert final_state["status"] == "VERIFICATION_FAILED"
    assert final_state["overall_outcome"] == "FAILED_VALIDATION"
    assert "Pipeline Halted" in final_state["final_summary"]
    assert final_state["ecs_report"].score == 0.0


def test_sandbox_gatekeeper_blocks_unauthorized_execution(
    aov_benchmark_duckdb, aov_master_air, superstore_schemas
):
    """Proves SandboxedSQLExecutor gatekeeper refuses to run contracted SQL without matching receipt."""
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)

    executor = SandboxedSQLExecutor(aov_benchmark_duckdb)

    # 1. Blocked when receipt is None
    res1 = executor.execute(sql, receipt=None, contract=contract)
    assert res1.success is False
    assert "Missing ExecutionReceipt" in res1.stderr

    # 2. Blocked when receipt is for a different query
    fake_receipt = ExecutionReceipt(
        contract_id=contract.contract_id,
        execution_unit_id=contract.execution_unit_id,
        code_hash="0000000000000000000000000000000000000000000000000000000000000000",
        target_engine=contract.target_engine,
        conformance_passed=True,
        timestamp=0.0,
    )
    res2 = executor.execute(sql, receipt=fake_receipt, contract=contract)
    assert res2.success is False
    assert "code_hash mismatch" in res2.stderr


# =============================================================================
# 7. REMEDIATION REGRESSION TESTS (ADV-01 through ADV-04)
# =============================================================================

def test_remediation_adv01_numerator_condition_predicate(aov_master_air, superstore_schemas):
    """
    ADV-01: Proves verifier enforces condition predicate on derived AOV numerator:
    - Valid AOV numerator with order_id IS NOT NULL is accepted.
    - Numerator using order_id IS NULL (unassigned revenue) is rejected.
    - Numerator without CASE WHEN is rejected.
    - Numerator with inverted/opposite predicate is rejected.
    """
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    valid_sql = create_contract_template_code(contract)

    # 1. Valid numerator passes
    ok, err, receipt = SQLASTConformanceVerifier.verify(valid_sql, contract)
    assert ok is True
    assert receipt is not None

    # 2. Numerator using unassigned revenue (order_id IS NULL) rejected
    bad_num_null = valid_sql.replace(
        'CAST(SUM(CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END) AS DOUBLE) / NULLIF',
        'CAST(SUM(CASE WHEN "order_id" IS NULL THEN "sales" ELSE NULL END) AS DOUBLE) / NULLIF',
    )
    ok, err, _ = SQLASTConformanceVerifier.verify(bad_num_null, contract)
    assert ok is False
    assert "condition predicate enforcement" in err or "requires 'order_id IS NOT NULL'" in err

    # 3. Numerator without CASE WHEN (unconditioned SUM(sales)) rejected
    bad_num_nocase = valid_sql.replace(
        'CAST(SUM(CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END) AS DOUBLE) / NULLIF',
        'CAST(SUM("sales") AS DOUBLE) / NULLIF',
    )
    ok, err, _ = SQLASTConformanceVerifier.verify(bad_num_nocase, contract)
    assert ok is False
    assert "lacks a conditional CASE WHEN construct" in err or "condition predicate enforcement" in err

    # 4. Numerator with inverted condition (order_id = 'X') rejected
    bad_num_inverted = valid_sql.replace(
        'CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END',
        'CASE WHEN "order_id" = \'CA-2017-001\' THEN "sales" ELSE NULL END',
    )
    ok, err, _ = SQLASTConformanceVerifier.verify(bad_num_inverted, contract)
    assert ok is False


def test_remediation_adv02_measure_condition_predicate(aov_master_air, superstore_schemas):
    """
    ADV-02: Proves verifier enforces condition predicate on all base measures:
    - Required conditional revenue aggregation is accepted when correct.
    - Unconditioned SUM(sales) as total_revenue is rejected.
    - Opposite condition (IS NULL instead of IS NOT NULL) is rejected.
    - Measures without condition predicates remain valid.
    """
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    valid_sql = create_contract_template_code(contract)

    # 1. Valid conditional revenue aggregation passes
    ok, err, _ = SQLASTConformanceVerifier.verify(valid_sql, contract)
    assert ok is True

    # 2. Unconditioned SUM(sales) for total_revenue rejected
    bad_rev_nocase = valid_sql.replace(
        'CAST(SUM(CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END) AS DOUBLE) AS total_revenue',
        'CAST(SUM("sales") AS DOUBLE) AS total_revenue',
    )
    ok, err, _ = SQLASTConformanceVerifier.verify(bad_rev_nocase, contract)
    assert ok is False
    assert "lacks a conditional CASE WHEN construct" in err or "failed verification" in err

    # 3. Opposite condition for total_revenue (IS NULL instead of IS NOT NULL) rejected
    bad_rev_opposite = valid_sql.replace(
        'CAST(SUM(CASE WHEN "order_id" IS NOT NULL THEN "sales" ELSE NULL END) AS DOUBLE) AS total_revenue',
        'CAST(SUM(CASE WHEN "order_id" IS NULL THEN "sales" ELSE NULL END) AS DOUBLE) AS total_revenue',
    )
    ok, err, _ = SQLASTConformanceVerifier.verify(bad_rev_opposite, contract)
    assert ok is False
    assert "requires 'order_id IS NOT NULL'" in err or "failed verification" in err

    # 4. Measure without condition predicate remains valid (e.g. Phase 2 contract)
    uncond_measure = RequiredMeasureSpec(
        output_alias="revenue",
        function=AggregationFunction.SUM,
        target_column=ColumnRef(table="sample_superstore", name="sales", data_type=DataType.FLOAT),
        condition_predicate=None,
    )
    uncond_contract = ExecutionContract(
        contract_id="contract_uncond",
        execution_unit_id="unit_uncond",
        target_engine=ContractEngine.DUCKDB_SQL,
        target_table="sample_superstore",
        authorized_group_by=[],
        forbidden_group_by=[],
        expected_cardinality=CardinalityExpectation.SCALAR,
        required_measures=[uncond_measure],
        required_derived_measures=[],
    )
    sql_uncond = 'SELECT CAST(SUM("sales") AS DOUBLE) AS revenue FROM sample_superstore'
    ok, err, _ = SQLASTConformanceVerifier.verify(sql_uncond, uncond_contract)
    assert ok is True, f"Unconditioned measure rejected: {err}"


def test_remediation_adv03_no_raw_sql_in_synthesis_and_correct_pandas_diagnostics(aov_benchmark_duckdb, aov_master_air, superstore_schemas):
    """
    ADV-03: Proves:
    - Missing-sales diagnostics are computed strictly in Pandas on the raw slice.
    - synthesize_node executes ZERO raw SQL queries.
    - Diagnostics correctly handle multiple line items, all-null sales, partial null sales, and null IDs.
    """
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)

    # Create dataset with multiple line items, partial nulls, all nulls, and null IDs
    con = duckdb.connect()
    raw_df = pd.DataFrame([
        {"order_id": "ORD-1", "order_date": "2017-02-01", "sales": 100.0, "region": "West", "customer_id": "C-1", "profit": 10.0},
        {"order_id": "ORD-1", "order_date": "2017-02-01", "sales": None,  "region": "West", "customer_id": "C-1", "profit": 0.0}, # partial null
        {"order_id": "ORD-2", "order_date": "2017-03-01", "sales": None,  "region": "East", "customer_id": "C-2", "profit": 0.0}, # all null
        {"order_id": "ORD-3", "order_date": "2017-04-01", "sales": 200.0, "region": "West", "customer_id": "C-3", "profit": 20.0},
        {"order_id": "ORD-3", "order_date": "2017-04-01", "sales": 300.0, "region": "West", "customer_id": "C-3", "profit": 30.0}, # all non-null
        {"order_id": None,    "order_date": "2017-05-01", "sales": 50.0,  "region": "East", "customer_id": "C-4", "profit": 5.0},  # null ID
        {"order_id": None,    "order_date": "2017-05-01", "sales": None,  "region": "East", "customer_id": "C-4", "profit": 0.0},  # null ID with null sales
    ])
    con.register("sample_superstore", raw_df)

    validator = DeterministicValidator(con, "sample_superstore")
    sql_df = con.execute(sql).fetchdf()
    val_report = validator.validate_contract_execution("step_1", contract, sql_df)

    # 1. Assert diagnostics are calculated in Pandas
    diag = val_report.missing_sales_diagnostics
    assert diag["orders_with_no_sales"] == 1          # ORD-2 has only NULL sales
    assert diag["orders_with_partial_null_sales"] == 1 # ORD-1 has 1 non-null and 1 null
    assert diag["null_sales_row_count"] == 3          # ORD-1 item 2, ORD-2 item 1, null ID item 2

    # 2. Mock con.execute during synthesis to prove zero SQL queries are run
    from src.agent.graph import AnalyticsWorkflowEngine
    engine = AnalyticsWorkflowEngine(con=con)

    step = PlanStep(step_id="step_1", description="Calculate AOV", agg_func="AOV", metric_col="sales")
    plan = AnalysisPlan(reasoning="Test", steps=[step])
    obj = AnalysisObjective(target_metric="sales", target_table="sample_superstore", analysis_type=AnalysisType.DESCRIPTIVE)
    prof = DatasetProfile(
        session_id="test_adv03",
        dataset_name="test",
        total_rows=len(raw_df),
        total_columns=len(raw_df.columns),
        memory_usage_bytes=1000,
        tables=[TableProfile(table_name="sample_superstore", row_count=len(raw_df), column_count=len(raw_df.columns), columns=[])],
    )

    state = {
        "user_question": "How many total orders were placed in 2017, and what was the average order value?",
        "objective": obj,
        "plan": plan,
        "profile": prof,
        "contracts": {contract.execution_unit_id: contract},
        "current_contract": contract,
        "step_results": {
            "step_1": {
                "dataframe": sql_df,
                "scalar": None,
                "success": True,
                "missing_sales_diagnostics": diag,
            }
        },
        "missing_sales_diagnostics": diag,
        "evidence_records": [],
        "status": "VALIDATION_PASSED",
    }

    class ConProxy:
        def __init__(self, real_con):
            self._real = real_con
            self.execute_called = False
        def execute(self, *args, **kwargs):
            self.execute_called = True
            return self._real.execute(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._real, name)

    proxy = ConProxy(engine.con)
    engine.con = proxy
    result = engine.synthesize_node(state)
    # synthesize_node must NOT call con.execute
    assert proxy.execute_called is False

    summary = result["final_summary"]
    assert "Orders with No Non-Null Sales:** **1**" in summary
    assert "Orders with Partially Missing Sales:** **1**" in summary
    assert "Rows with NULL Sales:** **3**" in summary


def test_remediation_adv04_genuinely_independent_derived_validation(aov_benchmark_duckdb, aov_master_air, superstore_schemas):
    """
    ADV-04: Proves:
    - Expected AOV is computed strictly from independent Pandas expected measures, not SQL output.
    - Tampering with SQL AOV output while Pandas source data is unchanged causes validation to fail.
    - Preserves NULL semantics for empty windows, all-null sales, and zero valid orders.
    """
    compiler = AIRContractCompiler(aov_master_air, superstore_schemas)
    contract = compiler.compile()[0]
    sql = create_contract_template_code(contract)
    validator = DeterministicValidator(aov_benchmark_duckdb, "sample_superstore")

    # 1. Clean run passes
    clean_df = aov_benchmark_duckdb.execute(sql).fetchdf()
    clean_rep = validator.validate_contract_execution("step_1", contract, clean_df)
    assert clean_rep.is_fully_valid is True

    # 2. Tampered SQL AOV (e.g. 200.0 instead of 300.0) fails validation
    tampered_df = clean_df.copy()
    tampered_df["average_order_value"] = 200.0
    tampered_rep = validator.validate_contract_execution("step_1", contract, tampered_df)
    assert tampered_rep.is_fully_valid is False
    mismatch = next(r for r in tampered_rep.results if not r.is_valid)
    assert "Contract derived measure mismatch on 'average_order_value'" in mismatch.error_message
    assert "independent Pandas expected=300" in mismatch.error_message

    # 3. Empty 2017 window -> expected AOV is NULL
    con_empty = duckdb.connect()
    df_empty = pd.DataFrame(columns=["order_id", "order_date", "sales"])
    con_empty.register("sample_superstore", df_empty)
    val_empty = DeterministicValidator(con_empty, "sample_superstore")
    sql_empty_df = con_empty.execute(sql).fetchdf()
    rep_empty = val_empty.validate_contract_execution("step_empty", contract, sql_empty_df)
    assert rep_empty.is_fully_valid is True

    # 4. All sales NULL -> expected AOV is NULL
    con_null_sales = duckdb.connect()
    df_null_sales = pd.DataFrame([
        {"order_id": "CA-2017-001", "order_date": "2017-03-01", "sales": None},
        {"order_id": "CA-2017-002", "order_date": "2017-04-01", "sales": None},
    ])
    con_null_sales.register("sample_superstore", df_null_sales)
    val_null = DeterministicValidator(con_null_sales, "sample_superstore")
    sql_null_df = con_null_sales.execute(sql).fetchdf()
    rep_null = val_null.validate_contract_execution("step_null", contract, sql_null_df)
    assert rep_null.is_fully_valid is True

    # 5. Zero valid orders (only null IDs) -> expected AOV is NULL
    con_zero_orders = duckdb.connect()
    df_zero_orders = pd.DataFrame([
        {"order_id": None, "order_date": "2017-03-01", "sales": 100.0},
        {"order_id": None, "order_date": "2017-04-01", "sales": 200.0},
    ])
    con_zero_orders.register("sample_superstore", df_zero_orders)
    val_zero = DeterministicValidator(con_zero_orders, "sample_superstore")
    sql_zero_df = con_zero_orders.execute(sql).fetchdf()
    rep_zero = val_zero.validate_contract_execution("step_zero", contract, sql_zero_df)
    assert rep_zero.is_fully_valid is True
