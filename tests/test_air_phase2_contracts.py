"""
Unit and End-to-End Tests for AIR Phase 2:
Execution Contracts, Deterministic Compilation, Scope-Aware SQLGlot Conformance,
ExecutionReceipt Gatekeeping, and Sandbox Enforcement.
"""

from __future__ import annotations

import duckdb
import pytest
from unittest.mock import MagicMock, patch

from src.agent.air.schemas import (
    AIRLifecycleState,
    ColumnRef,
    ComparisonOperator,
    DataType,
    DimensionAuthorizationMatrix,
    GrainAssessment,
    Predicate,
)
from src.agent.air.operations import (
    AggregateMeasureSpec,
    AggregateOperation,
    AggregationFunction,
    FilterOperation,
)
from src.agent.air.dag import MasterAIR
from src.agent.air.contract import (
    CardinalityExpectation,
    ContractEngine,
    ExecutionContract,
    ExecutionReceipt,
    PredicateOperator,
    RequiredMeasureSpec,
    TypedPredicate,
)
from src.agent.air.compiler import AIRContractCompiler, CompilerError
from src.agent.air.sql_verifier import SQLASTConformanceVerifier
from src.agent.schemas import AnalysisObjective, AnalysisPlan, AnalysisType, DatasetProfile, PlanStep
from src.agent.graph import create_analytics_graph
from src.agent.state import AnalyticsAgentState
from src.execution.sandbox import SandboxedSQLExecutor


@pytest.fixture
def base_superstore_schemas():
    return {
        "sample_superstore": {
            "order_id": DataType.STRING,
            "order_date": DataType.DATETIME,
            "sales": DataType.FLOAT,
            "region": DataType.STRING,
            "ship_mode": DataType.STRING,
            "segment": DataType.STRING,
            "product_name": DataType.STRING,
        }
    }


@pytest.fixture
def sample_duckdb():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE sample_superstore (
            order_id VARCHAR,
            order_date TIMESTAMP,
            sales DOUBLE,
            region VARCHAR,
            ship_mode VARCHAR,
            segment VARCHAR,
            product_name VARCHAR
        );
        INSERT INTO sample_superstore VALUES
            ('CA-2017-1001', '2017-01-15 10:00:00', 250.0, 'West', 'Standard Class', 'Consumer', 'Product A'),
            ('CA-2017-1002', '2017-06-20 14:30:00', 400.0, 'East', 'Second Class', 'Corporate', 'Product B'),
            ('CA-2017-1003', '2017-12-31 23:45:00', 150.0, 'West', 'First Class', 'Consumer', 'Product C'),
            ('CA-2018-1004', '2018-01-01 09:00:00', 500.0, 'West', 'Standard Class', 'Consumer', 'Product D'),
            ('CA-2016-1005', '2016-11-10 12:00:00', 300.0, 'Central', 'Same Day', 'Home Office', 'Product E');
    """)
    yield con
    con.close()


@pytest.fixture
def vertical_slice_air(base_superstore_schemas):
    """AIR for 'Show total revenue by region in 2017'"""
    dims = [
        DimensionAuthorizationMatrix(column_name="region", allow_grouping=True, is_requested_in_final_output=True),
        DimensionAuthorizationMatrix(column_name="order_date", allow_filter_predicate=True),
        DimensionAuthorizationMatrix(column_name="ship_mode", allow_grouping=False),
        DimensionAuthorizationMatrix(column_name="segment", allow_grouping=False),
    ]
    f_op = FilterOperation(
        operation_id="op_filter_2017",
        description="Filter orders in 2017",
        input_datasets=["sample_superstore"],
        output_alias="view_2017",
        output_schema=dict(base_superstore_schemas["sample_superstore"]),
        predicates=[
            Predicate(column=ColumnRef(name="order_date", data_type=DataType.DATETIME), operator=ComparisonOperator.GTE, literal_value="2017-01-01"),
            Predicate(column=ColumnRef(name="order_date", data_type=DataType.DATETIME), operator=ComparisonOperator.LT, literal_value="2018-01-01"),
        ],
        is_terminal=False,
    )
    agg_op = AggregateOperation(
        operation_id="op_agg_revenue_by_region",
        description="Aggregate revenue by region",
        depends_on=["op_filter_2017"],
        input_datasets=["view_2017"],
        output_alias="view_revenue_by_region",
        output_schema={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", data_type=DataType.STRING)],
        measures=[
            AggregateMeasureSpec(output_alias="total_revenue", function=AggregationFunction.SUM, target_column=ColumnRef(name="sales", data_type=DataType.FLOAT))
        ],
        is_terminal=True,
    )
    return MasterAIR(
        lifecycle_state=AIRLifecycleState.VALIDATED,
        raw_user_question="Show total revenue by region in 2017.",
        metadata_summary="Revenue by region in 2017",
        grain_assessment=GrainAssessment(target_entity="order_item", supporting_evidence="line-item"),
        dimensions=dims,
        operations=[f_op, agg_op],
        terminal_operation_ids=["op_agg_revenue_by_region"],
        requested_outputs=["region", "total_revenue"],
    )


# ===========================================================================
# 1. Contract & Compiler Tests
# ===========================================================================

def test_compiler_authorizes_only_requested_dimension(vertical_slice_air, base_superstore_schemas):
    compiler = AIRContractCompiler(vertical_slice_air, base_superstore_schemas)
    contracts = compiler.compile()

    assert len(contracts) == 1
    c = contracts[0]
    assert c.authorized_group_by == ["region"]
    assert "ship_mode" in c.forbidden_group_by
    assert "segment" in c.forbidden_group_by
    assert c.target_table == "sample_superstore"
    assert c.expected_cardinality == CardinalityExpectation.DIMENSIONAL


def test_compiler_preserves_filter_to_aggregate_dependencies(vertical_slice_air, base_superstore_schemas):
    compiler = AIRContractCompiler(vertical_slice_air, base_superstore_schemas)
    contracts = compiler.compile()
    c = contracts[0]

    assert "op_filter_2017" in c.air_operation_ids
    assert "op_agg_revenue_by_region" in c.air_operation_ids
    assert len(c.required_filters) == 2
    assert c.required_filters[0].operator == PredicateOperator.GTE
    assert c.required_filters[0].literal_value == "2017-01-01"
    assert c.required_filters[1].operator == PredicateOperator.LT
    assert c.required_filters[1].literal_value == "2018-01-01"


def test_compiler_unsupported_operation_fails_explicitly(base_superstore_schemas):
    from src.agent.air.operations import RankOperation, SortCriterion
    rank_op = RankOperation(
        operation_id="op_rank",
        description="Rank",
        input_datasets=["sample_superstore"],
        output_alias="view_rank",
        output_schema=dict(base_superstore_schemas["sample_superstore"]),
        order_by=[
            SortCriterion(column=ColumnRef(name="sales", data_type=DataType.FLOAT), is_metric=True)
        ],
        is_terminal=True,
    )
    air = MasterAIR(
        lifecycle_state=AIRLifecycleState.VALIDATED,
        raw_user_question="test",
        metadata_summary="test",
        grain_assessment=GrainAssessment(target_entity="sales", supporting_evidence="test"),
        operations=[rank_op],
        terminal_operation_ids=["op_rank"],
    )
    compiler = AIRContractCompiler(air, base_superstore_schemas)
    with pytest.raises(CompilerError, match="Unsupported operation type"):
        compiler.compile()


# ===========================================================================
# 2. SQL AST Conformance Verifier Tests
# ===========================================================================

@pytest.fixture
def revenue_by_region_contract():
    return ExecutionContract(
        contract_id="contract_step_1",
        execution_unit_id="unit_step_1",
        air_operation_ids=["op_filter_2017", "op_agg_revenue_by_region"],
        target_table="sample_superstore",
        authorized_source_columns=["order_id", "order_date", "sales", "region", "ship_mode", "segment"],
        authorized_group_by=["region"],
        forbidden_group_by=["ship_mode", "segment"],
        required_filters=[
            TypedPredicate(column=ColumnRef(name="order_date", data_type=DataType.DATETIME), operator=PredicateOperator.GTE, literal_value="2017-01-01", is_temporal=True),
            TypedPredicate(column=ColumnRef(name="order_date", data_type=DataType.DATETIME), operator=PredicateOperator.LT, literal_value="2018-01-01", is_temporal=True),
        ],
        required_measures=[
            RequiredMeasureSpec(output_alias="total_revenue", function=AggregationFunction.SUM, target_column=ColumnRef(name="sales", data_type=DataType.FLOAT))
        ],
        expected_output_columns={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        expected_cardinality=CardinalityExpectation.DIMENSIONAL,
        target_engine=ContractEngine.DUCKDB_SQL,
    )


def test_sql_verifier_valid_query_passes(revenue_by_region_contract):
    sql = """
    SELECT
        region,
        CAST(SUM(sales) AS DOUBLE) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    ORDER BY total_revenue DESC
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is True
    assert err is None
    assert receipt is not None
    assert receipt.conformance_passed is True
    assert receipt.contract_id == revenue_by_region_contract.contract_id


def test_sql_verifier_rejects_unauthorized_grouping_ship_mode(revenue_by_region_contract):
    sql = """
    SELECT region, ship_mode, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region, ship_mode
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions detected" in err
    assert "ship_mode" in err
    assert receipt is None


def test_sql_verifier_rejects_unauthorized_grouping_segment(revenue_by_region_contract):
    sql = """
    SELECT region, segment, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region, segment
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions detected" in err
    assert "segment" in err


def test_sql_verifier_rejects_ship_mode_alone(revenue_by_region_contract):
    sql = """
    SELECT ship_mode, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY ship_mode
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions detected" in err


def test_sql_verifier_rejects_missing_date_lower_bound(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "No top-level conjunct in WHERE clause soundly enforces 'order_date >= '2017-01-01''" in err


def test_sql_verifier_rejects_missing_date_upper_bound(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "No top-level conjunct in WHERE clause soundly enforces 'order_date < '2018-01-01''" in err


def test_sql_verifier_rejects_inclusive_upper_bound(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date <= '2017-12-31'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Upper boundary uses inclusive (<= '2017-12-31'), violating strict half-open interval" in err


def test_sql_verifier_rejects_weakened_or_condition(revenue_by_region_contract):
    # Required filter placed inside OR branch
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE (order_date >= '2017-01-01' OR region = 'West') AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "weakened OR condition" in err


def test_sql_verifier_rejects_missing_measure_aggregation(revenue_by_region_contract):
    sql = """
    SELECT region, AVG(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Required measure 'sum(sales)' (alias: 'total_revenue') was not found in query" in err


def test_sql_verifier_rejects_wrong_table_binding(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM other_retail_table
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Table binding mismatch" in err


def test_sql_verifier_parenthesized_boolean_expressions_pass(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE ((order_date >= '2017-01-01') AND (order_date < '2018-01-01'))
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is True
    assert receipt is not None


def test_sql_verifier_rejects_predicate_placement_in_unrelated_subquery(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE sales > (SELECT AVG(sales) FROM sample_superstore WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01')
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Predicate enforcement failure" in err


def test_sql_verifier_malformed_sql_rejected(revenue_by_region_contract):
    sql = "SELECT region, SUM(sales FROM sample_superstore GROUP BY"
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Parse Error" in err


def test_sql_verifier_non_select_rejected(revenue_by_region_contract):
    sql = "DROP TABLE sample_superstore"
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Expected root SELECT statement" in err


def test_sql_verifier_scope_isolation_inner_cte_grouping():
    """Inner CTE grouping by order_id must not bleed into root scalar contract."""
    scalar_contract = ExecutionContract(
        contract_id="contract_scalar",
        execution_unit_id="unit_scalar",
        target_table="sample_superstore",
        authorized_group_by=[],
        forbidden_group_by=["order_id", "region"],
        required_measures=[
            RequiredMeasureSpec(output_alias="aov", function=AggregationFunction.AVG, target_column=ColumnRef(name="order_total", data_type=DataType.FLOAT))
        ],
        expected_output_columns={"aov": DataType.FLOAT},
        expected_cardinality=CardinalityExpectation.SCALAR,
    )
    sql = """
    SELECT AVG(order_total) AS aov
    FROM (
        SELECT order_id, SUM(sales) AS order_total
        FROM sample_superstore
        GROUP BY order_id
    ) AS sub
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, scalar_contract)
    assert is_valid is True
    assert receipt is not None


# ===========================================================================
# 3. ExecutionReceipt and Sandbox Gatekeeper Tests
# ===========================================================================

def test_sandbox_gatekeeper_valid_receipt_executes(sample_duckdb, revenue_by_region_contract):
    sql = """
    SELECT region, CAST(SUM(sales) AS DOUBLE) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    ORDER BY total_revenue DESC
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is True

    executor = SandboxedSQLExecutor(sample_duckdb)
    res = executor.execute(sql, receipt=receipt, contract=revenue_by_region_contract)

    assert res.success is True
    assert res.result_df is not None
    assert len(res.result_df) == 2  # West ($400: $250+$150) and East ($400)


def test_sandbox_gatekeeper_rejects_modified_sql(sample_duckdb, revenue_by_region_contract):
    valid_sql = """
    SELECT region, CAST(SUM(sales) AS DOUBLE) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, _, receipt = SQLASTConformanceVerifier.verify(valid_sql, revenue_by_region_contract)
    assert is_valid is True

    tampered_sql = valid_sql + "\n-- tampered comment"
    executor = SandboxedSQLExecutor(sample_duckdb)
    res = executor.execute(tampered_sql, receipt=receipt, contract=revenue_by_region_contract)

    assert res.success is False
    assert "Sandbox Gatekeeper" in res.stderr
    assert "code payload was altered after verification" in res.stderr


def test_sandbox_gatekeeper_rejects_missing_receipt(sample_duckdb, revenue_by_region_contract):
    sql = "SELECT region, SUM(sales) FROM sample_superstore GROUP BY region"
    executor = SandboxedSQLExecutor(sample_duckdb)
    res = executor.execute(sql, receipt=None, contract=revenue_by_region_contract)

    assert res.success is False
    assert "Missing ExecutionReceipt" in res.stderr


def test_sandbox_gatekeeper_rejects_wrong_contract_id(sample_duckdb, revenue_by_region_contract):
    valid_sql = """
    SELECT region, CAST(SUM(sales) AS DOUBLE) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    _, _, receipt = SQLASTConformanceVerifier.verify(valid_sql, revenue_by_region_contract)

    mismatched_contract = revenue_by_region_contract.model_copy(update={"contract_id": "contract_step_99"})
    executor = SandboxedSQLExecutor(sample_duckdb)
    res = executor.execute(valid_sql, receipt=receipt, contract=mismatched_contract)

    assert res.success is False
    assert "contract_id mismatch" in res.stderr


def test_sandbox_gatekeeper_rejects_wrong_execution_unit_id(sample_duckdb, revenue_by_region_contract):
    valid_sql = """
    SELECT region, CAST(SUM(sales) AS DOUBLE) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    _, _, receipt = SQLASTConformanceVerifier.verify(valid_sql, revenue_by_region_contract)

    mismatched_contract = revenue_by_region_contract.model_copy(update={"execution_unit_id": "unit_step_99"})
    executor = SandboxedSQLExecutor(sample_duckdb)
    res = executor.execute(valid_sql, receipt=receipt, contract=mismatched_contract)

    assert res.success is False
    assert "execution_unit_id mismatch" in res.stderr


# ===========================================================================
# 4. End-to-End Vertical Slice Tests
# ===========================================================================

def test_vertical_slice_end_to_end_revenue_by_region_2017(sample_duckdb):
    """
    Proves the full pipeline for 'Show total revenue by region in 2017':
    Objective -> Plan -> Validated AIR -> Contract -> CodeGen -> SQLGlot Verifier -> Receipt -> Sandbox -> Result
    """
    from src.ingestion.profiler import AutomatedDataProfiler
    profile = AutomatedDataProfiler.profile_dataset(sample_duckdb, session_id="test_vslice")

    # Mock LLM to return valid query adhering to contract
    mock_llm = MagicMock()
    code_resp = MagicMock()
    code_resp.content = """```sql
    SELECT
        "region",
        CAST(SUM("sales") AS DOUBLE) AS total_revenue
    FROM sample_superstore
    WHERE "order_date" >= '2017-01-01' AND "order_date" < '2018-01-01'
    GROUP BY "region"
    ORDER BY total_revenue DESC
    ```"""
    mock_llm.invoke.return_value = code_resp

    graph = create_analytics_graph(sample_duckdb, llm=mock_llm)

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(sales)",
        target_table="sample_superstore",
        candidate_dimensions=["region", "ship_mode", "segment"],
        filter_conditions=["order_date >= '2017-01-01' AND order_date < '2018-01-01'"],
    )

    plan = AnalysisPlan(
        reasoning="Compute revenue by region for 2017",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Calculate total revenue by region in 2017",
                metric_col="sales",
                agg_func="SUM",
                group_col="region",
                where_clause="order_date >= '2017-01-01' AND order_date < '2018-01-01'",
            )
        ]
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "vslice_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": profile,
        "objective": objective,
        "plan": plan,
    }

    final_state = graph.invoke(initial_state)

    # Asserts that pipeline succeeded
    assert final_state["status"] == "COMPLETED"
    assert final_state["current_receipt"] is not None
    assert final_state["current_receipt"].conformance_passed is True

    # Assert execution result
    step_res = final_state["step_results"]["step_1"]
    assert step_res["success"] is True
    df = step_res["dataframe"]
    assert df is not None
    assert set(df.columns) == {"region", "total_revenue"}
    assert len(df) == 2  # West ($400.0) and East ($400.0)

    # Numerical assertion (Fix F-06)
    res_dict = dict(zip(df["region"], df["total_revenue"]))
    assert res_dict == {"West": 400.0, "East": 400.0}


# ===========================================================================
# 5. Targeted Remediation Tests (F-01 through F-05)
# ===========================================================================

# --- F-01: Unauthorized JOINs, CTEs, Derived Tables ---

def test_sql_verifier_rejects_unauthorized_table_join(revenue_by_region_contract):
    sql = """
    SELECT s.region, SUM(s.sales) AS total_revenue
    FROM sample_superstore s
    JOIN other_table o ON s.order_id = o.order_id
    WHERE s.order_date >= '2017-01-01' AND s.order_date < '2018-01-01'
    GROUP BY s.region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized JOIN detected" in err


def test_sql_verifier_rejects_self_join(revenue_by_region_contract):
    sql = """
    SELECT a.region, SUM(a.sales) AS total_revenue
    FROM sample_superstore a
    JOIN sample_superstore b ON a.order_id = b.order_id
    WHERE a.order_date >= '2017-01-01' AND a.order_date < '2018-01-01'
    GROUP BY a.region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized self-join detected" in err or "Unauthorized JOIN detected" in err


def test_sql_verifier_rejects_cte_table_source(revenue_by_region_contract):
    sql = """
    WITH extra_cte AS (
        SELECT * FROM other_table
    )
    SELECT region, SUM(sales) AS total_revenue
    FROM extra_cte
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized table source(s) detected" in err or "Table binding mismatch" in err


def test_sql_verifier_rejects_derived_table_source(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM (
        SELECT * FROM other_table
    ) sub
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized table source(s) detected" in err or "Table binding mismatch" in err


# --- F-02: Output Projections & Schema Verification ---

def test_sql_verifier_rejects_extra_projection_ship_mode(revenue_by_region_contract):
    sql = """
    SELECT region, ship_mode, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region, ship_mode
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions detected" in err or "Unauthorized column" in err


def test_sql_verifier_rejects_extra_projection_segment(revenue_by_region_contract):
    sql = """
    SELECT region, segment, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region, segment
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions detected" in err or "Unauthorized column" in err


def test_sql_verifier_rejects_extra_constant_projection(revenue_by_region_contract):
    sql = """
    SELECT region, 1 AS flag, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized non-aggregate expression/constant in projection" in err or "Schema projection count mismatch" in err


def test_sql_verifier_rejects_extra_arbitrary_expression(revenue_by_region_contract):
    sql = """
    SELECT region, sales * 2 AS double_sales, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region, sales
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions" in err or "Schema projection count mismatch" in err or "Unexpected projection" in err


def test_sql_verifier_rejects_renamed_unauthorized_column(revenue_by_region_contract):
    sql = """
    SELECT ship_mode AS region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY ship_mode
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized GROUP BY dimensions detected" in err or "Unauthorized source column" in err


# --- F-03: ORDER BY & HAVING Authorization ---

def test_sql_verifier_order_by_total_revenue_passes(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    ORDER BY total_revenue DESC
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is True
    assert err is None


def test_sql_verifier_order_by_region_passes(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    ORDER BY region ASC
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is True
    assert err is None


def test_sql_verifier_rejects_order_by_ship_mode(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    ORDER BY ship_mode ASC
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized ORDER BY column: 'ship_mode'" in err


def test_sql_verifier_rejects_order_by_segment(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    ORDER BY segment DESC
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized ORDER BY column: 'segment'" in err


def test_sql_verifier_rejects_unauthorized_having(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    HAVING SUM(sales) > 100
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized HAVING clause" in err


def test_sql_verifier_rejects_having_with_base_column(revenue_by_region_contract):
    contract_with_having = revenue_by_region_contract.model_copy(
        update={"authorized_having": ["SUM(sales) > 100"]}
    )
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    HAVING region = 'West'
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, contract_with_having)
    assert is_valid is False
    assert "Base column 'region' in HAVING clause is forbidden" in err


# --- F-04: Filter Authorization Policy ---

def test_sql_verifier_exact_required_temporal_filter_passes(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is True
    assert err is None


def test_sql_verifier_rejects_extra_unspecified_filter(revenue_by_region_contract):
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01' AND region = 'West'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, revenue_by_region_contract)
    assert is_valid is False
    assert "Unauthorized/unspecified filter condition detected in WHERE clause" in err


def test_sql_verifier_explicitly_permitted_filter_passes(revenue_by_region_contract):
    contract_with_permitted = revenue_by_region_contract.model_copy(
        update={
            "permitted_filters": [
                TypedPredicate(
                    column=ColumnRef(name="region", data_type=DataType.STRING),
                    operator=PredicateOperator.EQ,
                    literal_value="West",
                )
            ]
        }
    )
    sql = """
    SELECT region, SUM(sales) AS total_revenue
    FROM sample_superstore
    WHERE order_date >= '2017-01-01' AND order_date < '2018-01-01' AND region = 'West'
    GROUP BY region
    """
    is_valid, err, receipt = SQLASTConformanceVerifier.verify(sql, contract_with_permitted)
    assert is_valid is True
    assert err is None


# --- F-05: Compiler Fail-Closed Authorization ---

def test_compiler_fails_when_dimension_missing_from_authorization_matrix(base_superstore_schemas):
    agg_op = AggregateOperation(
        operation_id="op_agg_missing_auth",
        description="Aggregate revenue by region without auth record",
        input_datasets=["sample_superstore"],
        output_alias="view_res",
        output_schema={"region": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="region", data_type=DataType.STRING)],
        measures=[
            AggregateMeasureSpec(output_alias="total_revenue", function=AggregationFunction.SUM, target_column=ColumnRef(name="sales", data_type=DataType.FLOAT))
        ],
        is_terminal=True,
    )
    air = MasterAIR(
        lifecycle_state=AIRLifecycleState.VALIDATED,
        raw_user_question="Show revenue by region",
        metadata_summary="Revenue by region",
        grain_assessment=GrainAssessment(target_entity="order_item", supporting_evidence="line-item"),
        dimensions=[],  # No authorization record for 'region'
        operations=[agg_op],
        terminal_operation_ids=["op_agg_missing_auth"],
    )
    compiler = AIRContractCompiler(air, base_superstore_schemas)
    with pytest.raises(CompilerError, match="has NO authorization record"):
        compiler.compile()


def test_compiler_fails_when_dimension_explicitly_denied_grouping(base_superstore_schemas):
    agg_op = AggregateOperation(
        operation_id="op_agg_denied_auth",
        description="Aggregate revenue by ship_mode which is denied",
        input_datasets=["sample_superstore"],
        output_alias="view_res",
        output_schema={"ship_mode": DataType.STRING, "total_revenue": DataType.FLOAT},
        group_by_columns=[ColumnRef(name="ship_mode", data_type=DataType.STRING)],
        measures=[
            AggregateMeasureSpec(output_alias="total_revenue", function=AggregationFunction.SUM, target_column=ColumnRef(name="sales", data_type=DataType.FLOAT))
        ],
        is_terminal=True,
    )
    air = MasterAIR(
        lifecycle_state=AIRLifecycleState.VALIDATED,
        raw_user_question="Show revenue by ship mode",
        metadata_summary="Revenue by ship mode",
        grain_assessment=GrainAssessment(target_entity="order_item", supporting_evidence="line-item"),
        dimensions=[
            DimensionAuthorizationMatrix(column_name="ship_mode", allow_grouping=False)
        ],
        operations=[agg_op],
        terminal_operation_ids=["op_agg_denied_auth"],
    )
    compiler = AIRContractCompiler(air, base_superstore_schemas)
    with pytest.raises(CompilerError, match="not authorized for GROUP BY"):
        compiler.compile()
