"""
AIR Phase 4.0D — End-to-End Integration and Adversarial Test Suite.

Validates the full capability integration through the actual LangGraph workflow:
- Benchmark 1: "Show total revenue by region in 2017." (Exact numerical match)
- Benchmark 2: "Show total revenue by region." (No temporal filter)
- Benchmark 3: "Show total revenue in 2017." (No grouping dimension)

Adversarial Matrix:
- Unauthorized candidate grouping dimension fails closed.
- Unauthorized filter column fails closed.
- Missing capability ID or version mismatch fails closed.
- Incompatible port types fail closed.
- Cyclic DAG fails closed.
- Verifier rejects extra projections.
- Verifier rejects altered temporal bounds.
- Verifier rejects altered aggregation function / arithmetic.
- Sandbox gatekeeper rejects missing or tampered ExecutionReceipt.
- Validator detects incorrect SQL results and halts fail-closed.
- Unrestricted fallback is strictly prevented in registry mode.
"""

import math
import duckdb
import pandas as pd
import pytest

from src.agent.air.schemas import (
    ColumnRef,
    DataType,
    DimensionAuthorizationMatrix,
)
from src.agent.air.registry.models import ExecutionMode
from src.agent.air.registry.registry import AnalyticalCapabilityRegistry
from src.agent.air.registry.immutability import RegistryError
from src.agent.air.registry.bootstrap import bootstrap_registry
from src.agent.air.registry.composition import (
    CapabilityDAG,
    CapabilityDAGNode,
    CapabilityResolver,
    ComposedExecutionContract,
    ComposedPipelineCompiler,
    ComposedPipelineVerifier,
    ComposedPipelineValidator,
)
from src.agent.air.registry.ports import (
    BaseTableColumnBinding,
    BaseTableRelationBinding,
    OperationOutputBinding,
)
from src.agent.air.registry.builtin.temporal_filter import (
    TEMPORAL_FILTER_CAPABILITY_ID,
    TEMPORAL_FILTER_VERSION,
)
from src.agent.air.registry.builtin.grouped_sum import (
    GROUPED_SUM_CAPABILITY_ID,
    GROUPED_SUM_VERSION,
)
from src.agent.graph import create_analytics_graph
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    ColumnProfile,
    DatasetProfile,
    PlanStep,
    TableProfile,
)
from src.agent.state import AnalyticsAgentState
from src.execution.sandbox import SandboxedSQLExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def ensure_frozen_registry():
    """Ensure registry singleton is bootstrapped and frozen for tests."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    if not reg.is_frozen:
        bootstrap_registry(reg)
    return reg


@pytest.fixture
def superstore_duckdb():
    """
    Creates an in-memory DuckDB database with the authoritative 2017 Superstore dataset.
    Target totals in 2017:
      - Central: 337010.61
      - East: 449166.57
      - South: 170071.53
      - West: 506046.06
    Plus rows outside 2017 to verify temporal filtering boundary.
    """
    con = duckdb.connect(":memory:")

    # Synthetic rows that match exact regional totals in 2017
    # and contain out-of-boundary 2016 and 2018 records
    rows = [
        # 2017 Central ($337010.61)
        (1, "2017-01-15 10:00:00", "Central", 300000.00),
        (2, "2017-06-20 14:30:00", "Central", 37010.61),
        # 2017 East ($449166.57)
        (3, "2017-03-10 09:15:00", "East", 400000.00),
        (4, "2017-11-05 16:45:00", "East", 49166.57),
        # 2017 South ($170071.53)
        (5, "2017-04-22 11:20:00", "South", 100000.00),
        (6, "2017-08-14 13:10:00", "South", 70071.53),
        # 2017 West ($506046.06)
        (7, "2017-02-18 08:00:00", "West", 500000.00),
        (8, "2017-12-30 23:59:59", "West", 6046.06),
        # Outside 2017 (must be excluded by temporal filter)
        (9, "2016-12-31 23:59:59", "Central", 99999.00),
        (10, "2018-01-01 00:00:00", "East", 88888.00),
        (11, "2018-05-12 12:00:00", "West", 77777.00),
    ]

    df = pd.DataFrame(rows, columns=["row_id", "order_date", "region", "revenue"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    con.register("superstore", df)
    con.execute("CREATE TABLE sample_superstore AS SELECT * FROM superstore")

    return con


@pytest.fixture
def superstore_profile():
    return DatasetProfile(
        session_id="test_superstore_sess",
        tables=[
            TableProfile(
                table_name="sample_superstore",
                row_count=11,
                column_count=4,
                columns=[
                    ColumnProfile(name="row_id", data_type="INTEGER", is_numeric=True),
                    ColumnProfile(name="order_date", data_type="TIMESTAMP", is_temporal=True),
                    ColumnProfile(name="region", data_type="VARCHAR", distinct_count=4),
                    ColumnProfile(name="revenue", data_type="FLOAT", is_numeric=True),
                ],
            )
        ]
    )


@pytest.fixture
def authorized_dimensions():
    return {
        "region": DimensionAuthorizationMatrix(
            column_name="region",
            allow_grouping=True,
            allow_filter_predicate=True,
        ),
        "order_date": DimensionAuthorizationMatrix(
            column_name="order_date",
            allow_grouping=False,
            allow_filter_predicate=True,
        ),
        "candidate_unauthorized_dim": DimensionAuthorizationMatrix(
            column_name="candidate_unauthorized_dim",
            allow_grouping=False,
            allow_filter_predicate=False,
        ),
    }


# ---------------------------------------------------------------------------
# 1. Primary Objective Benchmark 1: "Show total revenue by region in 2017."
# ---------------------------------------------------------------------------

def test_benchmark_1_total_revenue_by_region_in_2017(superstore_duckdb, superstore_profile, authorized_dimensions):
    """
    Primary Objective Benchmark:
    "Show total revenue by region in 2017."
    Autonomously resolves capabilities, compiles DAG, verifies AST, executes,
    and independently validates exact numerical match:
      - Central: 337010.61
      - East: 449166.57
      - South: 170071.53
      - West: 506046.06
    """
    graph = create_analytics_graph(superstore_duckdb)

    initial_state: AnalyticsAgentState = {
        "session_id": "bench_1_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
    }

    final_state = graph.invoke(initial_state)

    # 1. Pipeline Status
    assert final_state["status"] == "COMPLETED"
    assert final_state.get("compilation_error") is None
    assert final_state.get("verification_error") is None

    # 2. Receipt and Conformance
    receipt = final_state.get("current_receipt")
    assert receipt is not None
    assert receipt.conformance_passed is True

    # 3. Composed Contract and DAG
    contract: ComposedExecutionContract = final_state.get("composed_contract")
    assert contract is not None
    assert (TEMPORAL_FILTER_CAPABILITY_ID, TEMPORAL_FILTER_VERSION) in contract.bound_capabilities
    assert (GROUPED_SUM_CAPABILITY_ID, GROUPED_SUM_VERSION) in contract.bound_capabilities
    assert contract.authorized_group_by == ["region"]
    assert len(contract.required_filters) == 1
    assert contract.required_filters[0].operator.value == "between_half_open"

    # 4. Numerical Accuracy Against Expected Totals
    step_res = final_state["step_results"]["step_1"]
    assert step_res["success"] is True
    res_df: pd.DataFrame = step_res["dataframe"]
    assert res_df is not None
    assert set(res_df.columns) == {"region", "total_revenue"}
    assert len(res_df) == 4

    results_dict = dict(zip(res_df["region"], res_df["total_revenue"]))
    expected_totals = {
        "Central": 337010.61,
        "East": 449166.57,
        "South": 170071.53,
        "West": 506046.06,
    }

    for region, exp_val in expected_totals.items():
        assert region in results_dict
        act_val = results_dict[region]
        assert math.isclose(act_val, exp_val, rel_tol=1e-5, abs_tol=1e-4)

    # 5. Provenance and Evidence Records
    evidence_list = final_state.get("evidence_records", [])
    assert len(evidence_list) > 0
    val_evidence = next(e for e in evidence_list if e.step_id == "step_1")
    assert val_evidence.validation_status == "VALIDATED_DETERMINISTICALLY"

    prov_dag = final_state.get("provenance_dag", {})
    nodes = prov_dag.get("nodes", [])
    node_data = next(n for n in nodes if n["step_id"] == "step_1")
    assert node_data is not None
    assert node_data["metadata"]["execution_mode"] == "registry"
    assert len(node_data["metadata"]["bound_capabilities"]) == 2


# ---------------------------------------------------------------------------
# 2. Benchmark 2: "Show total revenue by region." (No temporal filter)
# ---------------------------------------------------------------------------

def test_benchmark_2_total_revenue_by_region_no_filter(superstore_duckdb, superstore_profile, authorized_dimensions):
    """
    Benchmark 2:
    "Show total revenue by region."
    Binds only aggregate.grouped_sum.
    Strictly verifies NO temporal filter is introduced in contract or SQL.
    """
    graph = create_analytics_graph(superstore_duckdb)

    initial_state: AnalyticsAgentState = {
        "session_id": "bench_2_sess",
        "user_question": "Show total revenue by region.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPLETED"

    contract: ComposedExecutionContract = final_state.get("composed_contract")
    assert contract is not None
    # Strictly grouped sum, NO temporal filter
    assert len(contract.bound_capabilities) == 1
    assert contract.bound_capabilities[0] == (GROUPED_SUM_CAPABILITY_ID, GROUPED_SUM_VERSION)
    assert len(contract.required_filters) == 0

    # Verify executed SQL has NO WHERE clause
    code = final_state.get("current_code", "").upper()
    assert "WHERE" not in code
    assert "GROUP BY" in code

    # Check results (includes all 2016, 2017, and 2018 records)
    res_df = final_state["step_results"]["step_1"]["dataframe"]
    assert len(res_df) == 4
    results_dict = dict(zip(res_df["region"], res_df["total_revenue"]))
    # Central includes 337010.61 + 99999.00 = 437009.61
    assert math.isclose(results_dict["Central"], 437009.61, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 3. Benchmark 3: "Show total revenue in 2017." (No grouping dimension)
# ---------------------------------------------------------------------------

def test_benchmark_3_total_revenue_in_2017_no_grouping(superstore_duckdb, superstore_profile, authorized_dimensions):
    """
    Benchmark 3:
    "Show total revenue in 2017."
    Binds only filter.temporal_range on revenue measure.
    Strictly verifies NO grouping dimension is introduced in contract or SQL.
    """
    graph = create_analytics_graph(superstore_duckdb)

    initial_state: AnalyticsAgentState = {
        "session_id": "bench_3_sess",
        "user_question": "Show total revenue in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPLETED"

    contract: ComposedExecutionContract = final_state.get("composed_contract")
    assert contract is not None
    # Strictly filter capability, NO grouped sum
    assert len(contract.bound_capabilities) == 1
    assert contract.bound_capabilities[0] == (TEMPORAL_FILTER_CAPABILITY_ID, TEMPORAL_FILTER_VERSION)
    assert len(contract.authorized_group_by) == 0

    # Verify executed SQL has NO GROUP BY clause
    code = final_state.get("current_code", "").upper()
    assert "WHERE" in code
    assert "GROUP BY" not in code

    # Expected scalar sum across all 4 regions in 2017:
    # 337010.61 + 449166.57 + 170071.53 + 506046.06 = 1462294.77
    res_df = final_state["step_results"]["step_1"]["dataframe"]
    assert len(res_df) == 1
    assert "total_revenue" in res_df.columns
    total_val = float(res_df["total_revenue"].iloc[0])
    assert math.isclose(total_val, 1462294.77, rel_tol=1e-5, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# 4. Adversarial & Boundary Matrix Tests
# ---------------------------------------------------------------------------

def test_adversarial_unauthorized_grouping_dimension_fails_closed(superstore_duckdb, superstore_profile):
    """Adversarial 1: Candidate dimension without grouping authorization fails closed."""
    graph = create_analytics_graph(superstore_duckdb)

    unauthorized_dims = {
        "region": DimensionAuthorizationMatrix(
            column_name="region",
            allow_grouping=False,  # Unauthorized for grouping
            allow_filter_predicate=True,
        ),
        "order_date": DimensionAuthorizationMatrix(
            column_name="order_date",
            allow_grouping=False,
            allow_filter_predicate=True,
        ),
    }

    initial_state: AnalyticsAgentState = {
        "session_id": "adv_1_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": unauthorized_dims,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPILATION_FAILED"
    assert "Security Authorization Failure" in final_state["compilation_error"]
    assert "not authorized for grouping" in final_state["compilation_error"]
    assert final_state.get("current_receipt") is None


def test_adversarial_unauthorized_filter_column_fails_closed(superstore_duckdb, superstore_profile):
    """Adversarial 2: Temporal column without filter authorization fails closed."""
    graph = create_analytics_graph(superstore_duckdb)

    unauthorized_dims = {
        "region": DimensionAuthorizationMatrix(
            column_name="region",
            allow_grouping=True,
            allow_filter_predicate=True,
        ),
        "order_date": DimensionAuthorizationMatrix(
            column_name="order_date",
            allow_grouping=False,
            allow_filter_predicate=False,  # Unauthorized for filtering
        ),
    }

    initial_state: AnalyticsAgentState = {
        "session_id": "adv_2_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": unauthorized_dims,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPILATION_FAILED"
    assert "Security Authorization Failure" in final_state["compilation_error"]
    assert "not authorized for filtering" in final_state["compilation_error"]


def test_adversarial_missing_capability_or_version_mismatch(superstore_profile):
    """Adversarial 3: Unregistered capability ID or invalid version fails closed."""
    registry = AnalyticalCapabilityRegistry.get_instance()
    base_schemas = {"sample_superstore": {"order_date": DataType.DATETIME, "region": DataType.STRING, "revenue": DataType.FLOAT}}

    # 1. Non-existent capability
    invalid_node = CapabilityDAGNode(
        node_id="bad_node",
        capability_id="aggregate.unapproved_capability",
        capability_version="1.0.0",
        port_bindings=(
            BaseTableRelationBinding(consumer_port_name="in_relation", table_name="sample_superstore"),
        ),
        dependencies=(),
    )
    dag = CapabilityDAG(nodes=(invalid_node,))
    with pytest.raises(RegistryError, match="is not registered"):
        dag.validate_dag(registry, base_schemas)

    # 2. Version mismatch
    version_node = CapabilityDAGNode(
        node_id="bad_version_node",
        capability_id=GROUPED_SUM_CAPABILITY_ID,
        capability_version="9.9.9",
        port_bindings=(
            BaseTableRelationBinding(consumer_port_name="in_relation", table_name="sample_superstore"),
        ),
        dependencies=(),
    )
    dag2 = CapabilityDAG(nodes=(version_node,))
    with pytest.raises(RegistryError, match="is not registered"):
        dag2.validate_dag(registry, base_schemas)


def test_adversarial_incompatible_port_types_fails_closed(superstore_profile):
    """Adversarial 4: Binding non-temporal column to time_col or type mismatch fails closed."""
    registry = AnalyticalCapabilityRegistry.get_instance()
    base_schemas = {"sample_superstore": {"order_date": DataType.DATETIME, "region": DataType.STRING, "revenue": DataType.FLOAT}}

    # Binding STRING region column to DATETIME time_col port
    invalid_port_node = CapabilityDAGNode(
        node_id="op_filter_bad_port",
        capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
        capability_version=TEMPORAL_FILTER_VERSION,
        port_bindings=(
            BaseTableRelationBinding(consumer_port_name="in_relation", table_name="sample_superstore"),
            BaseTableColumnBinding(
                consumer_port_name="time_col",
                table_name="sample_superstore",
                column_name="region",  # STRING instead of DATETIME
                expected_data_type=DataType.STRING,
            ),
        ),
        dependencies=(),
    )
    dag = CapabilityDAG(nodes=(invalid_port_node,))
    with pytest.raises(TypeError, match="rejects column 'region' of type"):
        dag.validate_dag(registry, base_schemas)


def test_adversarial_cyclic_dag_fails_closed():
    """Adversarial 5: Cyclic DAG dependencies fail closed during DAG validation."""
    registry = AnalyticalCapabilityRegistry.get_instance()
    base_schemas = {"sample_superstore": {"order_date": DataType.DATETIME, "region": DataType.STRING, "revenue": DataType.FLOAT}}

    node_a = CapabilityDAGNode(
        node_id="node_a",
        capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
        capability_version=TEMPORAL_FILTER_VERSION,
        port_bindings=(
            BaseTableRelationBinding(consumer_port_name="in_relation", table_name="sample_superstore"),
            BaseTableColumnBinding(consumer_port_name="time_col", table_name="sample_superstore", column_name="order_date", expected_data_type=DataType.DATETIME),
        ),
        dependencies=("node_b",),  # Cycle: A depends on B
    )
    node_b = CapabilityDAGNode(
        node_id="node_b",
        capability_id=GROUPED_SUM_CAPABILITY_ID,
        capability_version=GROUPED_SUM_VERSION,
        port_bindings=(
            BaseTableRelationBinding(consumer_port_name="in_relation", table_name="sample_superstore"),
            BaseTableColumnBinding(consumer_port_name="group_col", table_name="sample_superstore", column_name="region", expected_data_type=DataType.STRING),
            BaseTableColumnBinding(consumer_port_name="measure_col", table_name="sample_superstore", column_name="revenue", expected_data_type=DataType.FLOAT),
        ),
        dependencies=("node_a",),  # Cycle: B depends on A
    )
    dag = CapabilityDAG(nodes=(node_a, node_b))
    with pytest.raises(ValueError, match="Cycle detected in CapabilityDAG"):
        dag.validate_dag(registry, base_schemas)


def test_adversarial_verifier_rejects_extra_projections(superstore_duckdb, superstore_profile, authorized_dimensions):
    """Adversarial 6: Verifier rejects unauthorized injected projections."""
    graph = create_analytics_graph(superstore_duckdb)

    # Injected extra projection: AVG(revenue)
    injected_sql = (
        'SELECT "region", CAST(SUM("revenue") AS DOUBLE) AS total_revenue, AVG("revenue") AS extra_avg '
        'FROM "sample_superstore" '
        'WHERE "order_date" >= \'2017-01-01 00:00:00\' AND "order_date" < \'2018-01-01 00:00:00\' '
        'GROUP BY "region"'
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "adv_6_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
        "current_code": injected_sql,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "VERIFICATION_FAILED"
    assert "Expected exactly 2 projected columns, got 3" in final_state["verification_error"]
    assert final_state.get("current_receipt") is None


def test_adversarial_verifier_rejects_altered_temporal_bounds(superstore_duckdb, superstore_profile, authorized_dimensions):
    """Adversarial 7: Verifier rejects altered temporal bounds."""
    graph = create_analytics_graph(superstore_duckdb)

    # Altered boundary: 2016-01-01 instead of 2017-01-01
    altered_sql = (
        'SELECT "region", CAST(SUM("revenue") AS DOUBLE) AS total_revenue '
        'FROM "sample_superstore" '
        'WHERE "order_date" >= \'2016-01-01 00:00:00\' AND "order_date" < \'2018-01-01 00:00:00\' '
        'GROUP BY "region"'
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "adv_7_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
        "current_code": altered_sql,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "VERIFICATION_FAILED"
    assert "Temporal filter verification failed" in final_state["verification_error"]


def test_adversarial_verifier_rejects_altered_aggregation_function(superstore_duckdb, superstore_profile, authorized_dimensions):
    """Adversarial 8: Verifier rejects altered aggregation arithmetic (e.g. SUM * 1.1)."""
    graph = create_analytics_graph(superstore_duckdb)

    altered_sql = (
        'SELECT "region", CAST(SUM("revenue") AS DOUBLE) * 1.1 AS total_revenue '
        'FROM "sample_superstore" '
        'WHERE "order_date" >= \'2017-01-01 00:00:00\' AND "order_date" < \'2018-01-01 00:00:00\' '
        'GROUP BY "region"'
    )

    initial_state: AnalyticsAgentState = {
        "session_id": "adv_8_sess",
        "user_question": "Show total revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
        "current_code": altered_sql,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "VERIFICATION_FAILED"
    assert "Altered aggregation expression detected" in final_state["verification_error"]


def test_adversarial_sandbox_rejects_missing_or_tampered_receipt(superstore_duckdb, superstore_profile, authorized_dimensions):
    """Adversarial 9: Sandboxed executor gatekeeper blocks query without valid receipt."""
    sandbox = SandboxedSQLExecutor(superstore_duckdb)

    # 1. Compile legitimate contract
    base_schemas = {"sample_superstore": {"order_date": DataType.DATETIME, "region": DataType.STRING, "revenue": DataType.FLOAT}}
    registry = AnalyticalCapabilityRegistry.get_instance()
    dag = CapabilityResolver.resolve(
        "Show total revenue by region in 2017.",
        "sample_superstore",
        authorized_dimensions,
        base_schemas,
        registry,
    )
    contract, valid_sql = ComposedPipelineCompiler.compile(dag, "sample_superstore", base_schemas, registry)

    # Case A: Receipt is None
    res_no_receipt = sandbox.execute(valid_sql, receipt=None, contract=contract)
    assert res_no_receipt.success is False
    assert "Missing ExecutionReceipt" in res_no_receipt.stderr

    # Case B: Receipt with tampered code hash
    _, _, receipt = ComposedPipelineVerifier.verify(valid_sql, contract)
    tampered_receipt = receipt.model_copy(update={"code_hash": "bad_hash_1234567890"})
    res_tampered = sandbox.execute(valid_sql, receipt=tampered_receipt, contract=contract)
    assert res_tampered.success is False
    assert "Sandbox Gatekeeper" in res_tampered.stderr
    assert "code_hash mismatch" in res_tampered.stderr


def test_adversarial_validator_detects_manipulated_sql_results(superstore_duckdb, superstore_profile, authorized_dimensions):
    """Adversarial 10: Independent validator detects manipulated SQL results and fails closed."""
    raw_df = superstore_duckdb.execute('SELECT * FROM "sample_superstore"').df()

    # Legitimate contract
    base_schemas = {"sample_superstore": {"order_date": DataType.DATETIME, "region": DataType.STRING, "revenue": DataType.FLOAT}}
    registry = AnalyticalCapabilityRegistry.get_instance()
    dag = CapabilityResolver.resolve(
        "Show total revenue by region in 2017.",
        "sample_superstore",
        authorized_dimensions,
        base_schemas,
        registry,
    )
    contract, _ = ComposedPipelineCompiler.compile(dag, "sample_superstore", base_schemas, registry)

    # Manipulated SQL dataframe with altered Central value
    manipulated_df = pd.DataFrame([
        {"region": "Central", "total_revenue": 999999.99},
        {"region": "East", "total_revenue": 449166.57},
        {"region": "South", "total_revenue": 170071.53},
        {"region": "West", "total_revenue": 506046.06},
    ])

    val_res = ComposedPipelineValidator.validate(raw_df, manipulated_df, contract)
    assert val_res.is_valid is False
    assert val_res.error_type == "VALUE_MISMATCH"
    assert "Central" in val_res.error_message


def test_adversarial_unrestricted_fallback_prevented_in_registry_mode(superstore_duckdb, superstore_profile, authorized_dimensions):
    """Adversarial 11: In registry mode, unsupported requests fail closed without falling back to LLM."""
    graph = create_analytics_graph(superstore_duckdb)

    # Request involving operations outside approved capabilities (e.g. AVG or RANK)
    initial_state: AnalyticsAgentState = {
        "session_id": "adv_11_sess",
        "user_question": "Show average revenue by region in 2017.",
        "profile": superstore_profile,
        "execution_mode": ExecutionMode.REGISTRY,
        "dimension_authorizations": authorized_dimensions,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "COMPILATION_FAILED"
    assert "outside Phase 4.0 approved capabilities" in final_state["compilation_error"]
    assert final_state.get("current_receipt") is None
    # No SQL was executed; step_results records fail-closed error
    assert final_state["step_results"]["step_1"]["success"] is False
    assert final_state["step_results"]["step_1"]["dataframe"] is None
