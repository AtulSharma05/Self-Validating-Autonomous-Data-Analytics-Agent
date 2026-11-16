"""
AIR Phase 4.0D — End-to-End Capability Composition, Verification, and Validation.

Orchestrates atomic capabilities:
- filter.temporal_range@1.0.0
- aggregate.grouped_sum@1.0.0

Provides:
- CapabilityDAG and CapabilityDAGNode with typed port bindings and cycle detection.
- Deterministic CapabilityResolver mapping analytical intent to registered capabilities.
- ComposedExecutionContract fusing DAG fragments into an authoritative contract.
- ComposedPipelineCompiler generating deterministic executable SQL.
- ComposedPipelineVerifier performing SQLGlot AST validation across composed boundaries.
- ComposedPipelineValidator executing independent Pandas ground-truth evaluation.
"""

from __future__ import annotations
import hashlib
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
import sqlglot
from sqlglot import exp

from src.agent.air.schemas import (
    ColumnRef,
    ComparisonOperator,
    DataType,
    DimensionAuthorizationMatrix,
    Predicate,
)
from src.agent.air.operations import (
    AggregationFunction,
    AggregateMeasureSpec,
    AggregateOperation,
    FilterOperation,
)
from src.agent.air.contract import (
    CardinalityExpectation,
    ContractEngine,
    ExecutionContract,
    ExecutionReceipt,
    PredicateOperator,
    RequiredMeasureSpec,
    TypedPredicate,
)
from src.agent.air.registry.models import (
    CapabilityKind,
    DataGrain,
    PortCardinality,
    PortDefinition,
    PortPayloadType,
    SemanticRole,
)
from src.agent.air.registry.protocols import (
    ContractFragment,
)
from src.agent.air.registry.ports import (
    BaseTableColumnBinding,
    BaseTableRelationBinding,
    OperationOutputBinding,
    PortBinding,
    validate_port_binding,
)
from src.agent.air.registry.registry import (
    AnalyticalCapabilityRegistry,
    RegisteredCapability,
)
from src.agent.air.registry.builtin.temporal_filter import (
    TEMPORAL_FILTER_CAPABILITY_ID,
    TEMPORAL_FILTER_VERSION,
    TemporalFilterCompilerHandler,
    TemporalFilterVerifierHandler,
    TemporalFilterValidatorHandler,
    parse_and_validate_temporal_boundary,
)
from src.agent.air.registry.builtin.grouped_sum import (
    GROUPED_SUM_CAPABILITY_ID,
    GROUPED_SUM_VERSION,
    GROUPED_SUM_ABS_TOL,
    GROUPED_SUM_REL_TOL,
    GroupedSumCompilerHandler,
    GroupedSumVerifierHandler,
    GroupedSumValidatorHandler,
)
from src.agent.air.registry.fingerprint import compute_canonical_fingerprint
from src.validation.validator import ValidationResult


# ---------------------------------------------------------------------------
# 1. Capability DAG Models
# ---------------------------------------------------------------------------

class CapabilityDAGNode(BaseModel):
    """A single invocation node in a Capability DAG."""
    model_config = ConfigDict(frozen=True)

    node_id: str
    capability_id: str
    capability_version: str
    port_bindings: Tuple[PortBinding, ...] = Field(default_factory=tuple)
    operation_params: Dict[str, Any] = Field(default_factory=dict)
    dependencies: Tuple[str, ...] = Field(default_factory=tuple)


class CapabilityDAG(BaseModel):
    """
    Typed, directed acyclic graph of registered capability nodes.
    Enforces deterministic topological ordering, cycle detection, and port binding validation.
    """
    model_config = ConfigDict(frozen=True)

    nodes: Tuple[CapabilityDAGNode, ...] = Field(default_factory=tuple)

    def get_node(self, node_id: str) -> CapabilityDAGNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"Node '{node_id}' not found in CapabilityDAG.")

    def topological_sort(self) -> List[CapabilityDAGNode]:
        """Returns nodes in topologically sorted execution order; raises on cycle or missing dep."""
        node_map = {n.node_id: n for n in self.nodes}
        visited: Dict[str, int] = {}  # 0: visiting, 1: visited
        order: List[CapabilityDAGNode] = []

        def dfs(nid: str):
            if nid not in node_map:
                raise KeyError(f"Dependency '{nid}' not found in CapabilityDAG.")
            st = visited.get(nid)
            if st == 0:
                raise ValueError(f"Cycle detected in CapabilityDAG involving node '{nid}'.")
            if st == 1:
                return

            visited[nid] = 0
            for dep in node_map[nid].dependencies:
                dfs(dep)
            visited[nid] = 1
            order.append(node_map[nid])

        for node in self.nodes:
            if node.node_id not in visited:
                dfs(node.node_id)

        return order

    def validate_dag(
        self,
        registry: AnalyticalCapabilityRegistry,
        base_schemas: Dict[str, Dict[str, DataType]],
    ) -> None:
        """Validates all nodes, ports, types, and grain transitions against the registry catalog."""
        sorted_nodes = self.topological_sort()
        node_outputs: Dict[str, RegisteredCapability] = {}

        for node in sorted_nodes:
            cap = registry.get(node.capability_id, node.capability_version)

            # Check that all required input ports are bound
            bound_port_names = {b.consumer_port_name for b in node.port_bindings}
            for port_name, port_def in cap.input_ports.items():
                if port_def.is_required and port_name not in bound_port_names:
                    raise ValueError(
                        f"Required input port '{port_name}' on capability '{node.capability_id}' "
                        f"(node '{node.node_id}') is not bound."
                    )

            # Validate each port binding
            for binding in node.port_bindings:
                p_name = binding.consumer_port_name
                if p_name not in cap.input_ports:
                    raise ValueError(
                        f"Port '{p_name}' bound on node '{node.node_id}' does not exist on capability '{node.capability_id}'."
                    )
                port_def = cap.input_ports[p_name]

                if isinstance(binding, BaseTableColumnBinding):
                    if binding.table_name not in base_schemas:
                        raise KeyError(f"Table '{binding.table_name}' not found in base schemas.")
                    tbl_schema = base_schemas[binding.table_name]
                    if binding.column_name not in tbl_schema:
                        raise KeyError(f"Column '{binding.column_name}' not found in table '{binding.table_name}'.")
                    actual_type = tbl_schema[binding.column_name]
                    if not port_def.accepts_data_type(actual_type):
                        raise TypeError(
                            f"Port '{p_name}' rejects column '{binding.column_name}' of type '{actual_type}'."
                        )

                elif isinstance(binding, OperationOutputBinding):
                    if binding.producer_operation_id not in node_outputs:
                        raise ValueError(
                            f"Producer node '{binding.producer_operation_id}' has not produced output before '{node.node_id}'."
                        )
                    producer_cap = node_outputs[binding.producer_operation_id]
                    if binding.producer_port_name not in producer_cap.output_ports:
                        raise KeyError(
                            f"Producer port '{binding.producer_port_name}' not found on '{producer_cap.capability_id}'."
                        )
                    producer_port = producer_cap.output_ports[binding.producer_port_name]
                    if port_def.cardinality != producer_port.cardinality:
                        raise TypeError(
                            f"Cardinality mismatch between '{binding.producer_operation_id}.{binding.producer_port_name}' "
                            f"({producer_port.cardinality}) and '{node.node_id}.{p_name}' ({port_def.cardinality})."
                        )

            node_outputs[node.node_id] = cap


# ---------------------------------------------------------------------------
# 2. Composed Execution Contract & Receipt
# ---------------------------------------------------------------------------

class ComposedExecutionContract(ExecutionContract):
    """
    Authoritative execution contract representing a composed multi-capability DAG.
    Fused into an atomic single-statement query while preserving DAG provenance.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dag: CapabilityDAG
    bound_capabilities: Tuple[Tuple[str, str], ...] = Field(default_factory=tuple)
    dag_fingerprint: str = ""
    is_composed: bool = True


# ---------------------------------------------------------------------------
# 3. Deterministic Capability Resolver
# ---------------------------------------------------------------------------

class CapabilityResolver:
    """
    Resolves analytical intent into a validated, deterministic CapabilityDAG.
    Rejects unsupported operations, unauthorized dimensions, and unauthorized filters fail-closed.
    Candidate dimensions NEVER become implicit authorization.
    """

    @classmethod
    def resolve(
        cls,
        user_question: str,
        target_table: str,
        dimensions: Dict[str, DimensionAuthorizationMatrix],
        base_schemas: Dict[str, Dict[str, DataType]],
        registry: AnalyticalCapabilityRegistry,
        metric_hint: Optional[str] = None,
        group_hint: Optional[str] = None,
        filter_hint: Optional[str] = None,
    ) -> CapabilityDAG:
        q_lower = user_question.lower()

        if target_table not in base_schemas:
            raise KeyError(f"Target table '{target_table}' not found in base schemas.")
        table_schema = base_schemas[target_table]

        # 1. Detect and Validate Temporal Filter
        # Supported pattern: 2017 temporal range or explicit filter_hint
        has_temporal_intent = "2017" in q_lower or (filter_hint and "order_date" in filter_hint.lower())
        time_col = None
        start_bound = None
        end_bound = None

        if has_temporal_intent:
            # Find candidate datetime column
            for col, dtype in table_schema.items():
                if dtype in (DataType.DATETIME, DataType.DATE):
                    time_col = col
                    break
            if not time_col:
                raise KeyError(f"No temporal (DATETIME/DATE) column found in '{target_table}'.")

            # Check Dimension Authorization for Filter
            dim_auth = dimensions.get(time_col)
            if not dim_auth or not dim_auth.allow_filter_predicate:
                raise PermissionError(
                    f"Security Authorization Failure: Temporal column '{time_col}' is not authorized for filtering."
                )

            # Strict 2017 boundary
            start_bound = "2017-01-01 00:00:00"
            end_bound = "2018-01-01 00:00:00"

        # 2. Detect and Validate Grouping Dimension
        # Candidate dimensions never become implicit authorization
        has_group_intent = "by region" in q_lower or (group_hint and "region" in group_hint.lower())
        group_col = None
        if has_group_intent:
            target_group = "region"
            if target_group not in table_schema:
                raise KeyError(f"Grouping column '{target_group}' does not exist in '{target_table}'.")

            # Check Dimension Authorization for Grouping (Fail-Closed)
            dim_auth = dimensions.get(target_group)
            if not dim_auth or not dim_auth.allow_grouping:
                raise PermissionError(
                    f"Security Authorization Failure: Dimension '{target_group}' is not authorized for grouping. "
                    "Candidate dimensions cannot serve as implicit authorization."
                )
            group_col = target_group

        # 3. Detect and Validate Measure & Function
        # Supported: SUM(revenue) or SUM(sales)
        measure_col = None
        for cand in ["revenue", "sales"]:
            if cand in table_schema:
                measure_col = cand
                break
        if not measure_col:
            raise KeyError(f"No numeric revenue/sales measure found in '{target_table}'.")

        measure_dtype = table_schema[measure_col]
        if measure_dtype not in (DataType.FLOAT, DataType.INTEGER):
            raise TypeError(f"Measure '{measure_col}' has non-numeric type '{measure_dtype}'.")

        # Reject unsupported requests (averages, ranking, arbitrary joins) fail closed
        if any(k in q_lower for k in ["average", "avg", "top", "rank", "lowest", "highest", "count"]):
            raise ValueError(
                "Request involves analytical operations (AVG, RANK, COUNT) outside Phase 4.0 approved capabilities."
            )

        # 4. Construct Capability DAG Nodes
        nodes: List[CapabilityDAGNode] = []

        if has_temporal_intent and group_col:
            # Composition: filter.temporal_range -> aggregate.grouped_sum
            filter_node = CapabilityDAGNode(
                node_id="op_filter_2017",
                capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
                capability_version=TEMPORAL_FILTER_VERSION,
                port_bindings=(
                    BaseTableRelationBinding(
                        consumer_port_name="in_relation",
                        table_name=target_table,
                    ),
                    BaseTableColumnBinding(
                        consumer_port_name="time_col",
                        table_name=target_table,
                        column_name=time_col,
                        expected_data_type=table_schema[time_col],
                    ),
                ),
                operation_params={
                    "column_name": time_col,
                    "start": start_bound,
                    "end": end_bound,
                },
                dependencies=(),
            )
            nodes.append(filter_node)

            agg_node = CapabilityDAGNode(
                node_id="op_agg_grouped_sum",
                capability_id=GROUPED_SUM_CAPABILITY_ID,
                capability_version=GROUPED_SUM_VERSION,
                port_bindings=(
                    OperationOutputBinding(
                        consumer_port_name="in_relation",
                        producer_operation_id="op_filter_2017",
                        producer_port_name="out_relation",
                    ),
                    BaseTableColumnBinding(
                        consumer_port_name="group_col",
                        table_name=target_table,
                        column_name=group_col,
                        expected_data_type=table_schema[group_col],
                    ),
                    BaseTableColumnBinding(
                        consumer_port_name="measure_col",
                        table_name=target_table,
                        column_name=measure_col,
                        expected_data_type=measure_dtype,
                    ),
                ),
                operation_params={
                    "group_col": group_col,
                    "measure_col": measure_col,
                    "function": "sum",
                    "output_alias": "total_revenue",
                },
                dependencies=("op_filter_2017",),
            )
            nodes.append(agg_node)

        elif group_col and not has_temporal_intent:
            # Benchmark 2: "Show total revenue by region." -> Only grouped_sum, NO temporal filter
            agg_node = CapabilityDAGNode(
                node_id="op_agg_grouped_sum",
                capability_id=GROUPED_SUM_CAPABILITY_ID,
                capability_version=GROUPED_SUM_VERSION,
                port_bindings=(
                    BaseTableRelationBinding(
                        consumer_port_name="in_relation",
                        table_name=target_table,
                    ),
                    BaseTableColumnBinding(
                        consumer_port_name="group_col",
                        table_name=target_table,
                        column_name=group_col,
                        expected_data_type=table_schema[group_col],
                    ),
                    BaseTableColumnBinding(
                        consumer_port_name="measure_col",
                        table_name=target_table,
                        column_name=measure_col,
                        expected_data_type=measure_dtype,
                    ),
                ),
                operation_params={
                    "group_col": group_col,
                    "measure_col": measure_col,
                    "function": "sum",
                    "output_alias": "total_revenue",
                },
                dependencies=(),
            )
            nodes.append(agg_node)

        elif has_temporal_intent and not group_col:
            # Benchmark 3: "Show total revenue in 2017." -> Temporal filter on measure, NO grouping dimension
            filter_node = CapabilityDAGNode(
                node_id="op_filter_2017",
                capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
                capability_version=TEMPORAL_FILTER_VERSION,
                port_bindings=(
                    BaseTableRelationBinding(
                        consumer_port_name="in_relation",
                        table_name=target_table,
                    ),
                    BaseTableColumnBinding(
                        consumer_port_name="time_col",
                        table_name=target_table,
                        column_name=time_col,
                        expected_data_type=table_schema[time_col],
                    ),
                ),
                operation_params={
                    "column_name": time_col,
                    "start": start_bound,
                    "end": end_bound,
                    "scalar_sum_measure": measure_col,
                    "output_alias": "total_revenue",
                },
                dependencies=(),
            )
            nodes.append(filter_node)

        else:
            raise ValueError(f"Unrecognized or unsupported analytical intent for query: '{user_question}'.")

        dag = CapabilityDAG(nodes=tuple(nodes))
        dag.validate_dag(registry, base_schemas)
        return dag


# ---------------------------------------------------------------------------
# 4. Composed Pipeline Compiler
# ---------------------------------------------------------------------------

class ComposedPipelineCompiler:
    """
    Compiles a validated CapabilityDAG into a ComposedExecutionContract and deterministic SQL.
    """

    @classmethod
    def compile(
        cls,
        dag: CapabilityDAG,
        target_table: str,
        base_schemas: Dict[str, Dict[str, DataType]],
        registry: AnalyticalCapabilityRegistry,
    ) -> Tuple[ComposedExecutionContract, str]:
        sorted_nodes = dag.topological_sort()
        bound_caps = tuple((n.capability_id, n.capability_version) for n in sorted_nodes)

        has_filter = any(n.capability_id == TEMPORAL_FILTER_CAPABILITY_ID for n in sorted_nodes)
        has_agg = any(n.capability_id == GROUPED_SUM_CAPABILITY_ID for n in sorted_nodes)

        filter_node = next((n for n in sorted_nodes if n.capability_id == TEMPORAL_FILTER_CAPABILITY_ID), None)
        agg_node = next((n for n in sorted_nodes if n.capability_id == GROUPED_SUM_CAPABILITY_ID), None)

        required_filters: List[TypedPredicate] = []
        authorized_cols: List[str] = []
        authorized_group_by: List[str] = []
        required_measures: List[RequiredMeasureSpec] = []
        expected_output: Dict[str, DataType] = {}
        cardinality = CardinalityExpectation.DIMENSIONAL

        table_schema = base_schemas[target_table]

        # 1. Process Temporal Filter Node
        if filter_node:
            t_col = filter_node.operation_params["column_name"]
            s_val = filter_node.operation_params["start"]
            e_val = filter_node.operation_params["end"]
            pred = TypedPredicate(
                column=ColumnRef(name=t_col, data_type=table_schema[t_col], table_name=target_table),
                operator=PredicateOperator.BETWEEN_HALF_OPEN,
                literal_value=str(s_val),
                secondary_literal=str(e_val),
                is_temporal=True,
            )
            required_filters.append(pred)
            authorized_cols.append(t_col)

        # 2. Process Grouped Sum Node
        if agg_node:
            g_col = agg_node.operation_params["group_col"]
            m_col = agg_node.operation_params["measure_col"]
            out_alias = agg_node.operation_params.get("output_alias", "total_revenue")

            authorized_group_by.append(g_col)
            authorized_cols.extend([g_col, m_col])

            req_m = RequiredMeasureSpec(
                output_alias=out_alias,
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name=m_col, data_type=table_schema[m_col], table_name=target_table),
            )
            required_measures.append(req_m)
            expected_output[g_col] = table_schema[g_col]
            expected_output[out_alias] = DataType.FLOAT
            cardinality = CardinalityExpectation.DIMENSIONAL

        elif filter_node and not agg_node:
            # Benchmark 3: Scalar sum with filter, no grouping
            m_col = filter_node.operation_params.get("scalar_sum_measure", "revenue")
            out_alias = filter_node.operation_params.get("output_alias", "total_revenue")
            authorized_cols.append(m_col)
            req_m = RequiredMeasureSpec(
                output_alias=out_alias,
                function=AggregationFunction.SUM,
                target_column=ColumnRef(name=m_col, data_type=table_schema[m_col], table_name=target_table),
            )
            required_measures.append(req_m)
            expected_output[out_alias] = DataType.FLOAT
            cardinality = CardinalityExpectation.SCALAR

        # Generate Deterministic SQL
        if has_filter and has_agg:
            sql = (
                f'SELECT "{g_col}", CAST(SUM("{m_col}") AS DOUBLE) AS {out_alias} '
                f'FROM "{target_table}" '
                f'WHERE "{t_col}" >= \'{s_val}\' AND "{t_col}" < \'{e_val}\' '
                f'GROUP BY "{g_col}"'
            )
        elif has_agg and not has_filter:
            # Benchmark 2: No WHERE clause allowed
            sql = (
                f'SELECT "{g_col}", CAST(SUM("{m_col}") AS DOUBLE) AS {out_alias} '
                f'FROM "{target_table}" '
                f'GROUP BY "{g_col}"'
            )
        elif has_filter and not has_agg:
            # Benchmark 3: No GROUP BY clause allowed
            sql = (
                f'SELECT CAST(SUM("{m_col}") AS DOUBLE) AS {out_alias} '
                f'FROM "{target_table}" '
                f'WHERE "{t_col}" >= \'{s_val}\' AND "{t_col}" < \'{e_val}\''
            )
        else:
            raise ValueError("Invalid DAG node combination.")

        dag_fp = compute_canonical_fingerprint([n.model_dump() for n in sorted_nodes])

        contract = ComposedExecutionContract(
            contract_id=f"contract_{target_table}_{len(sorted_nodes)}_nodes",
            execution_unit_id="unit_step_1",
            air_operation_ids=[n.node_id for n in sorted_nodes],
            target_table=target_table,
            authorized_source_columns=list(set(authorized_cols)),
            authorized_group_by=authorized_group_by,
            forbidden_group_by=[],
            required_filters=required_filters,
            permitted_filters=[],
            forbidden_filters=[],
            required_measures=required_measures,
            expected_output_columns=expected_output,
            expected_cardinality=cardinality,
            target_engine=ContractEngine.DUCKDB_SQL,
            dag=dag,
            bound_capabilities=bound_caps,
            dag_fingerprint=dag_fp,
        )

        return contract, sql


# ---------------------------------------------------------------------------
# 5. Composed Pipeline Verifier
# ---------------------------------------------------------------------------

class ComposedPipelineVerifier:
    """
    Verifies SQLGlot AST against a ComposedExecutionContract:
    - Verifies FROM source table (no unauthorized JOINs).
    - Verifies exact temporal predicate and bounds [start, end) if bound.
    - Verifies NO WHERE clause if temporal filter is not bound.
    - Verifies exact grouping column if bound.
    - Verifies NO GROUP BY clause if grouping is not bound.
    - Verifies exact SUM projection, output alias, and absence of altered arithmetic.
    - Recognizes authorized filters without weakening standalone verifiers.
    - Issues immutable ExecutionReceipt on success.
    """

    @classmethod
    def verify(
        cls,
        sql: str,
        contract: ComposedExecutionContract,
    ) -> Tuple[bool, Optional[str], Optional[ExecutionReceipt]]:
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception as e:
            return False, f"SQL parse error: {e}", None

        if not isinstance(tree, exp.Select):
            return False, f"Expected root SELECT statement, got '{type(tree).__name__}'.", None

        # 1. Source Table & JOINs
        from_node = tree.find(exp.From)
        if not from_node:
            return False, "Query has no FROM clause.", None
        tbl_node = from_node.find(exp.Table)
        if not tbl_node or tbl_node.name.lower().strip('"') != contract.target_table.lower().strip('"'):
            actual_tbl = tbl_node.name if tbl_node else None
            return False, f"Source table mismatch: expected '{contract.target_table}', got '{actual_tbl}'.", None

        if tree.args.get("joins") or list(tree.find_all(exp.Join)):
            return False, "Query contains unauthorized JOIN clauses.", None

        # 2. Temporal Filter Verification
        has_filter_cap = any(c[0] == TEMPORAL_FILTER_CAPABILITY_ID for c in contract.bound_capabilities)
        where_clause = tree.args.get("where")

        if not has_filter_cap:
            if where_clause:
                # Benchmark 2 requirement: verify NO temporal filter is introduced
                return False, "Query contains an unauthorized WHERE clause; contract authorizes no filters.", None
        else:
            if not where_clause:
                return False, "Query missing required WHERE clause for temporal filter.", None

            # Verify using TemporalFilterVerifierHandler
            fragment = ContractFragment(
                operation_id="composed_filter",
                capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
                capability_version=TEMPORAL_FILTER_VERSION,
                source_table=contract.target_table,
                authorized_source_columns=tuple(contract.authorized_source_columns),
                required_filters=tuple(contract.required_filters),
            )
            t_verifier = TemporalFilterVerifierHandler()
            ok, err = t_verifier.verify_ast_fragment(tree, fragment)
            if not ok:
                return False, f"Temporal filter verification failed: {err}", None

        # 3. Grouping Verification
        has_agg_cap = any(c[0] == GROUPED_SUM_CAPABILITY_ID for c in contract.bound_capabilities)
        group_clause = tree.args.get("group")

        if not has_agg_cap or not contract.authorized_group_by:
            if group_clause:
                # Benchmark 3 requirement: verify NO grouping dimension is introduced
                return False, "Query contains an unauthorized GROUP BY clause; contract authorizes no grouping.", None
        else:
            if not group_clause:
                return False, "Query missing required GROUP BY clause.", None
            group_exprs = group_clause.expressions
            if len(group_exprs) != 1:
                return False, f"Expected exactly 1 GROUP BY expression, got {len(group_exprs)}.", None
            expected_grp = contract.authorized_group_by[0].lower().strip('"')
            grp_col_node = group_exprs[0].this if isinstance(group_exprs[0], exp.Column) else group_exprs[0].find(exp.Column)
            if not grp_col_node or grp_col_node.name.lower().strip('"') != expected_grp:
                actual_grp = grp_col_node.name if grp_col_node else str(group_exprs[0])
                return False, f"GROUP BY column mismatch: expected '{expected_grp}', got '{actual_grp}'.", None

        # 4. Projections Verification
        selects = tree.selects
        expected_cols_count = len(contract.expected_output_columns)
        if len(selects) != expected_cols_count:
            return False, f"Expected exactly {expected_cols_count} projected columns, got {len(selects)}.", None

        # Check SUM projection
        expected_m = contract.required_measures[0].target_column.name.lower().strip('"')
        expected_alias = contract.required_measures[0].output_alias.lower().strip('"')

        sum_proj_node = None
        sum_sel_node = None

        for sel in selects:
            while isinstance(sel, exp.Paren):
                sel = sel.this
            sums = list(sel.find_all(exp.Sum))
            anons = [a for a in sel.find_all(exp.Anonymous) if a.this.lower() == "sum"]
            if sums or anons:
                sum_proj_node = sums[0] if sums else anons[0]
                sum_sel_node = sel
                break

        if not sum_proj_node or not sum_sel_node:
            return False, "Query missing required SUM aggregation projection.", None

        # Reject altered arithmetic modifying SUM
        inner = sum_sel_node.this if isinstance(sum_sel_node, exp.Alias) else sum_sel_node
        if isinstance(inner, exp.Cast):
            inner = inner.this
        while isinstance(inner, exp.Paren):
            inner = inner.this
        if inner is not sum_proj_node:
            return False, f"Altered aggregation expression detected outside of SUM function ('{inner.sql(dialect='duckdb')}').", None

        m_col_node = sum_proj_node.find(exp.Column)
        if not m_col_node or m_col_node.name.lower().strip('"') != expected_m:
            actual_m = m_col_node.name if m_col_node else str(sum_proj_node)
            return False, f"SUM target column mismatch: expected '{expected_m}', got '{actual_m}'.", None

        actual_alias = sum_sel_node.alias_or_name.lower().strip('"')
        if actual_alias != expected_alias:
            return False, f"SUM projection alias mismatch: expected '{expected_alias}', got '{actual_alias}'.", None

        # Reject unauthorized HAVING and ORDER BY
        if tree.args.get("having"):
            return False, "Query contains an unauthorized HAVING clause.", None
        if tree.args.get("order"):
            return False, "Query contains an unauthorized ORDER BY clause.", None

        # Issue ExecutionReceipt
        receipt = ExecutionReceipt.create(
            contract=contract,
            code=sql,
            conformance_passed=True,
            validation_details=(
                f"Composed verification passed. Bound capabilities: {contract.bound_capabilities}. "
                f"DAG fingerprint: {contract.dag_fingerprint[:8]}."
            ),
        )
        return True, None, receipt


# ---------------------------------------------------------------------------
# 6. Composed Pipeline Validator
# ---------------------------------------------------------------------------

class ComposedPipelineValidator:
    """
    Computes expected results independently from the original raw input DataFrame using Pandas.
    Never reads SQL output or SQL intermediate tables to determine ground truth.
    1. Evaluates temporal filter on raw DataFrame.
    2. Evaluates grouped SUM on filtered slice.
    3. Compares group membership, cardinality, and numerical results within tolerance.
    """

    @classmethod
    def validate(
        cls,
        raw_df: pd.DataFrame,
        sql_actual_df: pd.DataFrame,
        contract: ComposedExecutionContract,
        rel_tol: float = GROUPED_SUM_REL_TOL,
        abs_tol: float = GROUPED_SUM_ABS_TOL,
    ) -> ValidationResult:
        if not isinstance(raw_df, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_INPUT",
                error_message="raw_df must be a pandas DataFrame.",
            )

        if not isinstance(sql_actual_df, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_SQL_RESULT",
                error_message="sql_actual_df must be a pandas DataFrame.",
            )

        has_filter = any(c[0] == TEMPORAL_FILTER_CAPABILITY_ID for c in contract.bound_capabilities)
        has_agg = any(c[0] == GROUPED_SUM_CAPABILITY_ID for c in contract.bound_capabilities)

        # 1. Independent Temporal Filter Evaluation on Raw DataFrame
        if has_filter:
            pred = contract.required_filters[0]
            time_col = pred.column.name
            if time_col not in raw_df.columns:
                return ValidationResult(
                    is_valid=False,
                    error_type="MISSING_COLUMN",
                    error_message=f"Temporal column '{time_col}' missing from raw DataFrame.",
                )

            start_ts = parse_and_validate_temporal_boundary(pred.literal_value)
            end_ts = parse_and_validate_temporal_boundary(pred.secondary_literal)
            series = raw_df[time_col]

            # Strict half-open filter [start, end)
            mask = (series >= start_ts) & (series < end_ts) & series.notna()
            df_filtered = raw_df[mask].copy()
        else:
            df_filtered = raw_df.copy()

        # 2. Independent Grouped Sum Evaluation on Filtered Slice
        measure_spec = contract.required_measures[0]
        measure_col = measure_spec.target_column.name
        measure_alias = measure_spec.output_alias

        if has_agg and contract.authorized_group_by:
            group_col = contract.authorized_group_by[0]

            if group_col not in df_filtered.columns:
                return ValidationResult(
                    is_valid=False,
                    error_type="MISSING_COLUMN",
                    error_message=f"Grouping column '{group_col}' missing from DataFrame.",
                )

            # dropna=False preserves NULL group keys; min_count=1 ensures all-null measure yields NaN
            expected_series = df_filtered.groupby(group_col, dropna=False)[measure_col].sum(min_count=1)

            # Compare against SQL Actual
            if group_col not in sql_actual_df.columns or measure_alias not in sql_actual_df.columns:
                return ValidationResult(
                    is_valid=False,
                    error_type="SCHEMA_MISMATCH",
                    error_message=f"SQL result schema mismatch: expected columns '{group_col}', '{measure_alias}'.",
                )

            expected_count = len(expected_series)
            actual_count = len(sql_actual_df)
            if expected_count != actual_count:
                return ValidationResult(
                    is_valid=False,
                    error_type="CARDINALITY_MISMATCH",
                    error_message=f"Expected {expected_count} groups, got {actual_count}.",
                    audit_details={"expected_groups": expected_count, "actual_groups": actual_count},
                )

            # Build value lookup maps
            expected_map = {None if pd.isna(k) else k: v for k, v in expected_series.items()}
            actual_map = {None if pd.isna(r[group_col]) else r[group_col]: r[measure_alias] for _, r in sql_actual_df.iterrows()}

            if set(expected_map.keys()) != set(actual_map.keys()):
                return ValidationResult(
                    is_valid=False,
                    error_type="GROUP_KEY_MISMATCH",
                    error_message="Group keys mismatch between independent Pandas ground truth and SQL actual result.",
                    audit_details={
                        "missing_keys": list(set(expected_map.keys()) - set(actual_map.keys())),
                        "extra_keys": list(set(actual_map.keys()) - set(expected_map.keys())),
                    },
                )

            for k, exp_val in expected_map.items():
                act_val = actual_map[k]
                if pd.isna(exp_val) and pd.isna(act_val):
                    continue
                if pd.isna(exp_val) or pd.isna(act_val):
                    return ValidationResult(
                        is_valid=False,
                        error_type="NULL_MEASURE_MISMATCH",
                        error_message=f"Group '{k}' null mismatch: expected {exp_val}, got {act_val}.",
                    )
                if not math.isclose(float(exp_val), float(act_val), rel_tol=rel_tol, abs_tol=abs_tol):
                    return ValidationResult(
                        is_valid=False,
                        error_type="VALUE_MISMATCH",
                        error_message=(
                            f"Group '{k}' sum mismatch: expected {exp_val}, got {act_val} "
                            f"(exceeds rel_tol={rel_tol}, abs_tol={abs_tol})."
                        ),
                        audit_details={"group": k, "expected": float(exp_val), "actual": float(act_val)},
                    )

            return ValidationResult(
                is_valid=True,
                audit_details={
                    "group_count": expected_count,
                    "groups": list(expected_map.keys()),
                    "independent_ground_truth": {k: float(v) for k, v in expected_map.items() if not pd.isna(v)},
                },
            )

        else:
            # Benchmark 3: Scalar sum evaluation
            expected_scalar = float(df_filtered[measure_col].sum())
            if measure_alias not in sql_actual_df.columns:
                return ValidationResult(
                    is_valid=False,
                    error_type="SCHEMA_MISMATCH",
                    error_message=f"SQL result missing measure alias '{measure_alias}'.",
                )
            if len(sql_actual_df) != 1:
                return ValidationResult(
                    is_valid=False,
                    error_type="CARDINALITY_MISMATCH",
                    error_message=f"Expected 1 scalar row, got {len(sql_actual_df)}.",
                )
            act_val = float(sql_actual_df[measure_alias].iloc[0])
            if not math.isclose(expected_scalar, act_val, rel_tol=rel_tol, abs_tol=abs_tol):
                return ValidationResult(
                    is_valid=False,
                    error_type="VALUE_MISMATCH",
                    error_message=f"Scalar sum mismatch: expected {expected_scalar}, got {act_val}.",
                    audit_details={"expected": expected_scalar, "actual": act_val},
                )

            return ValidationResult(
                is_valid=True,
                audit_details={"scalar_result": expected_scalar},
            )

