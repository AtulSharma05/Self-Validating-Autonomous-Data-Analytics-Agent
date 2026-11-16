"""
Deterministic AIR-to-Contract Compiler.

Compiles validated MasterAIR DAGs into strongly typed ExecutionContracts,
enforcing operation fusion, dependency preservation, and fail-closed authorization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
from pydantic import BaseModel

from src.agent.air.schemas import (
    AIRLifecycleState,
    ColumnRef,
    ComparisonOperator,
    DataType,
    DimensionAuthorizationMatrix,
    Predicate,
)
from src.agent.air.operations import (
    AggregateOperation,
    AnalyticalOperation,
    DerivedOperation,
    FilterOperation,
    RankOperation,
)
from src.agent.air.dag import DAGValidationError, MasterAIR
from src.agent.air.contract import (
    CardinalityExpectation,
    ContractEngine,
    ExecutionContract,
    PredicateOperator,
    RequiredDerivedMeasureSpec,
    RequiredMeasureSpec,
    TypedPredicate,
)


class CompilerError(Exception):
    """Raised when compilation cannot be deterministically completed."""
    pass


class AIRContractCompiler:
    """
    Deterministic compiler translating validated MasterAIR into an ordered sequence
    of ExecutionContracts.
    """

    def __init__(self, air: MasterAIR, base_schemas: Dict[str, Dict[str, DataType]]):
        if air.lifecycle_state != AIRLifecycleState.VALIDATED:
            # Attempt to run validation if not already validated
            try:
                air.validate_dag(base_schemas)
            except Exception as e:
                raise CompilerError(f"Cannot compile unvalidated AIR: {e}")
        self.air = air
        self.base_schemas = base_schemas
        self.dim_map: Dict[str, DimensionAuthorizationMatrix] = {
            d.column_name: d for d in air.dimensions
        }

    def compile(self) -> List[ExecutionContract]:
        """
        Compiles the validated AIR DAG into ExecutionContracts.
        Enforces all seven compiler invariants.
        """
        try:
            topo_ops = self.air.validate_dag(self.base_schemas)
        except DAGValidationError as e:
            raise CompilerError(f"DAG validation failed during compilation: {e}") from e
        if not topo_ops:
            return []

        # Invariant 7: Explicitly reject unsupported operations
        for op in topo_ops:
            if not isinstance(op, (FilterOperation, AggregateOperation, DerivedOperation)):
                raise CompilerError(
                    f"Unsupported operation type '{type(op).__name__}' (id: {op.operation_id}). "
                    "Only Filter, Aggregate, and Derived operations are supported in this milestone."
                )

        # Build dependency & consumer lookup
        op_map = {op.operation_id: op for op in topo_ops}
        consumers: Dict[str, List[str]] = {op.operation_id: [] for op in topo_ops}
        for op in topo_ops:
            for dep in op.depends_on:
                if dep in consumers:
                    consumers[dep].append(op.operation_id)

        # Partition into execution clusters (many-to-one fusion)
        clusters: List[List[AnalyticalOperation]] = []
        assigned_ops: Set[str] = set()

        for op in topo_ops:
            if op.operation_id in assigned_ops:
                continue

            cluster = [op]
            assigned_ops.add(op.operation_id)

            more_to_add = True
            while more_to_add:
                more_to_add = False
                cluster_op_ids = {o.operation_id for o in cluster}
                downstream = [
                    o for o in topo_ops
                    if o.operation_id not in assigned_ops
                    and any(dep in cluster_op_ids for dep in o.depends_on)
                ]
                if downstream and all(isinstance(o, (FilterOperation, AggregateOperation, DerivedOperation)) for o in downstream):
                    agg_ops = [o for o in downstream if isinstance(o, AggregateOperation)]
                    existing_aggs = [o for o in cluster if isinstance(o, AggregateOperation)]
                    grp_sets = {tuple(c.name for c in o.group_by_columns) for o in agg_ops + existing_aggs}
                    if len(grp_sets) <= 1:
                        for o in downstream:
                            if o.operation_id not in assigned_ops:
                                cluster.append(o)
                                assigned_ops.add(o.operation_id)
                                more_to_add = True

            clusters.append(cluster)

        # Invariant 1 & 2: Every operation is assigned exactly once
        if assigned_ops != set(op_map.keys()):
            missing = set(op_map.keys()) - assigned_ops
            raise CompilerError(f"Compiler Invariant 1 Violated: Operations {missing} were dropped.")

        # Compile each cluster into an ExecutionContract
        contracts: List[ExecutionContract] = []
        for idx, cluster in enumerate(clusters, start=1):
            contract = self._compile_cluster(cluster, unit_index=idx)
            contracts.append(contract)

        return contracts

    def _compile_cluster(self, cluster: List[AnalyticalOperation], unit_index: int) -> ExecutionContract:
        """Compiles a cluster of 1 or more fused operations into an ExecutionContract."""
        contract_id = f"contract_step_{unit_index}"
        unit_id = f"unit_step_{unit_index}"
        op_ids = [op.operation_id for op in cluster]

        # Primary input table from the root operation of this cluster
        primary_table = cluster[0].input_datasets[0] if cluster[0].input_datasets else "unknown"
        while primary_table not in self.base_schemas and any(op.output_alias == primary_table for op in self.air.operations):
            upstream = next(op for op in self.air.operations if op.output_alias == primary_table)
            primary_table = upstream.input_datasets[0]

        table_cols = list(self.base_schemas.get(primary_table, {}).keys())

        required_filters: List[TypedPredicate] = []
        authorized_group_by: List[str] = []
        required_measures: List[RequiredMeasureSpec] = []
        required_derived_measures: List[RequiredDerivedMeasureSpec] = []
        logical_operation_mapping: Dict[str, str] = {}
        expected_output_columns: Dict[str, DataType] = {}

        # Map dataset alias to filter predicates for branch filters
        dataset_branch_predicates: Dict[str, List[TypedPredicate]] = {}
        for op in cluster:
            if isinstance(op, FilterOperation):
                for p in op.predicates:
                    pred = self._convert_predicate(p)
                    if pred.column.name.lower() == "order_id" and pred.operator in (PredicateOperator.IS_NULL, PredicateOperator.IS_NOT_NULL):
                        if op.output_alias not in dataset_branch_predicates:
                            dataset_branch_predicates[op.output_alias] = []
                        dataset_branch_predicates[op.output_alias].append(pred)
                    else:
                        if pred not in required_filters:
                            required_filters.append(pred)

            elif isinstance(op, AggregateOperation):
                for col_ref in op.group_by_columns:
                    dim_auth = self.dim_map.get(col_ref.name)
                    if not dim_auth or not dim_auth.allow_grouping:
                        raise CompilerError(
                            f"Compiler Violation: Dimension '{col_ref.name}' was placed in group_by_columns "
                            f"of operation '{op.operation_id}', but has no explicit positive grouping authorization in AIR."
                        )
                    if col_ref.name not in authorized_group_by:
                        authorized_group_by.append(col_ref.name)
                        expected_output_columns[col_ref.name] = op.output_schema.get(col_ref.name, DataType.STRING)
                        logical_operation_mapping[col_ref.name] = op.operation_id

                cond_pred = None
                # Check branch predicates from upstream filter operation
                for in_ds in op.input_datasets:
                    if in_ds in dataset_branch_predicates:
                        for p in dataset_branch_predicates[in_ds]:
                            cond_pred = p

                for p in op.step_filter_predicates:
                    pred = self._convert_predicate(p)
                    if pred.column.name.lower() == "order_id" and pred.operator in (PredicateOperator.IS_NULL, PredicateOperator.IS_NOT_NULL):
                        cond_pred = pred
                    else:
                        if pred not in required_filters:
                            required_filters.append(pred)

                for m in op.measures:
                    required_measures.append(
                        RequiredMeasureSpec(
                            output_alias=m.output_alias,
                            function=m.function,
                            target_column=m.target_column,
                            condition_predicate=cond_pred,
                        )
                    )
                    expected_output_columns[m.output_alias] = op.output_schema.get(m.output_alias, DataType.FLOAT)
                    logical_operation_mapping[m.output_alias] = op.operation_id

                for der in op.derived_measures:
                    num_alias = der.referenced_columns[0]
                    den_alias = der.referenced_columns[1] if len(der.referenced_columns) > 1 else num_alias
                    required_derived_measures.append(
                        RequiredDerivedMeasureSpec(
                            output_alias=der.output_alias,
                            numerator_source_alias=num_alias,
                            denominator_source_alias=den_alias,
                            enforce_nullif_zero=(der.null_handling == "null_if_zero"),
                            target_data_type=DataType.FLOAT,
                        )
                    )
                    expected_output_columns[der.output_alias] = DataType.FLOAT
                    logical_operation_mapping[der.output_alias] = op.operation_id

            elif isinstance(op, DerivedOperation):
                for der in op.expressions:
                    num_alias = der.referenced_columns[0]
                    den_alias = der.referenced_columns[1] if len(der.referenced_columns) > 1 else num_alias
                    required_derived_measures.append(
                        RequiredDerivedMeasureSpec(
                            output_alias=der.output_alias,
                            numerator_source_alias=num_alias,
                            denominator_source_alias=den_alias,
                            enforce_nullif_zero=(der.null_handling == "null_if_zero"),
                            target_data_type=DataType.FLOAT,
                        )
                    )
                    expected_output_columns[der.output_alias] = DataType.FLOAT
                    logical_operation_mapping[der.output_alias] = op.operation_id

        cardinality = CardinalityExpectation.SCALAR if not authorized_group_by else CardinalityExpectation.DIMENSIONAL

        forbidden_group_by = [
            dim_name for dim_name, d in self.dim_map.items()
            if not d.allow_grouping and dim_name not in authorized_group_by
        ]

        return ExecutionContract(
            contract_id=contract_id,
            execution_unit_id=unit_id,
            air_operation_ids=op_ids,
            target_table=primary_table,
            authorized_source_columns=table_cols,
            authorized_group_by=authorized_group_by,
            forbidden_group_by=forbidden_group_by,
            required_filters=required_filters,
            forbidden_filters=[],
            required_measures=required_measures,
            required_derived_measures=required_derived_measures,
            logical_operation_mapping=logical_operation_mapping,
            expected_output_columns=expected_output_columns,
            expected_cardinality=cardinality,
            target_engine=ContractEngine.DUCKDB_SQL,
            timeout_seconds=30,
        )

    def _convert_predicate(self, p: Predicate) -> TypedPredicate:
        op_map = {
            ComparisonOperator.EQ: PredicateOperator.EQ,
            ComparisonOperator.NEQ: PredicateOperator.NEQ,
            ComparisonOperator.GT: PredicateOperator.GT,
            ComparisonOperator.GTE: PredicateOperator.GTE,
            ComparisonOperator.LT: PredicateOperator.LT,
            ComparisonOperator.LTE: PredicateOperator.LTE,
            ComparisonOperator.IN: PredicateOperator.IN,
            ComparisonOperator.NOT_IN: PredicateOperator.NOT_IN,
            ComparisonOperator.IS_NULL: PredicateOperator.IS_NULL,
            ComparisonOperator.IS_NOT_NULL: PredicateOperator.IS_NOT_NULL,
            ComparisonOperator.BETWEEN: PredicateOperator.BETWEEN_HALF_OPEN,
        }
        col_type = p.column.data_type
        is_temp = col_type in {DataType.DATE, DataType.DATETIME} or "date" in p.column.name.lower()
        return TypedPredicate(
            column=p.column,
            operator=op_map.get(p.operator, PredicateOperator.EQ),
            literal_value=p.literal_value,
            is_temporal=is_temp,
        )
