"""MasterAIR Container and Deterministic DAG Integrity Engine with Derived Output Schemas."""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

from src.agent.air.schemas import (
    AIRLifecycleState,
    DataType,
    DimensionAuthorizationMatrix,
    FeasibilityStatus,
    GrainAssessment,
    LegacyContext,
)
from src.agent.air.operations import (
    AggregateOperation,
    AggregationFunction,
    AnalyticalOperation,
    DerivedOperation,
    FilterOperation,
    RankOperation,
)


class DAGValidationError(Exception):
    """Raised when an OperationDAG fails structural, semantic, or type integrity checks."""
    pass


class MasterAIR(BaseModel):
    """Canonical Analytical Intermediate Representation (AIR v2.1)."""
    air_version: str = "2.1.0"
    lifecycle_state: AIRLifecycleState = AIRLifecycleState.EXTRACTED
    raw_user_question: str
    metadata_summary: str
    feasibility_status: FeasibilityStatus = FeasibilityStatus.FEASIBLE
    clarification_prompt: Optional[str] = None
    grain_assessment: GrainAssessment
    dimensions: List[DimensionAuthorizationMatrix] = Field(default_factory=list)
    operations: List[AnalyticalOperation] = Field(default_factory=list)
    terminal_operation_ids: List[str] = Field(default_factory=list)
    requested_outputs: List[str] = Field(default_factory=list)
    legacy_context: Optional[LegacyContext] = None

    def validate_dag(self, base_table_schemas: Dict[str, Dict[str, DataType]]) -> List[AnalyticalOperation]:
        """Runs the deterministic DAGIntegrityValidator and returns topologically sorted operations."""
        sorted_ops = DAGIntegrityValidator.validate_and_sort(
            operations=self.operations,
            terminal_ids=self.terminal_operation_ids,
            base_table_schemas=base_table_schemas,
            dimensions=self.dimensions,
            requested_outputs=self.requested_outputs,
            lifecycle_state=self.lifecycle_state,
        )
        self.lifecycle_state = AIRLifecycleState.VALIDATED
        return sorted_ops


class DimensionAuthorizationEnforcer:
    """Enforces fail-closed authorization boundaries per dimension."""

    @classmethod
    def verify_action(
        cls,
        column_name: str,
        action: Literal["group_by", "partition_by", "order_by", "filter"],
        matrix: Dict[str, DimensionAuthorizationMatrix],
        operation_id: str,
        is_metric: bool = False,
    ) -> None:
        # If the column being ordered is semantically a metric, dimension authorization does not apply
        if action == "order_by" and is_metric:
            return

        # Fail-closed lookup: missing entry is immediately rejected
        if column_name not in matrix:
            raise DAGValidationError(
                f"Fail-Closed Authorization: Column '{column_name}' referenced in '{action}' for "
                f"operation '{operation_id}' has NO authorization record in Dimension Matrix."
            )

        auth = matrix[column_name]

        if action == "group_by" and not auth.allow_grouping:
            raise DAGValidationError(
                f"Unauthorized Grouping: Column '{column_name}' not authorized for GROUP BY "
                f"in operation '{operation_id}'."
            )

        if action == "partition_by" and not auth.allow_grouping:
            raise DAGValidationError(
                f"Unauthorized Partitioning: Column '{column_name}' not authorized for window partitioning "
                f"in operation '{operation_id}'."
            )

        if action == "order_by" and not auth.allow_order_by:
            raise DAGValidationError(
                f"Unauthorized Dimension Ordering: Column '{column_name}' not authorized for sorting "
                f"in operation '{operation_id}'."
            )

        if action == "filter" and not auth.allow_filter_predicate:
            raise DAGValidationError(
                f"Unauthorized Filtering: Column '{column_name}' not authorized in filter predicate "
                f"for operation '{operation_id}'."
            )


class DAGIntegrityValidator:
    """Deterministic validation of structural topology, dataset bindings, typing, and derived schemas."""

    @classmethod
    def validate_and_sort(
        cls,
        operations: List[AnalyticalOperation],
        terminal_ids: List[str],
        base_table_schemas: Dict[str, Dict[str, DataType]],
        dimensions: List[DimensionAuthorizationMatrix],
        requested_outputs: List[str],
        lifecycle_state: AIRLifecycleState = AIRLifecycleState.PLANNED,
    ) -> List[AnalyticalOperation]:
        # ---------------------------------------------------------------------
        # Invariant 1: Non-Empty Graph for PLANNED and VALIDATED Lifecycles
        # ---------------------------------------------------------------------
        if not operations:
            if lifecycle_state == AIRLifecycleState.EXTRACTED:
                return []
            raise DAGValidationError(
                f"Invariant 1: Operation graph cannot be empty when validating a plan in '{lifecycle_state.value}' state."
            )

        # ---------------------------------------------------------------------
        # Invariant 2: Unique IDs, Output Aliases & Base Table Collision Check
        # ---------------------------------------------------------------------
        op_map: Dict[str, AnalyticalOperation] = {}
        output_aliases: Set[str] = set()
        producers: Dict[str, str] = {}

        for op in operations:
            if op.operation_id in op_map:
                raise DAGValidationError(f"Invariant 2: Duplicate operation_id '{op.operation_id}'.")
            if op.output_alias in base_table_schemas:
                raise DAGValidationError(
                    f"Invariant 2: Output alias '{op.output_alias}' collides with base table name."
                )
            if op.output_alias in output_aliases:
                raise DAGValidationError(
                    f"Invariant 2: Duplicate output_alias '{op.output_alias}' produced by multiple operations."
                )
            if len(op.depends_on) != len(set(op.depends_on)):
                raise DAGValidationError(
                    f"Invariant 2: Duplicate dependencies declared in operation '{op.operation_id}'."
                )

            op_map[op.operation_id] = op
            output_aliases.add(op.output_alias)
            producers[op.output_alias] = op.operation_id

        # ---------------------------------------------------------------------
        # Invariant 3: Valid Dependency References
        # ---------------------------------------------------------------------
        for op in operations:
            for dep in op.depends_on:
                if dep not in op_map:
                    raise DAGValidationError(
                        f"Invariant 3: Operation '{op.operation_id}' depends on missing operation '{dep}'."
                    )

        # ---------------------------------------------------------------------
        # Invariant 4: Terminal Operations Consistency
        # ---------------------------------------------------------------------
        if not terminal_ids:
            raise DAGValidationError("Invariant 4: At least one terminal_operation_id must be declared.")
        if len(terminal_ids) != len(set(terminal_ids)):
            raise DAGValidationError("Invariant 4: Duplicate entries in terminal_operation_ids.")

        for t_id in terminal_ids:
            if t_id not in op_map:
                raise DAGValidationError(f"Invariant 4: Terminal ID '{t_id}' does not exist in graph.")
            if not op_map[t_id].is_terminal:
                raise DAGValidationError(
                    f"Invariant 4: Operation '{t_id}' is listed in terminal_ids but has is_terminal=False."
                )

        # ---------------------------------------------------------------------
        # Invariant 5: Cycle Detection & Kahn's Topological Sort
        # ---------------------------------------------------------------------
        in_degree: Dict[str, int] = {op.operation_id: len(op.depends_on) for op in operations}
        adj_list: Dict[str, List[str]] = {op.operation_id: [] for op in operations}
        for op in operations:
            for dep in op.depends_on:
                adj_list[dep].append(op.operation_id)

        queue: List[str] = [op_id for op_id, deg in in_degree.items() if deg == 0]
        sorted_ops: List[AnalyticalOperation] = []

        while queue:
            curr = queue.pop(0)
            sorted_ops.append(op_map[curr])
            for child in adj_list[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_ops) != len(operations):
            raise DAGValidationError("Invariant 5: Circular dependency detected in operation graph.")

        # ---------------------------------------------------------------------
        # Invariant 6: Independent Orphan Detection
        # ---------------------------------------------------------------------
        reverse_adj: Dict[str, List[str]] = {op.operation_id: list(op.depends_on) for op in operations}
        reachable_to_terminal: Set[str] = set()

        def _trace(node_id: str):
            if node_id in reachable_to_terminal:
                return
            reachable_to_terminal.add(node_id)
            for p in reverse_adj.get(node_id, []):
                _trace(p)

        for t_id in terminal_ids:
            _trace(t_id)

        orphans = set(op_map.keys()) - reachable_to_terminal
        if orphans:
            raise DAGValidationError(
                f"Invariant 6: Orphaned operations detected that do not reach terminal outputs: {orphans}"
            )

        # ---------------------------------------------------------------------
        # Invariant 7: Exact Dependency-to-Input Consistency & No Unused Dependencies
        # ---------------------------------------------------------------------
        for op in sorted_ops:
            consumed_producer_ops: Set[str] = set()
            for ds in op.input_datasets:
                if ds not in base_table_schemas:
                    if ds not in producers:
                        raise DAGValidationError(
                            f"Invariant 7: Dataset '{ds}' consumed by '{op.operation_id}' has no producer."
                        )
                    producer_id = producers[ds]
                    consumed_producer_ops.add(producer_id)
                    if producer_id not in op.depends_on:
                        raise DAGValidationError(
                            f"Invariant 7: Operation '{op.operation_id}' reads view '{ds}' from "
                            f"'{producer_id}' without declaring it in depends_on."
                        )

            # Assert no unused declared dependencies
            unused_deps = set(op.depends_on) - consumed_producer_ops
            if unused_deps:
                raise DAGValidationError(
                    f"Invariant 7: Operation '{op.operation_id}' declares unused dependencies in depends_on: {unused_deps}."
                )

        # ---------------------------------------------------------------------
        # Invariant 8 & 9: Column Existence, Real Typing, and Derived Schema Validation
        # ---------------------------------------------------------------------
        available_schemas: Dict[str, Dict[str, DataType]] = dict(base_table_schemas)
        dim_matrix = {d.column_name: d for d in dimensions}

        for op in sorted_ops:
            combined_input_schema: Dict[str, DataType] = {}
            for ds in op.input_datasets:
                combined_input_schema.update(available_schemas[ds])

            derived_schema: Dict[str, DataType] = {}

            if isinstance(op, FilterOperation):
                # Validates predicates and preserves exact input schema
                for pred in op.predicates:
                    col_name = pred.column.name
                    if col_name not in combined_input_schema:
                        raise DAGValidationError(
                            f"Invariant 8: Filter column '{col_name}' does not exist in inputs for '{op.operation_id}'."
                        )
                    DimensionAuthorizationEnforcer.verify_action(col_name, "filter", dim_matrix, op.operation_id)
                derived_schema = dict(combined_input_schema)

            elif isinstance(op, AggregateOperation):
                # 1. Group By Columns
                for grp in op.group_by_columns:
                    if grp.name not in combined_input_schema:
                        raise DAGValidationError(
                            f"Invariant 8: GROUP BY column '{grp.name}' not in inputs for '{op.operation_id}'."
                        )
                    DimensionAuthorizationEnforcer.verify_action(grp.name, "group_by", dim_matrix, op.operation_id)
                    derived_schema[grp.name] = combined_input_schema[grp.name]

                # 2. Aggregation Measures
                for m in op.measures:
                    col_name = m.target_column.name
                    if col_name not in combined_input_schema:
                        raise DAGValidationError(
                            f"Invariant 8: Measure column '{col_name}' not in inputs for '{op.operation_id}'."
                        )
                    col_type = combined_input_schema[col_name]
                    if m.function in {AggregationFunction.SUM, AggregationFunction.AVG}:
                        if col_type not in {DataType.INTEGER, DataType.FLOAT}:
                            raise DAGValidationError(
                                f"Invariant 8: Incompatible data type for {m.function.value.upper()}: "
                                f"Column '{col_name}' is '{col_type.value}', expected numeric."
                            )
                    out_type = DataType.INTEGER if m.function in {AggregationFunction.COUNT_DISTINCT, AggregationFunction.COUNT} else DataType.FLOAT
                    derived_schema[m.output_alias] = out_type

                # 3. Derived Measures
                for der in op.derived_measures:
                    for ref in der.referenced_columns:
                        if ref not in derived_schema:
                            raise DAGValidationError(
                                f"Invariant 8: Derived expression in '{op.operation_id}' references unknown alias '{ref}'."
                            )
                    derived_schema[der.output_alias] = DataType.FLOAT

            elif isinstance(op, RankOperation):
                # Validates ORDER BY and preserves input schema projection
                for crit in op.order_by:
                    col_name = crit.column.name
                    if col_name not in combined_input_schema:
                        raise DAGValidationError(
                            f"Invariant 8: ORDER BY column '{col_name}' not in inputs for '{op.operation_id}'."
                        )
                    # Semantically check: is it a metric or dimension?
                    DimensionAuthorizationEnforcer.verify_action(
                        col_name, "order_by", dim_matrix, op.operation_id, is_metric=crit.is_metric
                    )

                for part in op.group_by_partitions:
                    if part.name not in combined_input_schema:
                        raise DAGValidationError(
                            f"Invariant 8: Partition column '{part.name}' not in inputs for '{op.operation_id}'."
                        )
                    DimensionAuthorizationEnforcer.verify_action(part.name, "partition_by", dim_matrix, op.operation_id)

                derived_schema = dict(combined_input_schema)

            elif isinstance(op, DerivedOperation):
                # Preserves inputs or projects derived expressions
                derived_schema = dict(combined_input_schema)
                for der in op.expressions:
                    for ref in der.referenced_columns:
                        if ref not in combined_input_schema:
                            raise DAGValidationError(
                                f"Invariant 8: Derived expression in '{op.operation_id}' references unknown column/alias '{ref}'."
                            )
                    derived_schema[der.output_alias] = DataType.FLOAT

            # Assert declared schema matches derived schema exactly
            if op.output_schema != derived_schema:
                raise DAGValidationError(
                    f"Invariant 9: Output schema mismatch in operation '{op.operation_id}'. "
                    f"Declared: {op.output_schema}, Derived: {derived_schema}."
                )

            available_schemas[op.output_alias] = derived_schema

        # ---------------------------------------------------------------------
        # Invariant 10: Non-Trivial Terminal Output Verification
        # ---------------------------------------------------------------------
        terminal_available_columns: Dict[str, DataType] = {}
        for t_id in terminal_ids:
            terminal_available_columns.update(op_map[t_id].output_schema)

        requirement_synonym_map = {
            "total unique orders": ["unique_orders", "order_count", "distinct_order_id", "total_orders", "count_distinct_order_id"],
            "total orders": ["unique_orders", "order_count", "distinct_order_id", "total_orders", "count_distinct_order_id"],
            "average order value": ["aov", "average_order_value"],
            "total revenue": ["total_revenue", "total_sales", "sales", "revenue", "sum_sales", "sum_revenue"],
            "total sales": ["total_revenue", "total_sales", "sales", "revenue", "sum_sales", "sum_revenue"],
            "total profit": ["total_profit", "profit", "sum_profit"],
        }

        for req in requested_outputs:
            clean_req = req.strip().lower()
            valid_aliases = requirement_synonym_map.get(clean_req, [clean_req.replace(" ", "_")])
            matched = any(alias in terminal_available_columns for alias in valid_aliases)
            if not matched:
                raise DAGValidationError(
                    f"Invariant 10: Requested output requirement '{req}' is NOT satisfied by terminal operations. "
                    f"Expected one of {valid_aliases}, but terminal schema provides: {list(terminal_available_columns.keys())}."
                )

        return sorted_ops

