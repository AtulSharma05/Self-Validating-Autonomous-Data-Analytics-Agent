"""
Built-in Atomic Capability: aggregate.grouped_sum (v1.0.0).

Enforces strict grouped aggregation semantics:
- GROUP BY only the explicitly authorized grouping column.
- SUM only over the explicitly authorized numeric measure.
- Strict dimension authorization: candidate dimensions never become implicit authorization.
- Correct null handling consistent with DuckDB and Pandas semantics:
  * NULL group keys form an explicit distinct group (dropna=False).
  * NULL measure values are ignored; all-null groups yield NULL/NaN (min_count=1).
- Non-numeric measures and unauthorized dimensions fail closed.
- Floating-point comparisons enforce documented tolerance (rel_tol=1e-5, abs_tol=1e-4).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import sqlglot
from sqlglot import exp

from src.agent.air.schemas import (
    ColumnRef,
    DataType,
    DimensionAuthorizationMatrix,
)
from src.agent.air.operations import (
    AggregationFunction,
    AggregateOperation,
    AnalyticalOperation,
)
from src.agent.air.contract import (
    CardinalityExpectation,
    RequiredMeasureSpec,
)
from src.agent.air.registry.models import (
    CapabilityAuthorizationPolicy,
    CapabilityKind,
    DataGrain,
    PortCardinality,
    PortDefinition,
    PortPayloadType,
    ResourceConstraints,
    SemanticRole,
)
from src.agent.air.registry.protocols import (
    ContractFragment,
    CapabilityCompilerHandler,
    CapabilityVerifierHandler,
    IndependentValidatorHandler,
)
from src.agent.air.registry.registry import AnalyticalCapabilityRegistry
from src.validation.validator import ValidationResult


GROUPED_SUM_CAPABILITY_ID = "aggregate.grouped_sum"
GROUPED_SUM_VERSION = "1.0.0"

# Explicit documented tolerance for floating-point comparisons
GROUPED_SUM_REL_TOL = 1e-5
GROUPED_SUM_ABS_TOL = 1e-4


@dataclass(frozen=True)
class GroupedSumCompilerHandler:
    """
    Compiler handler for aggregate.grouped_sum.
    Compiles an AggregateOperation into an immutable, validated ContractFragment.
    Enforces strict dimension grouping authorization and numeric measure type guards.
    """
    name: str = "GroupedSumCompiler"
    allowed_measure_types: Tuple[DataType, ...] = (DataType.FLOAT, DataType.INTEGER)
    allowed_group_types: Tuple[DataType, ...] = (DataType.STRING, DataType.INTEGER, DataType.DATE)

    def compile_fragment(
        self,
        operation: AnalyticalOperation,
        dimensions: Dict[str, DimensionAuthorizationMatrix],
        base_schemas: Dict[str, Dict[str, DataType]],
    ) -> ContractFragment:
        if not isinstance(operation, AggregateOperation):
            raise TypeError(
                f"GroupedSumCompiler expected AggregateOperation, got '{type(operation).__name__}'."
            )

        if not operation.input_datasets:
            raise ValueError(f"Operation '{operation.operation_id}' specifies no input_datasets.")

        source_table = operation.input_datasets[0]
        if source_table not in base_schemas:
            raise KeyError(
                f"Source table '{source_table}' for operation '{operation.operation_id}' "
                "not found in base schemas."
            )

        table_schema = base_schemas[source_table]

        # 1. Grouping Column Validation
        if len(operation.group_by_columns) != 1:
            raise ValueError(
                f"Operation '{operation.operation_id}' must specify exactly 1 grouping column "
                f"for aggregate.grouped_sum, got {len(operation.group_by_columns)}."
            )

        group_col_ref = operation.group_by_columns[0]
        group_col = group_col_ref.name

        if group_col not in table_schema:
            raise KeyError(
                f"Grouping column '{group_col}' does not exist in schema of '{source_table}'."
            )

        group_dtype = table_schema[group_col]
        if group_dtype not in self.allowed_group_types:
            raise TypeError(
                f"Grouping column '{group_col}' has unsupported type '{group_dtype}'. "
                f"Authorizes only: {tuple(t.value for t in self.allowed_group_types)}."
            )

        # Dimension Authorization Check (Fail-Closed: candidate dimensions are not implicit authorization)
        dim_auth = dimensions.get(group_col)
        if not dim_auth or not dim_auth.allow_grouping:
            raise PermissionError(
                f"Security Authorization Failure: Dimension '{group_col}' is not authorized "
                "for grouping (allow_grouping is False or not explicitly granted). "
                "Candidate dimensions cannot serve as implicit authorization."
            )

        # 2. Measure Column Validation
        if len(operation.measures) != 1:
            raise ValueError(
                f"Operation '{operation.operation_id}' must specify exactly 1 measure "
                f"for aggregate.grouped_sum, got {len(operation.measures)}."
            )

        measure_spec = operation.measures[0]
        if measure_spec.function != AggregationFunction.SUM:
            raise ValueError(
                f"Invalid aggregation function '{measure_spec.function}' on measure '{measure_spec.output_alias}'. "
                "aggregate.grouped_sum authorizes only SUM."
            )

        measure_col = measure_spec.target_column.name
        if measure_col not in table_schema:
            raise KeyError(
                f"Measure column '{measure_col}' does not exist in schema of '{source_table}'."
            )

        measure_dtype = table_schema[measure_col]
        if measure_dtype not in self.allowed_measure_types:
            raise TypeError(
                f"Measure column '{measure_col}' has non-numeric type '{measure_dtype}'. "
                f"aggregate.grouped_sum authorizes only numeric measures: {tuple(t.value for t in self.allowed_measure_types)}."
            )

        # 3. Reject Unauthorized Projections, Filters, or Derived Calculations
        if getattr(operation, "derived_measures", None):
            raise ValueError(
                f"Operation '{operation.operation_id}' specifies derived measures, "
                "which are unauthorized for atomic aggregate.grouped_sum."
            )

        if getattr(operation, "step_filter_predicates", None):
            raise ValueError(
                f"Operation '{operation.operation_id}' specifies step_filter_predicates, "
                "which are unauthorized for atomic aggregate.grouped_sum."
            )

        output_alias = measure_spec.output_alias
        req_measure = RequiredMeasureSpec(
            output_alias=output_alias,
            function=AggregationFunction.SUM,
            target_column=ColumnRef(name=measure_col, data_type=measure_dtype, table_name=source_table),
        )

        expected_outputs = (
            (group_col, group_dtype),
            (output_alias, DataType.FLOAT),
        )

        return ContractFragment(
            operation_id=operation.operation_id,
            capability_id=GROUPED_SUM_CAPABILITY_ID,
            capability_version=GROUPED_SUM_VERSION,
            source_table=source_table,
            authorized_source_columns=(group_col, measure_col),
            authorized_group_by=(group_col,),
            required_measures=(req_measure,),
            expected_output_columns=expected_outputs,
            cardinality=CardinalityExpectation.DIMENSIONAL,
            logical_dependencies=tuple(operation.depends_on),
        )


@dataclass(frozen=True)
class GroupedSumVerifierHandler:
    """
    Verifier handler for aggregate.grouped_sum.
    Statically verifies the SQLGlot AST SELECT statement:
    - Root FROM matches source table (no unauthorized JOINs).
    - GROUP BY matches exact authorized grouping column.
    - Projection schema matches exactly (group dimension + SUM(measure)).
    - Function is strictly SUM without altering arithmetic.
    - No unauthorized WHERE, HAVING, or ORDER BY clauses.
    """
    name: str = "GroupedSumVerifier"

    def verify_ast_fragment(
        self,
        scope_expression: exp.Expression,
        fragment: ContractFragment,
    ) -> Tuple[bool, Optional[str]]:
        if not isinstance(scope_expression, exp.Select):
            return False, f"Expected root SELECT statement, got '{type(scope_expression).__name__}'."

        # 1. Source Table & JOIN Verification
        from_node = scope_expression.find(exp.From)
        if not from_node:
            return False, "Query has no FROM clause."

        table_node = from_node.find(exp.Table)
        if not table_node or table_node.name.lower().strip('"') != fragment.source_table.lower().strip('"'):
            actual_tbl = table_node.name if table_node else None
            return False, f"Source table mismatch: expected '{fragment.source_table}', got '{actual_tbl}'."

        joins = scope_expression.args.get("joins")
        if joins:
            return False, "Query contains unauthorized JOIN clauses."

        # 2. GROUP BY Verification
        group_clause = scope_expression.args.get("group")
        if not group_clause:
            return False, "Query is missing required GROUP BY clause for grouped_sum capability."

        group_exprs = group_clause.expressions
        if len(group_exprs) != 1:
            return False, f"Expected exactly 1 GROUP BY expression, got {len(group_exprs)}."

        expected_grp = fragment.authorized_group_by[0].lower().strip('"')
        grp_col_node = group_exprs[0].this if isinstance(group_exprs[0], exp.Column) else group_exprs[0].find(exp.Column)
        if not grp_col_node or grp_col_node.name.lower().strip('"') != expected_grp:
            actual_grp = grp_col_node.name if grp_col_node else str(group_exprs[0])
            return False, f"GROUP BY column mismatch: expected '{expected_grp}', got '{actual_grp}'."

        # 3. Unauthorized Clauses Verification
        where_clause = scope_expression.args.get("where")
        if where_clause and not fragment.required_filters and not fragment.permitted_filters:
            return False, "Query contains an unauthorized WHERE clause."

        having_clause = scope_expression.args.get("having")
        if having_clause and not fragment.authorized_having:
            return False, "Query contains an unauthorized HAVING clause."

        order_clause = scope_expression.args.get("order")
        if order_clause:
            return False, "Query contains an unauthorized ORDER BY clause."

        # 4. Projections Verification
        selects = scope_expression.selects
        if len(selects) != 2:
            return False, f"Expected exactly 2 projected columns, got {len(selects)}."

        expected_measure = fragment.required_measures[0].target_column.name.lower().strip('"')
        expected_alias = fragment.required_measures[0].output_alias.lower().strip('"')

        # Identify group projection and sum projection
        group_proj_found = False
        sum_proj_node: Optional[exp.Expression] = None
        sum_sel_node: Optional[exp.Expression] = None

        for sel in selects:
            while isinstance(sel, exp.Paren):
                sel = sel.this

            # Check if this projection is the grouping column
            if isinstance(sel, exp.Column) or (isinstance(sel, exp.Alias) and isinstance(sel.this, exp.Column)):
                col = sel if isinstance(sel, exp.Column) else sel.this
                if col.name.lower().strip('"') == expected_grp:
                    group_proj_found = True
                    continue

            # Check if this projection is the SUM aggregation
            sums = list(sel.find_all(exp.Sum))
            anons = [a for a in sel.find_all(exp.Anonymous) if a.this.lower() == "sum"]
            if sums or anons:
                sum_proj_node = sums[0] if sums else anons[0]
                sum_sel_node = sel

        if not group_proj_found:
            return False, f"Projected columns do not contain the authorized grouping column '{expected_grp}'."

        if not sum_proj_node or not sum_sel_node:
            return False, "Query missing required SUM aggregation projection."

        # Check for nested or altered expressions modifying aggregation
        if isinstance(sum_sel_node, exp.Alias):
            inner_expr = sum_sel_node.this
        else:
            inner_expr = sum_sel_node

        while isinstance(inner_expr, exp.Paren):
            inner_expr = inner_expr.this

        if inner_expr is not sum_proj_node:
            return False, (
                "Altered or nested aggregation expression detected outside of SUM function "
                f"('{inner_expr.sql(dialect='duckdb')}'). Arithmetic modification is prohibited."
            )

        # Check target column inside SUM
        measure_col_node = sum_proj_node.find(exp.Column)
        if not measure_col_node or measure_col_node.name.lower().strip('"') != expected_measure:
            actual_m = measure_col_node.name if measure_col_node else str(sum_proj_node)
            return False, f"SUM target column mismatch: expected '{expected_measure}', got '{actual_m}'."

        # Check output alias
        actual_alias = sum_sel_node.alias_or_name.lower().strip('"')
        if actual_alias != expected_alias:
            return False, f"SUM projection alias mismatch: expected '{expected_alias}', got '{actual_alias}'."

        # Reject any other aggregate functions
        for fn in (exp.Avg, exp.Count, exp.Min, exp.Max):
            if list(scope_expression.find_all(fn)):
                return False, f"Unauthorized aggregation function '{fn.__name__}' detected in query."

        return True, None


@dataclass(frozen=True)
class GroupedSumValidatorHandler:
    """
    Independent validator handler for aggregate.grouped_sum.
    Calculates expected grouped sums independently from the original input DataFrame using Pandas.
    Preserves null group keys and handles null measure values consistently with SQL.
    Floating-point comparisons enforce documented tolerance (rel_tol=1e-5, abs_tol=1e-4).
    """
    name: str = "GroupedSumValidator"
    rel_tol: float = GROUPED_SUM_REL_TOL
    abs_tol: float = GROUPED_SUM_ABS_TOL

    def validate_metric(
        self,
        operation: AnalyticalOperation,
        df_slice: pd.DataFrame,
        sql_actual_value: Any,
    ) -> ValidationResult:
        if not isinstance(df_slice, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_INPUT",
                error_message=f"df_slice must be a pandas DataFrame, got '{type(df_slice).__name__}'.",
            )

        if not isinstance(operation, AggregateOperation):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_OPERATION",
                error_message=f"Expected AggregateOperation, got '{type(operation).__name__}'.",
            )

        if len(operation.group_by_columns) != 1 or len(operation.measures) != 1:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_OPERATION",
                error_message="Operation must specify exactly 1 group_by_column and 1 measure.",
            )

        group_col = operation.group_by_columns[0].name
        measure_col = operation.measures[0].target_column.name
        measure_alias = operation.measures[0].output_alias

        if group_col not in df_slice.columns:
            return ValidationResult(
                is_valid=False,
                error_type="MISSING_COLUMN",
                error_message=f"Grouping column '{group_col}' does not exist in DataFrame.",
            )

        if measure_col not in df_slice.columns:
            return ValidationResult(
                is_valid=False,
                error_type="MISSING_COLUMN",
                error_message=f"Measure column '{measure_col}' does not exist in DataFrame.",
            )

        # Handle empty input DataFrame
        if len(df_slice) == 0:
            if isinstance(sql_actual_value, pd.DataFrame):
                if len(sql_actual_value) == 0:
                    return ValidationResult(
                        is_valid=True,
                        audit_details={"group_count": 0, "empty_input": True},
                    )
                return ValidationResult(
                    is_valid=False,
                    error_type="CARDINALITY_MISMATCH",
                    error_message=f"Expected 0 rows for empty input, got {len(sql_actual_value)}.",
                )
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_SQL_RESULT",
                error_message="sql_actual_value must be a pandas DataFrame.",
            )

        measure_series = df_slice[measure_col]
        if not pd.api.types.is_numeric_dtype(measure_series):
            if measure_series.isna().all():
                df_work = df_slice.copy()
                df_work[measure_col] = pd.to_numeric(measure_series, errors="coerce")
            else:
                return ValidationResult(
                    is_valid=False,
                    error_type="TYPE_MISMATCH",
                    error_message=f"Measure column '{measure_col}' has non-numeric dtype '{measure_series.dtype}'.",
                )
        else:
            df_work = df_slice

        # Independent calculation in Pandas:
        # dropna=False preserves NULL group keys (distinct group matching SQL)
        # min_count=1 ensures all-null measure groups return NaN (matching SQL NULL)
        try:
            expected_series = df_work.groupby(group_col, dropna=False)[measure_col].sum(min_count=1)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_type="CALCULATION_ERROR",
                error_message=f"Failed to calculate independent Pandas grouped sum: {e}",
            )

        if not isinstance(sql_actual_value, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_SQL_RESULT",
                error_message=f"sql_actual_value must be a pandas DataFrame, got '{type(sql_actual_value).__name__}'.",
            )

        # Output schema validation
        if group_col not in sql_actual_value.columns:
            return ValidationResult(
                is_valid=False,
                error_type="SCHEMA_MISMATCH",
                error_message=f"SQL result missing grouping column '{group_col}'.",
            )

        if measure_alias not in sql_actual_value.columns:
            return ValidationResult(
                is_valid=False,
                error_type="SCHEMA_MISMATCH",
                error_message=f"SQL result missing measure alias '{measure_alias}'.",
            )

        # Row cardinality validation
        expected_rows = len(expected_series)
        actual_rows = len(sql_actual_value)
        if actual_rows != expected_rows:
            return ValidationResult(
                is_valid=False,
                error_type="CARDINALITY_MISMATCH",
                error_message=f"Expected {expected_rows} groups, got {actual_rows}.",
                audit_details={"expected_rows": expected_rows, "actual_rows": actual_rows},
            )

        # Build group value maps with normalized null representation
        expected_map: Dict[Any, Any] = {}
        for k, v in expected_series.items():
            k_norm = None if pd.isna(k) else k
            expected_map[k_norm] = v

        actual_map: Dict[Any, Any] = {}
        for _, row in sql_actual_value.iterrows():
            k = row[group_col]
            k_norm = None if pd.isna(k) else k
            actual_map[k_norm] = row[measure_alias]

        # Group key membership check
        if set(expected_map.keys()) != set(actual_map.keys()):
            missing = set(expected_map.keys()) - set(actual_map.keys())
            extra = set(actual_map.keys()) - set(expected_map.keys())
            return ValidationResult(
                is_valid=False,
                error_type="GROUP_KEY_MISMATCH",
                error_message=f"Group keys mismatch. Missing: {list(missing)[:5]}, Extra: {list(extra)[:5]}.",
                audit_details={"missing_keys": list(missing), "extra_keys": list(extra)},
            )

        # Exact numerical comparison with documented tolerance
        for k, exp_val in expected_map.items():
            act_val = actual_map[k]

            # Both null / NaN
            if pd.isna(exp_val) and pd.isna(act_val):
                continue

            # Mismatched nullness
            if pd.isna(exp_val) or pd.isna(act_val):
                return ValidationResult(
                    is_valid=False,
                    error_type="NULL_MEASURE_MISMATCH",
                    error_message=f"Group '{k}' null mismatch: expected '{exp_val}', got '{act_val}'.",
                    audit_details={"group": k, "expected": str(exp_val), "actual": str(act_val)},
                )

            # Numerical comparison
            if not math.isclose(float(exp_val), float(act_val), rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                return ValidationResult(
                    is_valid=False,
                    error_type="VALUE_MISMATCH",
                    error_message=(
                        f"Group '{k}' sum mismatch: expected {exp_val}, got {act_val} "
                        f"(exceeds rel_tol={self.rel_tol}, abs_tol={self.abs_tol})."
                    ),
                    audit_details={"group": k, "expected": float(exp_val), "actual": float(act_val)},
                )

        return ValidationResult(
            is_valid=True,
            audit_details={
                "group_column": group_col,
                "measure_column": measure_col,
                "measure_alias": measure_alias,
                "group_count": expected_rows,
            },
        )


def register_grouped_sum_capability(
    registry: AnalyticalCapabilityRegistry,
    allowed_measure_types: Tuple[DataType, ...] = (DataType.FLOAT, DataType.INTEGER),
    allowed_group_types: Tuple[DataType, ...] = (DataType.STRING, DataType.INTEGER, DataType.DATE),
) -> None:
    """
    Registers the trusted aggregate.grouped_sum capability with the registry.
    Accurately declares input and output ports, authorization policy, and handlers.
    """
    input_ports = [
        PortDefinition(
            name="in_relation",
            payload_type=PortPayloadType(is_relation=True),
            semantic_role=SemanticRole.RAW_RECORD,
            grain=DataGrain.ANY,
            cardinality=PortCardinality.RELATION_TABLE,
            is_required=True,
            allow_nulls=False,
            description="Input relation dataset to be grouped and aggregated",
        ),
        PortDefinition(
            name="group_col",
            payload_type=PortPayloadType(
                is_relation=False,
                primitive_type=allowed_group_types[0] if allowed_group_types else DataType.STRING,
                allowed_primitive_types=allowed_group_types,
            ),
            semantic_role=SemanticRole.CATEGORICAL_DIMENSION,
            grain=DataGrain.ANY,
            cardinality=PortCardinality.COLUMN_VECTOR,
            is_required=True,
            allow_nulls=True,
            description="Dimension column vector to group by (supports null groups)",
        ),
        PortDefinition(
            name="measure_col",
            payload_type=PortPayloadType(
                is_relation=False,
                primitive_type=allowed_measure_types[0] if allowed_measure_types else DataType.FLOAT,
                allowed_primitive_types=allowed_measure_types,
            ),
            semantic_role=SemanticRole.NUMERIC_METRIC,
            grain=DataGrain.ANY,
            cardinality=PortCardinality.COLUMN_VECTOR,
            is_required=True,
            allow_nulls=True,
            description="Numeric metric column vector to sum over (nulls ignored; all-null group yields null)",
        ),
    ]

    output_ports = [
        PortDefinition(
            name="out_relation",
            payload_type=PortPayloadType(is_relation=True),
            semantic_role=SemanticRole.RAW_RECORD,
            grain=DataGrain.GROUPED_DIMENSIONAL,
            cardinality=PortCardinality.RELATION_TABLE,
            is_required=True,
            allow_nulls=False,
            description="Grouped relation containing unique dimension keys and aggregated sum measures",
        ),
    ]

    compiler_handler = GroupedSumCompilerHandler(
        allowed_measure_types=allowed_measure_types,
        allowed_group_types=allowed_group_types,
    )
    group_port = input_ports[1]
    measure_port = input_ports[2]

    # Enforce strict lockstep contract agreement
    assert group_port.payload_type.allowed_primitive_types == compiler_handler.allowed_group_types, (
        "Registry contract and compiler cannot disagree about accepted group types."
    )
    assert measure_port.payload_type.allowed_primitive_types == compiler_handler.allowed_measure_types, (
        "Registry contract and compiler cannot disagree about accepted measure types."
    )

    registry.register(
        capability_id=GROUPED_SUM_CAPABILITY_ID,
        version=GROUPED_SUM_VERSION,
        kind=CapabilityKind.ATOMIC,
        description="Aggregates an input relation by computing SUM over an authorized measure grouped by an authorized dimension.",
        input_ports=input_ports,
        output_ports=output_ports,
        compiler_handler=compiler_handler,
        validator_handler=GroupedSumValidatorHandler(),
        verifier_handler=GroupedSumVerifierHandler(),
        prerequisite_evaluators=[],
        authorization_policy=CapabilityAuthorizationPolicy(
            requires_grouping_authorization=True,
            requires_filtering_authorization=False,
        ),
        resource_constraints=ResourceConstraints(
            max_timeout_seconds=30,
            max_memory_mb=512,
            max_output_rows=10000,
        ),
    )
