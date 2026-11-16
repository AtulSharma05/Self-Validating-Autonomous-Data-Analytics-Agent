"""
Built-in Atomic Capability: filter.temporal_range (v1.0.0).

Enforces strict half-open temporal interval semantics [start, end):
- Exact lower bound is included (timestamp >= start).
- Exact upper bound is excluded (timestamp < end).
- Sub-second boundaries (1 microsecond before/after) are strictly honored.
- NULL/NaT timestamps are strictly excluded under SQL three-valued logic.
- Timezone naive vs aware mismatches fail closed.
- Lossy string conversions and unapproved mutations are prohibited.

TIMEZONE INTERPRETATION CONTRACT:
- Timezone-naive timestamps (TIMESTAMP / datetime64[ns]) represent pure abstract local
  wall-clock coordinates without political daylight saving transitions. In this model,
  every wall-clock point t is unique and monotonically ordered.
- Timezone-aware timestamps must possess an explicit fixed offset (e.g. UTC, 'Z', '+00:00', '-05:00')
  representing unambiguous point-in-time coordinates.
- Prohibited: Silent localization of naive timestamps, implicit timezone guessing, or mixing naive
  columns with aware boundaries (or vice-versa).
- Daylight Saving Time (DST) variable-offset regional names (e.g. 'America/New_York') that introduce
  ambiguous (fall-back overlap) or nonexistent (spring-forward gap) local clock times are genuinely
  outside the engine-neutral execution model. Unsupported DST inputs fail closed:
  * Nonexistent local times (e.g. 02:30 on spring forward day) fail closed (ValueError / DST_ERROR).
  * Ambiguous local times (e.g. 01:30 on fall back day) fail closed (ValueError / DST_ERROR).
"""

from __future__ import annotations
from dataclasses import dataclass
import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import sqlglot
from sqlglot import exp

from src.agent.air.schemas import (
    ColumnRef,
    ComparisonOperator,
    DataType,
    DimensionAuthorizationMatrix,
    Predicate,
)
from src.agent.air.operations import AnalyticalOperation, FilterOperation
from src.agent.air.contract import (
    CardinalityExpectation,
    PredicateOperator,
    TypedPredicate,
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


TEMPORAL_FILTER_CAPABILITY_ID = "filter.temporal_range"
TEMPORAL_FILTER_VERSION = "1.0.0"


def _flatten_and(node: exp.Expression) -> List[exp.Expression]:
    """Recursively flattens top-level AND conjuncts in an AST expression."""
    if isinstance(node, exp.Paren):
        return _flatten_and(node.this)
    if isinstance(node, exp.And):
        return _flatten_and(node.this) + _flatten_and(node.expression)
    return [node]


def _extract_literal_val(node: exp.Expression) -> str:
    """Extracts clean string representation of a literal, Cast, or Date expression."""
    if isinstance(node, exp.Literal):
        return node.this.strip("'").strip('"')
    if isinstance(node, (exp.Cast, exp.Date, exp.Ts)):
        return _extract_literal_val(node.this)
    return str(node).strip("'").strip('"')


def parse_and_validate_temporal_boundary(
    val: Any,
    target_tz: Optional[Any] = None,
) -> pd.Timestamp:
    """
    Parses and validates a temporal boundary adhering to the strict timezone interpretation contract.

    Contract:
    - Naive timestamps are preserved strictly as naive (abstract local timeline coordinates).
      They are NEVER silently localized or inferred to have a timezone.
    - Aware timestamps must possess an explicit fixed offset.
    - Daylight Saving Time (DST) variable-offset regional names that introduce ambiguous
      (fall-back overlap) or nonexistent (spring-forward gap) local clock times fail closed.
    """
    if isinstance(val, pd.Timestamp):
        ts = val
    elif isinstance(val, (datetime.datetime, datetime.date)):
        ts = pd.Timestamp(val)
    elif isinstance(val, str):
        val_clean = val.strip()
        try:
            ts = pd.Timestamp(val_clean)
        except Exception as e:
            raise ValueError(f"Failed to parse temporal boundary string '{val}': {e}") from e
    else:
        raise TypeError(f"Unsupported temporal boundary value type '{type(val).__name__}': {val}")

    if target_tz is not None:
        if ts.tz is None:
            try:
                # Strictly reject ambiguous and nonexistent DST local times
                ts = ts.tz_localize(target_tz, ambiguous="raise", nonexistent="raise")
            except Exception as e:
                raise ValueError(
                    f"DST / Timezone localization failure for '{val}' in timezone '{target_tz}': {e}"
                ) from e
        else:
            try:
                ts = ts.tz_convert(target_tz)
            except Exception as e:
                raise ValueError(f"Failed to convert timezone for '{val}' to '{target_tz}': {e}") from e

    return ts


@dataclass(frozen=True)
class TemporalFilterCompilerHandler:
    """
    Compiler handler for filter.temporal_range.
    Compiles a FilterOperation into an immutable, validated ContractFragment.
    Enforces exact contract alignment with port definition accepted types.
    """
    name: str = "TemporalFilterCompiler"
    accepted_types: Tuple[DataType, ...] = (DataType.DATETIME, DataType.DATE)

    def compile_fragment(
        self,
        operation: AnalyticalOperation,
        dimensions: Dict[str, DimensionAuthorizationMatrix],
        base_schemas: Dict[str, Dict[str, DataType]],
    ) -> ContractFragment:
        if not isinstance(operation, FilterOperation):
            raise TypeError(
                f"TemporalFilterCompiler expected FilterOperation, got '{type(operation).__name__}'."
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

        # Extract temporal predicate information
        col_name: Optional[str] = None
        start_val: Optional[Any] = None
        end_val: Optional[Any] = None

        for p in operation.predicates:
            p_col = p.column.name
            if isinstance(p, TypedPredicate):
                if p.operator == PredicateOperator.BETWEEN_HALF_OPEN:
                    col_name = p_col
                    start_val = p.literal_value
                    end_val = p.secondary_literal
                    break
                elif p.operator in (PredicateOperator.GTE, PredicateOperator.GT):
                    col_name = p_col
                    start_val = p.literal_value
                elif p.operator in (PredicateOperator.LT, PredicateOperator.LTE):
                    col_name = p_col
                    end_val = p.literal_value
            elif isinstance(p, Predicate):
                if p.operator == ComparisonOperator.BETWEEN:
                    col_name = p_col
                    if isinstance(p.literal_value, (list, tuple)) and len(p.literal_value) >= 2:
                        start_val = p.literal_value[0]
                        end_val = p.literal_value[1]
                    else:
                        start_val = p.literal_value
                    break
                elif p.operator in (ComparisonOperator.GTE, ComparisonOperator.GT):
                    col_name = p_col
                    start_val = p.literal_value
                elif p.operator in (ComparisonOperator.LT, ComparisonOperator.LTE):
                    col_name = p_col
                    end_val = p.literal_value

        if not col_name:
            raise ValueError(
                f"Operation '{operation.operation_id}' contains no recognized temporal predicates."
            )

        if start_val is None or end_val is None:
            raise ValueError(
                f"Operation '{operation.operation_id}' on temporal column '{col_name}' "
                f"requires both start and end boundaries. Got start={start_val}, end={end_val}."
            )

        # Validate column existence in base table schema
        if col_name not in table_schema:
            raise KeyError(
                f"Temporal filter column '{col_name}' does not exist in schema of '{source_table}'."
            )

        col_dtype = table_schema[col_name]
        # Contract-governed type acceptance check
        if col_dtype not in self.accepted_types:
            raise TypeError(
                f"Temporal filter column '{col_name}' has type '{col_dtype}'. "
                f"Port contract authorizes only: {tuple(t.value for t in self.accepted_types)}."
            )

        # Dimension Authorization Check (Fail-Closed)
        dim_auth = dimensions.get(col_name)
        if not dim_auth or not dim_auth.allow_filter_predicate:
            raise PermissionError(
                f"Security Authorization Failure: Dimension '{col_name}' is not authorized "
                "for filter predicates (allow_filter_predicate is False or not granted)."
            )

        # Interval consistency and DST / timezone validation
        try:
            ts_start = parse_and_validate_temporal_boundary(start_val)
            ts_end = parse_and_validate_temporal_boundary(end_val)
            if ts_start > ts_end:
                raise ValueError(
                    f"Contradictory temporal range: start '{start_val}' is after end '{end_val}'."
                )
        except Exception as e:
            if "Contradictory" in str(e):
                raise
            raise ValueError(f"Failed to parse temporal boundaries '{start_val}', '{end_val}': {e}") from e

        col_ref = ColumnRef(name=col_name, data_type=col_dtype, table_name=source_table)
        req_pred = TypedPredicate(
            column=col_ref,
            operator=PredicateOperator.BETWEEN_HALF_OPEN,
            literal_value=str(start_val),
            secondary_literal=str(end_val),
            is_temporal=True,
        )

        output_cols = tuple((k, v) for k, v in operation.output_schema.items())

        return ContractFragment(
            operation_id=operation.operation_id,
            capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
            capability_version=TEMPORAL_FILTER_VERSION,
            source_table=source_table,
            authorized_source_columns=(col_name,),
            required_filters=(req_pred,),
            expected_output_columns=output_cols,
            cardinality=CardinalityExpectation.DIMENSIONAL,
            logical_dependencies=tuple(operation.depends_on),
        )


@dataclass(frozen=True)
class TemporalFilterVerifierHandler:
    """
    Verifier handler for filter.temporal_range.
    Statically verifies the SQLGlot AST WHERE clause against the half-open interval contract.
    """
    name: str = "TemporalFilterVerifier"

    def verify_ast_fragment(
        self,
        scope_expression: exp.Expression,
        fragment: ContractFragment,
    ) -> Tuple[bool, Optional[str]]:
        where_clause = scope_expression.args.get("where")
        if not fragment.required_filters:
            if where_clause:
                return False, "Query contains a WHERE clause, but fragment requires no filters."
            return True, None

        if not where_clause:
            return False, (
                "Query is missing a WHERE clause, but fragment requires temporal filter: "
                f"{[p.describe() for p in fragment.required_filters]}."
            )

        conjuncts = _flatten_and(where_clause.this)

        # Check for weakened OR conditions containing any temporal filter column
        auth_cols = {c.lower().strip('"') for c in fragment.authorized_source_columns}
        for conj in conjuncts:
            while isinstance(conj, exp.Paren):
                conj = conj.this
            if isinstance(conj, exp.Or):
                or_cols = {c.name.lower().strip('"') for c in conj.find_all(exp.Column)}
                overlap = or_cols.intersection(auth_cols)
                if overlap:
                    return False, (
                        f"Temporal filter column '{list(overlap)[0]}' is placed in a weakened OR condition, "
                        "which fails to enforce the temporal boundary for all rows."
                    )

        # Check required filters
        for req in fragment.required_filters:
            req_col = req.column.name.lower().strip('"')
            has_lower = False
            has_upper = False

            for conj in conjuncts:
                while isinstance(conj, exp.Paren):
                    conj = conj.this

                # 1. Catch inclusive BETWEEN
                if isinstance(conj, exp.Between):
                    col_node = conj.this if isinstance(conj.this, exp.Column) else conj.find(exp.Column)
                    if col_node and col_node.name.lower().strip('"') == req_col:
                        return False, (
                            "BETWEEN operator is inclusive of both bounds, violating the strict "
                            "half-open temporal interval [start, end)."
                        )

                # 2. Check binary comparisons
                if isinstance(conj, (exp.GTE, exp.GT, exp.LTE, exp.LT, exp.EQ, exp.NEQ)):
                    col_node = conj.this if isinstance(conj.this, exp.Column) else conj.find(exp.Column)
                    if not col_node or col_node.name.lower().strip('"') != req_col:
                        continue

                    actual_literal = _extract_literal_val(conj.expression)

                    # Lower boundary check (must be >= start)
                    if isinstance(conj, exp.GT):
                        return False, (
                            f"Lower boundary uses strictly greater (>) on '{req_col}', "
                            "violating inclusive lower bound (>=)."
                        )
                    elif isinstance(conj, exp.GTE):
                        if actual_literal != str(req.literal_value):
                            return False, (
                                f"Lower boundary value mismatch: expected '{req.literal_value}', "
                                f"got '{actual_literal}'."
                            )
                        has_lower = True

                    # Upper boundary check (must be < end)
                    if isinstance(conj, exp.LTE):
                        return False, (
                            f"Upper boundary uses inclusive (<= '{actual_literal}') on '{req_col}', "
                            f"violating strict half-open interval (< '{req.secondary_literal}')."
                        )
                    elif isinstance(conj, exp.LT):
                        if actual_literal != str(req.secondary_literal):
                            return False, (
                                f"Upper boundary value mismatch: expected '{req.secondary_literal}', "
                                f"got '{actual_literal}'."
                            )
                        has_upper = True

            if not has_lower:
                return False, f"Missing required lower bound (>= '{req.literal_value}') on column '{req_col}'."
            if not has_upper:
                return False, f"Missing required upper bound (< '{req.secondary_literal}') on column '{req_col}'."

        # Check for unauthorized extra predicates
        for conj in conjuncts:
            while isinstance(conj, exp.Paren):
                conj = conj.this
            cols = {c.name.lower().strip('"') for c in conj.find_all(exp.Column)}
            unauthorized = cols - auth_cols
            if unauthorized:
                return False, (
                    f"Unauthorized filter predicate on column(s) {unauthorized} "
                    "not permitted by temporal filter contract fragment."
                )

        return True, None


@dataclass(frozen=True)
class TemporalFilterValidatorHandler:
    """
    Independent validator handler for filter.temporal_range.
    Computes expected slice in independent Pandas execution on raw data
    and compares row count and row identity against actual SQL results.
    """
    name: str = "TemporalFilterValidator"

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

        # Extract filter target column and bounds
        col_name: Optional[str] = None
        start_val: Optional[Any] = None
        end_val: Optional[Any] = None

        if isinstance(operation, FilterOperation):
            for p in operation.predicates:
                p_col = p.column.name
                if isinstance(p, TypedPredicate):
                    if p.operator == PredicateOperator.BETWEEN_HALF_OPEN:
                        col_name = p_col
                        start_val = p.literal_value
                        end_val = p.secondary_literal
                        break
                    elif p.operator in (PredicateOperator.GTE, PredicateOperator.GT):
                        col_name = p_col
                        start_val = p.literal_value
                    elif p.operator in (PredicateOperator.LT, PredicateOperator.LTE):
                        col_name = p_col
                        end_val = p.literal_value
                elif isinstance(p, Predicate):
                    if p.operator == ComparisonOperator.BETWEEN:
                        col_name = p_col
                        if isinstance(p.literal_value, (list, tuple)) and len(p.literal_value) >= 2:
                            start_val = p.literal_value[0]
                            end_val = p.literal_value[1]
                        else:
                            start_val = p.literal_value
                        break
                    elif p.operator in (ComparisonOperator.GTE, ComparisonOperator.GT):
                        col_name = p_col
                        start_val = p.literal_value
                    elif p.operator in (ComparisonOperator.LT, ComparisonOperator.LTE):
                        col_name = p_col
                        end_val = p.literal_value

        if not col_name or start_val is None or end_val is None:
            return ValidationResult(
                is_valid=False,
                error_type="INVALID_OPERATION",
                error_message="Operation does not define a complete temporal range filter.",
            )

        if col_name not in df_slice.columns:
            return ValidationResult(
                is_valid=False,
                error_type="MISSING_COLUMN",
                error_message=f"Target column '{col_name}' does not exist in DataFrame.",
            )

        series = df_slice[col_name]

        # Check temporal dtype: support datetime64 or object series containing date/datetime instances
        is_dt64 = pd.api.types.is_datetime64_any_dtype(series)
        is_date_objects = False
        if not is_dt64:
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                return ValidationResult(
                    is_valid=False,
                    error_type="TYPE_MISMATCH",
                    error_message=(
                        f"Target column '{col_name}' has non-temporal numeric/boolean dtype '{series.dtype}'. "
                        "String or numeric columns are rejected fail-closed without lossy conversion."
                    ),
                )
            # Inspect non-null elements
            non_null = series.dropna()
            if len(non_null) > 0 and all(isinstance(x, (datetime.date, datetime.datetime)) for x in non_null[:100]):
                is_date_objects = True
            else:
                return ValidationResult(
                    is_valid=False,
                    error_type="TYPE_MISMATCH",
                    error_message=(
                        f"Target column '{col_name}' has non-temporal dtype '{series.dtype}'. "
                        "String or numeric columns are rejected fail-closed without lossy conversion."
                    ),
                )

        # Parse boundaries with strict DST and timezone validation
        try:
            start_ts = parse_and_validate_temporal_boundary(start_val)
            end_ts = parse_and_validate_temporal_boundary(end_val)
        except Exception as e:
            err_type = "DST_ERROR" if ("daylight savings time" in str(e).lower() or "dst" in str(e).lower()) else "PARSE_ERROR"
            return ValidationResult(
                is_valid=False,
                error_type=err_type,
                error_message=f"Failed to parse temporal boundaries '{start_val}', '{end_val}': {e}",
            )

        # Timezone awareness mismatch check
        col_tz = series.dt.tz if hasattr(series, "dt") else None
        start_tz = start_ts.tz
        end_tz = end_ts.tz

        if (col_tz is None and (start_tz is not None or end_tz is not None)) or \
           (col_tz is not None and (start_tz is None or end_tz is None)):
            return ValidationResult(
                is_valid=False,
                error_type="TIMEZONE_MISMATCH",
                error_message=(
                    f"Timezone awareness mismatch: column timezone is '{col_tz}', "
                    f"filter bounds are '{start_tz}' and '{end_tz}'. "
                    "Cannot compare timezone-naive and timezone-aware timestamps. "
                    "Silent localization or timezone inference is strictly prohibited."
                ),
            )

        # Strict half-open interval evaluation [start, end)
        # NULL timestamps evaluate to False and are excluded
        if is_dt64:
            mask = (series >= start_ts) & (series < end_ts) & series.notna()
        elif is_date_objects:
            s_d = start_ts.date()
            e_d = end_ts.date()
            mask = series.map(lambda x: (x >= s_d) and (x < e_d) if (pd.notna(x) and x is not None) else False)
        else:
            mask = pd.Series(False, index=df_slice.index)

        expected_slice = df_slice[mask]
        expected_count = int(mask.sum())

        # Validate against actual SQL value
        if isinstance(sql_actual_value, pd.DataFrame):
            actual_count = len(sql_actual_value)
            if actual_count != expected_count:
                return ValidationResult(
                    is_valid=False,
                    error_type="CARDINALITY_MISMATCH",
                    error_message=f"Expected {expected_count} rows, got {actual_count} rows.",
                    audit_details={"expected_rows": expected_count, "actual_rows": actual_count},
                )

            # Check row identity / membership if identifiable key exists
            key_cols = [c for c in ("row_id", "id", "order_id") if c in expected_slice.columns and c in sql_actual_value.columns]
            if key_cols:
                key = key_cols[0]
                expected_ids = set(expected_slice[key])
                actual_ids = set(sql_actual_value[key])
                if expected_ids != actual_ids:
                    return ValidationResult(
                        is_valid=False,
                        error_type="ROW_MEMBERSHIP_MISMATCH",
                        error_message="Row keys in SQL result do not match expected slice keys.",
                        audit_details={
                            "missing_keys": list(expected_ids - actual_ids)[:10],
                            "extra_keys": list(actual_ids - expected_ids)[:10],
                        },
                    )
            elif expected_slice.index.isin(sql_actual_value.index).all() and len(sql_actual_value.index) == expected_count:
                pass

        elif isinstance(sql_actual_value, int):
            if sql_actual_value != expected_count:
                return ValidationResult(
                    is_valid=False,
                    error_type="CARDINALITY_MISMATCH",
                    error_message=f"Expected {expected_count} rows, got {sql_actual_value} rows.",
                    audit_details={"expected_rows": expected_count, "actual_rows": sql_actual_value},
                )
        elif sql_actual_value is None:
            return ValidationResult(
                is_valid=False,
                error_type="MISSING_ACTUAL_VALUE",
                error_message="SQL actual value is None.",
            )

        return ValidationResult(
            is_valid=True,
            audit_details={
                "column": col_name,
                "start": str(start_ts),
                "end": str(end_ts),
                "expected_rows": expected_count,
            },
        )


def register_temporal_filter_capability(
    registry: AnalyticalCapabilityRegistry,
    accepted_types: Tuple[DataType, ...] = (DataType.DATETIME, DataType.DATE),
) -> None:
    """
    Registers the trusted filter.temporal_range capability with the registry.
    Accurately declares accepted primitive types on the input port and guarantees
    lockstep agreement between port contract and compiler handler.
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
            description="Input relation table to be filtered by temporal range",
        ),
        PortDefinition(
            name="time_col",
            payload_type=PortPayloadType(
                is_relation=False,
                primitive_type=accepted_types[0] if accepted_types else DataType.DATETIME,
                allowed_primitive_types=accepted_types,
            ),
            semantic_role=SemanticRole.TEMPORAL_BOUNDARY,
            grain=DataGrain.ANY,
            cardinality=PortCardinality.COLUMN_VECTOR,
            is_required=True,
            allow_nulls=False,
            description="Timestamp or datetime/date column vector defining the temporal dimension",
        ),
    ]

    output_ports = [
        PortDefinition(
            name="out_relation",
            payload_type=PortPayloadType(is_relation=True),
            semantic_role=SemanticRole.RAW_RECORD,
            grain=DataGrain.ANY,
            cardinality=PortCardinality.RELATION_TABLE,
            is_required=True,
            allow_nulls=False,
            description="Filtered relation table satisfying [start, end) temporal boundary",
        ),
    ]

    compiler_handler = TemporalFilterCompilerHandler(accepted_types=accepted_types)
    time_port = input_ports[1]

    # Enforce strict lockstep contract agreement
    assert time_port.payload_type.allowed_primitive_types == compiler_handler.accepted_types, (
        "Registry contract and compiler cannot disagree about accepted types."
    )

    registry.register(
        capability_id=TEMPORAL_FILTER_CAPABILITY_ID,
        version=TEMPORAL_FILTER_VERSION,
        kind=CapabilityKind.ATOMIC,
        description="Filters a relation table to records within a strict half-open temporal interval [start, end).",
        input_ports=input_ports,
        output_ports=output_ports,
        compiler_handler=compiler_handler,
        validator_handler=TemporalFilterValidatorHandler(),
        verifier_handler=TemporalFilterVerifierHandler(),
        prerequisite_evaluators=[],
        authorization_policy=CapabilityAuthorizationPolicy(
            requires_filtering_authorization=True,
        ),
        resource_constraints=ResourceConstraints(
            max_timeout_seconds=30,
            max_memory_mb=512,
            max_output_rows=None,
        ),
    )
