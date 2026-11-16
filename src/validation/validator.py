"""
DeterministicValidator — orchestrates all three validation layers.

Usage:
    validator = DeterministicValidator(con, table_name="sales")
    report = validator.validate_aggregation_result(
        result_df=df,
        metric_col="revenue",
        group_col="region",
    )
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import duckdb
import pandas as pd

from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    SemanticCoverageReport,
    ValidationResult,
)
from src.validation.invariant_rules import (
    BoundaryRule,
    DomainBoundaryRules,
    SumOfPartsRule,
)


# ---------------------------------------------------------------------------
# Dual-Path Verification (Layer 3)
# ---------------------------------------------------------------------------

class DualPathVerifier:
    """
    Independently computes the same scalar aggregation via two orthogonal paths:
      - Path A: DuckDB SQL query
      - Path B: Vectorized Pandas aggregation on the same table slice

    Asserts |A - B| < EPSILON.  Any discrepancy reveals a logic error in the
    generated query (wrong filter, wrong JOIN, wrong column alias).
    """

    EPSILON = 1e-4

    @classmethod
    def verify(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        metric_col: str,
        agg_func: str = "SUM",
        where_clause: str = "1=1",
    ) -> ValidationResult:
        """
        Args:
            con:          Active DuckDB connection.
            table_name:   Source table.
            metric_col:   Column to aggregate.
            agg_func:     'SUM', 'AVG', or 'COUNT'.
            where_clause: SQL WHERE filter to apply consistently to both paths.

        Returns:
            ValidationResult with dual-path cross-check details.
        """
        agg_upper = agg_func.upper()
        supported = {"SUM", "AVG", "COUNT", "COUNT_DISTINCT", "COUNT(DISTINCT)", "DISTINCT_COUNT", "AOV", "MAX", "MIN"}
        if agg_upper not in supported:
            return ValidationResult(
                is_valid=False,
                error_type="SYNTAX_ERROR",
                error_message=f"Unsupported agg_func '{agg_func}'. Use: {supported}",
            )

        try:
            cols = [c[0] for c in con.execute(f'DESCRIBE "{table_name}"').fetchall()]
            ord_col = next((c for c in cols if "order_id" in c.lower() or c.lower() == "order_id"), "order_id")

            # --- Path A: DuckDB SQL ---
            if agg_upper in {"COUNT_DISTINCT", "COUNT(DISTINCT)", "DISTINCT_COUNT"}:
                sql_q = (
                    f"SELECT CAST(COUNT(DISTINCT \"{metric_col}\") AS DOUBLE) "
                    f"FROM {table_name} WHERE {where_clause}"
                )
            elif agg_upper == "AOV":
                sql_q = (
                    f"SELECT CAST(SUM(\"{metric_col}\") AS DOUBLE) / NULLIF(CAST(COUNT(DISTINCT \"{ord_col}\") AS DOUBLE), 0) "
                    f"FROM {table_name} WHERE {where_clause}"
                )
            else:
                # Cast to DOUBLE inside SQL to avoid decimal.Decimal vs float type mismatch
                sql_q = (
                    f"SELECT CAST({agg_upper}(\"{metric_col}\") AS DOUBLE) "
                    f"FROM {table_name} WHERE {where_clause}"
                )
            path_a_raw = con.execute(sql_q).fetchone()[0]
            path_a_val: float = float(path_a_raw) if path_a_raw is not None else 0.0

            # --- Path B: Pandas (pull the filtered slice from DuckDB) ---
            if agg_upper == "AOV":
                slice_q = f"SELECT \"{metric_col}\", \"{ord_col}\" FROM {table_name} WHERE {where_clause}"
                df_slice: pd.DataFrame = con.execute(slice_q).df()
                valid_slice = df_slice[df_slice[ord_col].notna()]
                num_orders = valid_slice[ord_col].nunique()
                valid_rev_series = valid_slice[metric_col].sum(min_count=1)
                path_b_val = float(valid_rev_series / num_orders) if num_orders > 0 and pd.notna(valid_rev_series) else 0.0
            else:
                slice_q = (
                    f"SELECT \"{metric_col}\" FROM {table_name} WHERE {where_clause}"
                )
                df_slice: pd.DataFrame = con.execute(slice_q).df()
                series = df_slice[metric_col].dropna()

                if agg_upper == "SUM":
                    sum_val = series.sum(min_count=1)
                    path_b_val = float(sum_val) if pd.notna(sum_val) else 0.0
                elif agg_upper == "AVG":
                    path_b_val = float(series.mean()) if len(series) > 0 else 0.0
                elif agg_upper == "COUNT":
                    path_b_val = float(len(series))
                elif agg_upper in {"COUNT_DISTINCT", "COUNT(DISTINCT)", "DISTINCT_COUNT"}:
                    path_b_val = float(series.nunique())
                elif agg_upper == "MAX":
                    path_b_val = float(series.max()) if len(series) > 0 else 0.0
                elif agg_upper == "MIN":
                    path_b_val = float(series.min()) if len(series) > 0 else 0.0
                else:
                    path_b_val = 0.0

            diff = abs(path_a_val - path_b_val)

            if diff > cls.EPSILON:
                return ValidationResult(
                    is_valid=False,
                    error_type="INVARIANT_ERROR",
                    error_message=(
                        f"Dual-path mismatch on {agg_func}({metric_col}): "
                        f"SQL={path_a_val:,.6f}, Pandas={path_b_val:,.6f}, "
                        f"Δ={diff:,.8f} (threshold={cls.EPSILON}). "
                        f"Query logic likely has a JOIN error or wrong filter."
                    ),
                    audit_details={
                        "path_a_sql": path_a_val,
                        "path_b_pandas": path_b_val,
                        "discrepancy": diff,
                        "epsilon": cls.EPSILON,
                        "agg_func": agg_func,
                        "metric_col": metric_col,
                        "where_clause": where_clause,
                    },
                )

            return ValidationResult(
                is_valid=True,
                audit_details={
                    "path_a_sql": path_a_val,
                    "path_b_pandas": path_b_val,
                    "discrepancy": diff,
                    "agg_func": agg_func,
                    "metric_col": metric_col,
                },
            )

        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                error_type="SYNTAX_ERROR",
                error_message=f"DualPathVerifier failed: {exc}",
                audit_details={"exception": str(exc)},
            )


# ---------------------------------------------------------------------------
# Arithmetic & Mathematical Consistency Verifier (Layer 4)
# ---------------------------------------------------------------------------

class ArithmeticVerifier:
    """
    Independently computes and verifies derived calculations (e.g. percentage changes, ratios)
    against underlying component values using pure deterministic math.
    """
    EPSILON = 0.05

    @classmethod
    def verify_percentage_change(
        cls,
        base_value: float,
        test_value: float,
        claimed_pct_change: float,
    ) -> ValidationResult:
        if base_value == 0:
            expected = 0.0
        else:
            expected = ((test_value - base_value) / abs(base_value)) * 100.0

        diff = abs(claimed_pct_change - expected)
        if diff > cls.EPSILON:
            return ValidationResult(
                is_valid=False,
                error_type="calculation_mismatch",
                error_message=f"Percentage change is incorrect. Expected: {expected:.2f}%, Actual: {claimed_pct_change:.2f}%.",
                audit_details={
                    "status": "failed",
                    "error_type": "calculation_mismatch",
                    "expected": round(expected, 2),
                    "actual": round(claimed_pct_change, 2),
                    "message": "Percentage change is incorrect.",
                    "base_value": base_value,
                    "test_value": test_value,
                }
            )
        return ValidationResult(
            is_valid=True,
            audit_details={
                "status": "passed",
                "expected": round(expected, 2),
                "actual": round(claimed_pct_change, 2),
            }
        )


# ---------------------------------------------------------------------------
# Data Quality & Mapping Verifier (Layer 5)
# ---------------------------------------------------------------------------

class DataQualityVerifier:
    """
    Independently verifies data quality / mapping conflict query results
    against the raw table to guarantee 100% accuracy.
    """
    @classmethod
    def verify_mapping_conflicts(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        group_entity: str,
        target_entity: str,
        result_df: Optional[pd.DataFrame],
    ) -> ValidationResult:
        try:
            # Independent ground truth count query
            q = f"""
                SELECT COUNT(*) FROM (
                    SELECT "{group_entity}"
                    FROM {table_name}
                    GROUP BY "{group_entity}"
                    HAVING COUNT(DISTINCT "{target_entity}") > 1
                )
            """
            expected_conflicts = int(con.execute(q).fetchone()[0])
            actual_conflicts = len(result_df) if result_df is not None else 0

            if expected_conflicts != actual_conflicts:
                return ValidationResult(
                    is_valid=False,
                    error_type="DATA_QUALITY_MISMATCH",
                    error_message=f"Conflict count mismatch on {group_entity} -> {target_entity}: Expected {expected_conflicts}, got {actual_conflicts}.",
                    audit_details={
                        "expected_conflicts": expected_conflicts,
                        "actual_conflicts": actual_conflicts,
                    }
                )
            return ValidationResult(
                is_valid=True,
                audit_details={
                    "conflict_count": actual_conflicts,
                    "group_entity": group_entity,
                    "target_entity": target_entity,
                }
            )
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                error_type="VERIFICATION_EXCEPTION",
                error_message=str(exc),
            )


# ---------------------------------------------------------------------------
# Schema and Output Structure Verifier (Layer 6)
# ---------------------------------------------------------------------------

class SchemaAndTypeVerifier:
    """
    Validates result DataFrame schema, non-empty constraints, and null fractions.
    """
    @classmethod
    def verify(
        cls,
        df: Optional[pd.DataFrame],
        expected_columns: Optional[List[str]] = None,
        disallow_empty: bool = False,
    ) -> ValidationResult:
        if df is None:
            return ValidationResult(
                is_valid=False,
                error_type="NULL_DATAFRAME",
                error_message="Execution produced None instead of a DataFrame.",
            )
        if disallow_empty and df.empty:
            return ValidationResult(
                is_valid=False,
                error_type="EMPTY_RESULT",
                error_message="Execution produced an empty DataFrame (0 rows).",
            )
        if expected_columns:
            missing = [c for c in expected_columns if c not in df.columns]
            if missing:
                return ValidationResult(
                    is_valid=False,
                    error_type="SCHEMA_MISMATCH",
                    error_message=f"Result missing expected columns: {missing}",
                    audit_details={"missing_columns": missing, "available_columns": list(df.columns)},
                )
        return ValidationResult(is_valid=True, audit_details={"columns": list(df.columns), "row_count": len(df)})


# ---------------------------------------------------------------------------
# Semantic Coverage Verifier (Layer 7)
# ---------------------------------------------------------------------------

class SemanticCoverageValidator:
    """
    Validates semantic question coverage by checking consistency across:
      1. Original user question
      2. Structured question specification (AnalysisObjective)
      3. Planned operations (AnalysisPlan)
      4. Executed analytical operations (step_results)
      5. Final report outputs

    Ensures that every requested metric, filter, and entity grain is represented.
    Does NOT pass coverage merely because a step succeeded. Fails if a requested
    metric is missing or replaced with an unauthorized substitution.
    """

    @classmethod
    def validate(
        cls,
        user_question: str,
        objective: AnalysisObjective,
        plan: Optional[AnalysisPlan],
        step_results: Dict[str, Any],
        executed_sqls: Optional[List[str]] = None,
    ) -> SemanticCoverageReport:
        import re

        q_lower = user_question.lower()
        requested_metrics: List[str] = []
        executed_metrics: List[str] = []
        missing_metrics: List[str] = []
        unauthorized_substitutions: List[Dict[str, str]] = []
        requested_filters: List[str] = []
        applied_filters: List[str] = []
        missing_filters: List[str] = []

        # 1. Check for explicit missing columns explanation on objective
        if getattr(objective, "missing_columns_explanation", None):
            return SemanticCoverageReport(
                is_covered=False,
                requested_metrics=objective.required_outputs or [objective.target_metric],
                executed_metrics=[],
                missing_metrics=objective.required_outputs or [objective.target_metric],
                unauthorized_substitutions=[],
                requested_filters=[],
                applied_filters=[],
                missing_filters=[],
                details=f"Required columns unavailable: {objective.missing_columns_explanation}",
            )

        # 2. Identify requested metrics in user_question
        is_scenario = getattr(objective, "is_scenario", False) or ("capped" in q_lower and "discount" in q_lower)
        wants_orders = not is_scenario and any(
            re.search(p, q_lower) for p in [
                r'\bhow many orders\b',
                r'\bhow many total orders\b',
                r'\btotal orders?\b',
                r'\border count\b',
                r'\bnumber of orders?\b',
                r'\bunique orders?\b',
                r'\bcount of orders?\b',
                r'\bcount\s+(?:distinct\s+)?orders?\b',
                r'\borders?\s+were\s+placed\b',
            ]
        )
        wants_aov = not is_scenario and bool(re.search(r'\b(average order value|aov)\b', q_lower))
        wants_profit = "profit" in q_lower or "earnings" in q_lower
        wants_sales = ("sales" in q_lower or "revenue" in q_lower or "turnover" in q_lower) and not wants_aov
        wants_margin = "margin" in q_lower or "profit margin" in q_lower
        wants_quantity = "quantity" in q_lower or "units sold" in q_lower

        if wants_orders:
            requested_metrics.append("total_orders")
        if wants_aov:
            requested_metrics.append("average_order_value")
        if wants_profit:
            requested_metrics.append("profit")
        if wants_sales:
            requested_metrics.append("sales_or_revenue")
        if wants_margin:
            requested_metrics.append("profit_margin")
        if wants_quantity:
            requested_metrics.append("quantity")

        # Temporal filter in question (e.g. 2017)
        yr_m = re.search(r'\b(20\d\d|19\d\d)\b', q_lower)
        if yr_m:
            requested_filters.append(yr_m.group(1))

        # 3. Inspect executed steps from plan and step_results
        executed_steps = plan.steps if plan else []
        for s in executed_steps:
            code_str = (s.code or "").lower()
            desc_str = (s.description or "").lower()
            agg_str = (s.agg_func or "").upper()
            col_str = (s.metric_col or "").lower()

            if "count_distinct" in agg_str or "count(distinct" in code_str or "distinct" in desc_str or "unique order" in desc_str:
                executed_metrics.append("total_orders")
            if "aov" in agg_str or "aov" in desc_str or "/ nullif" in code_str or ("count(distinct" in code_str and "sum(" in code_str and "/" in code_str):
                executed_metrics.append("average_order_value")
            if "profit" in col_str or "profit" in desc_str:
                executed_metrics.append("profit")
            if any(k in col_str or k in desc_str for k in ["sales", "revenue", "amount"]) and not ("aov" in desc_str or "aov" in agg_str):
                executed_metrics.append("sales_or_revenue")
            if "margin" in desc_str:
                executed_metrics.append("profit_margin")
            if any(k in col_str or k in desc_str for k in ["quantity", "units"]):
                executed_metrics.append("quantity")

            if s.where_clause and s.where_clause != "1=1":
                applied_filters.append(s.where_clause)
            elif "where" in code_str:
                applied_filters.append(code_str)

        # Also inspect columns in executed step_results DataFrames
        for s_id, s_data in (step_results or {}).items():
            if isinstance(s_data, dict):
                df_res = s_data.get("dataframe")
                if df_res is not None and not df_res.empty:
                    df_cols = [str(c).lower() for c in df_res.columns]
                    if any("order" in c for c in df_cols):
                        executed_metrics.append("total_orders")
                    if any("aov" in c or "average_order_value" in c for c in df_cols):
                        executed_metrics.append("average_order_value")
                    if any("sales" in c or "revenue" in c for c in df_cols):
                        executed_metrics.append("sales_or_revenue")

        # 4. Check requested vs executed & detect unauthorized substitutions
        if wants_orders and "total_orders" not in executed_metrics:
            missing_metrics.append("Total unique orders")
            if "sales_or_revenue" in executed_metrics:
                unauthorized_substitutions.append({
                    "requested": "Total unique orders",
                    "executed": "SUM(revenue/sales)",
                })

        if wants_aov and "average_order_value" not in executed_metrics:
            missing_metrics.append("Average Order Value (AOV)")
            if "sales_or_revenue" in executed_metrics:
                unauthorized_substitutions.append({
                    "requested": "Average Order Value (AOV)",
                    "executed": "SUM(revenue/sales)",
                })

        if wants_profit and "profit" not in executed_metrics:
            missing_metrics.append("Profit")
            if "sales_or_revenue" in executed_metrics:
                unauthorized_substitutions.append({
                    "requested": "Profit",
                    "executed": "Revenue/Sales",
                })

        if wants_sales and "sales_or_revenue" not in executed_metrics:
            missing_metrics.append("Sales/Revenue")

        # 5. Filter check (e.g. year 2017)
        for rf in requested_filters:
            found_filter = any(rf in af for af in applied_filters) or any(rf in sql for sql in (executed_sqls or []))
            if not found_filter:
                missing_filters.append(f"Year {rf}")

        is_covered = (
            len(missing_metrics) == 0
            and len(missing_filters) == 0
            and len(unauthorized_substitutions) == 0
        )

        details_parts = []
        if is_covered:
            details_parts.append("All requested metrics, filters, and entity grains are verified.")
        else:
            if missing_metrics:
                details_parts.append(f"Missing requested metrics: {', '.join(missing_metrics)}.")
            if unauthorized_substitutions:
                sub_desc = ", ".join(f"'{s['requested']}' replaced with '{s['executed']}'" for s in unauthorized_substitutions)
                details_parts.append(f"Unauthorized metric substitutions: {sub_desc}.")
            if missing_filters:
                details_parts.append(f"Missing filters: {', '.join(missing_filters)}.")

        return SemanticCoverageReport(
            is_covered=is_covered,
            requested_metrics=requested_metrics,
            executed_metrics=list(set(executed_metrics)),
            missing_metrics=missing_metrics,
            unauthorized_substitutions=unauthorized_substitutions,
            requested_filters=requested_filters,
            applied_filters=applied_filters,
            missing_filters=missing_filters,
            details=" ".join(details_parts),
        )


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Aggregated result of all validation layers for a single execution step."""

    step_id: str
    table_name: str
    metric_col: str
    is_fully_valid: bool
    checks_passed: int = 0
    checks_total: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    checksum_hash: str = ""
    missing_sales_diagnostics: Dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total > 0 else 0.0

    def summary(self) -> str:
        status = "✅ VALID" if self.is_fully_valid else "❌ INVALID"
        return (
            f"{status} | {self.checks_passed}/{self.checks_total} checks passed "
            f"({self.pass_rate * 100:.0f}%) | step={self.step_id}"
        )

    def first_error(self) -> Optional[ValidationResult]:
        for r in self.results:
            if not r.is_valid:
                return r
        return None


# ---------------------------------------------------------------------------
# DeterministicValidator — main orchestrator
# ---------------------------------------------------------------------------

class DeterministicValidator:
    """
    Orchestrates all three deterministic validation layers for a single execution step.

    Layers run in order:
      1. Sum-of-Parts (if group_col is provided)
      2. Domain Boundary Rules (auto-detected from column names)
      3. Dual-Path Verification (DuckDB SQL vs Pandas)

    A SHA-256 checksum is computed over (table_name, metric_col, where_clause)
    for provenance tracking.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, table_name: str):
        self.con = con
        self.table_name = table_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_aggregation(
        self,
        step_id: str,
        metric_col: str,
        agg_func: str = "SUM",
        group_col: Optional[str] = None,
        where_clause: str = "1=1",
        skip_dual_path: bool = False,
    ) -> ValidationReport:
        """
        Full 3-layer validation for a single aggregation result.

        Args:
            step_id:        Unique identifier for this execution step (e.g. 'step_003').
            metric_col:     The numeric column being aggregated.
            agg_func:       Aggregation function: 'SUM', 'AVG', or 'COUNT'.
            group_col:      If provided, runs the Sum-of-Parts check across this dimension.
            where_clause:   SQL WHERE filter applied to scope the check.
            skip_dual_path: Set True for large tables where pandas pull is prohibitive.

        Returns:
            ValidationReport with per-layer results and overall validity.
        """
        t_start = time.perf_counter()
        results: List[ValidationResult] = []

        # Sanitize metric_col against known table columns
        col_names = self._get_column_names()
        from src.agent.schemas import resolve_metric_column
        metric_col = resolve_metric_column(metric_col, col_names)

        # --- Layer 1: Sum-of-Parts ---
        if group_col:
            sop_result = SumOfPartsRule.check(
                self.con, self.table_name, metric_col, group_col, where_clause
            )
            results.append(sop_result)

        # --- Layer 2: Domain Boundary Rules ---
        col_names = self._get_column_names()
        boundary_rules = DomainBoundaryRules.build_rules_for_table(col_names)
        # Only check the metric column's rules (keep scope tight)
        relevant_rules = [r for r in boundary_rules if r.column == metric_col]
        if relevant_rules:
            boundary_results = DomainBoundaryRules.check_boundaries(
                self.con, self.table_name, relevant_rules, where_clause=where_clause
            )
            results.extend(boundary_results)

        # --- Layer 3: Dual-Path Verification ---
        if not skip_dual_path:
            dp_result = DualPathVerifier.verify(
                self.con, self.table_name, metric_col, agg_func, where_clause
            )
            results.append(dp_result)

        # --- Aggregate ---
        passed = sum(1 for r in results if r.is_valid)
        total = len(results)
        is_valid = all(r.is_valid for r in results)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        checksum = self._compute_checksum(metric_col, where_clause)

        return ValidationReport(
            step_id=step_id,
            table_name=self.table_name,
            metric_col=metric_col,
            is_fully_valid=is_valid,
            checks_passed=passed,
            checks_total=total,
            results=results,
            execution_time_ms=round(elapsed_ms, 2),
            checksum_hash=checksum,
        )

    def validate_data_quality_mapping(
        self,
        step_id: str,
        group_entity: str,
        target_entity: str,
        result_df: Optional[pd.DataFrame],
    ) -> ValidationReport:
        """
        Validates a data-quality / mapping consistency query result.
        """
        t_start = time.perf_counter()
        results: List[ValidationResult] = []

        # 1. Schema check: must have group_entity
        schema_res = SchemaAndTypeVerifier.verify(result_df, expected_columns=[group_entity])
        results.append(schema_res)

        # 2. Independent ground-truth conflict check
        mapping_res = DataQualityVerifier.verify_mapping_conflicts(
            self.con, self.table_name, group_entity, target_entity, result_df
        )
        results.append(mapping_res)

        passed = sum(1 for r in results if r.is_valid)
        total = len(results)
        is_valid = all(r.is_valid for r in results)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        checksum = self._compute_checksum(f"{group_entity}->{target_entity}", "HAVING COUNT>1")

        return ValidationReport(
            step_id=step_id,
            table_name=self.table_name,
            metric_col=f"{group_entity}->{target_entity}",
            is_fully_valid=is_valid,
            checks_passed=passed,
            checks_total=total,
            results=results,
            execution_time_ms=round(elapsed_ms, 2),
            checksum_hash=checksum,
        )

    def validate_arithmetic_calculation(
        self,
        step_id: str,
        base_value: float,
        test_value: float,
        claimed_pct_change: float,
    ) -> ValidationReport:
        """
        Validates derived percentage change calculation via independent arithmetic.
        """
        t_start = time.perf_counter()
        results = [
            ArithmeticVerifier.verify_percentage_change(
                base_value=base_value,
                test_value=test_value,
                claimed_pct_change=claimed_pct_change,
            )
        ]
        passed = sum(1 for r in results if r.is_valid)
        total = len(results)
        is_valid = all(r.is_valid for r in results)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        checksum = self._compute_checksum(f"pct_change({base_value},{test_value})", f"{claimed_pct_change}")

        return ValidationReport(
            step_id=step_id,
            table_name=self.table_name,
            metric_col="percentage_change",
            is_fully_valid=is_valid,
            checks_passed=passed,
            checks_total=total,
            results=results,
            execution_time_ms=round(elapsed_ms, 2),
            checksum_hash=checksum,
        )

    def validate_contract_execution(
        self,
        step_id: str,
        contract: Any,
        result_df: Optional[pd.DataFrame],
    ) -> ValidationReport:
        """
        Independently validates multi-branch and derived measures in an ExecutionContract.
        Extracts raw table slice using contract filters and re-evaluates all measures
        in vectorized Pandas across both Branch A (identifiable orders) and Branch B (null-ID quality metrics).
        """
        t_start = time.perf_counter()
        results: List[ValidationResult] = []

        if result_df is None or result_df.empty:
            results.append(ValidationResult(
                is_valid=False,
                error_type="CALCULATION_MISMATCH",
                error_message="Contract execution produced empty or None result DataFrame",
            ))
            return ValidationReport(
                step_id=step_id,
                table_name=self.table_name,
                metric_col="contract_execution",
                is_fully_valid=False,
                checks_passed=0,
                checks_total=1,
                results=results,
                execution_time_ms=0.0,
                checksum_hash=self._compute_checksum("contract_execution", "empty"),
            )

        # Build WHERE clause from contract filters
        op_sym_map = {
            "gte": ">=",
            "gt": ">",
            "lte": "<=",
            "lt": "<",
            "eq": "=",
            "neq": "!=",
            "is_null": "IS NULL",
            "is_not_null": "IS NOT NULL",
        }
        filter_preds = []
        for f in contract.required_filters:
            raw_op = (f.operator.value if hasattr(f.operator, "value") else str(f.operator)).lower()
            op_str = op_sym_map.get(raw_op, raw_op)
            col_name = f.column.name
            val = f.literal_value
            if "is_null" in raw_op or "is null" in op_str.lower():
                filter_preds.append(f'"{col_name}" IS NULL')
            elif "is_not_null" in raw_op or "not null" in op_str.lower():
                filter_preds.append(f'"{col_name}" IS NOT NULL')
            elif isinstance(val, str):
                filter_preds.append(f'"{col_name}" {op_str} \'{val}\'')
            else:
                filter_preds.append(f'"{col_name}" {op_str} {val}')
        where_cls = " AND ".join(filter_preds) if filter_preds else "1=1"

        slice_q = f'SELECT * FROM "{contract.target_table}" WHERE {where_cls}'
        df_slice: pd.DataFrame = self.con.execute(slice_q).df()

        # ADV-03: Compute missing-sales diagnostics strictly in Pandas on the raw slice
        null_sales_row_count = 0
        orders_with_no_sales = 0
        orders_with_partial_null_sales = 0

        if "sales" in df_slice.columns:
            null_sales_row_count = int(df_slice["sales"].isna().sum())

        if "order_id" in df_slice.columns and "sales" in df_slice.columns:
            df_orders = df_slice[df_slice["order_id"].notna()]
            if not df_orders.empty:
                grouped = df_orders.groupby("order_id")["sales"].agg(
                    non_null_count=lambda s: s.notna().sum(),
                    null_count=lambda s: s.isna().sum(),
                )
                orders_with_no_sales = int((grouped["non_null_count"] == 0).sum())
                orders_with_partial_null_sales = int(
                    ((grouped["non_null_count"] > 0) & (grouped["null_count"] > 0)).sum()
                )

        missing_sales_diag = {
            "orders_with_no_sales": orders_with_no_sales,
            "orders_with_partial_null_sales": orders_with_partial_null_sales,
            "null_sales_row_count": null_sales_row_count,
        }

        # Check each required measure
        actual_measures: Dict[str, Optional[float]] = {}
        expected_measures: Dict[str, Optional[float]] = {}
        row0 = result_df.iloc[0]

        for m in contract.required_measures:
            target_col = m.target_column.name
            alias = m.output_alias
            cond = m.condition_predicate

            if alias not in result_df.columns:
                results.append(ValidationResult(
                    is_valid=False,
                    error_type="SCHEMA_MISMATCH",
                    error_message=f"Expected contract measure alias '{alias}' missing from result columns {list(result_df.columns)}",
                ))
                continue

            # Apply measure condition if present
            sub_df = df_slice
            if cond:
                c_col = cond.column.name
                c_op_l = (cond.operator.value if hasattr(cond.operator, "value") else str(cond.operator)).lower()
                if "not_null" in c_op_l or "not null" in c_op_l:
                    sub_df = sub_df[sub_df[c_col].notna()]
                elif "is_null" in c_op_l or "is null" in c_op_l:
                    sub_df = sub_df[sub_df[c_col].isna()]

            fn_name = m.function.value if hasattr(m.function, "value") else str(m.function)
            fn_upper = fn_name.upper()

            if fn_upper in {"COUNT_DISTINCT", "COUNT(DISTINCT)", "DISTINCT_COUNT"}:
                expected_val = float(sub_df[target_col].nunique())
            elif fn_upper == "SUM":
                s = sub_df[target_col].sum(min_count=1)
                expected_val = float(s) if pd.notna(s) else None
            elif fn_upper == "COUNT":
                expected_val = float(len(sub_df))
            elif fn_upper == "AVG":
                m_val = sub_df[target_col].mean()
                expected_val = float(m_val) if len(sub_df) > 0 and pd.notna(m_val) else None
            else:
                expected_val = None

            expected_measures[alias] = expected_val
            actual_raw = row0[alias]
            actual_val = float(actual_raw) if pd.notna(actual_raw) else None
            actual_measures[alias] = actual_val

            # Compare actual vs expected preserving NULL semantics
            if expected_val is None:
                if actual_val is not None and not (isinstance(actual_val, float) and math.isnan(actual_val)):
                    results.append(ValidationResult(
                        is_valid=False,
                        error_type="CALCULATION_MISMATCH",
                        error_message=f"Contract measure mismatch on '{alias}': expected NULL, got {actual_val}",
                        audit_details={"alias": alias, "actual": actual_val, "expected": None},
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        audit_details={"alias": alias, "actual": None, "expected": None},
                    ))
            else:
                if actual_val is None or (isinstance(actual_val, float) and math.isnan(actual_val)):
                    results.append(ValidationResult(
                        is_valid=False,
                        error_type="CALCULATION_MISMATCH",
                        error_message=f"Contract measure mismatch on '{alias}': expected {expected_val}, got NULL",
                        audit_details={"alias": alias, "actual": None, "expected": expected_val},
                    ))
                else:
                    diff = abs(actual_val - expected_val)
                    if diff > 1e-4:
                        results.append(ValidationResult(
                            is_valid=False,
                            error_type="CALCULATION_MISMATCH",
                            error_message=(
                                f"Contract measure mismatch on '{alias}': actual SQL={actual_val:,.4f}, "
                                f"independent Pandas={expected_val:,.4f}, Δ={diff:,.6f}"
                            ),
                            audit_details={"alias": alias, "actual": actual_val, "expected": expected_val, "diff": diff},
                        ))
                    else:
                        results.append(ValidationResult(
                            is_valid=True,
                            audit_details={"alias": alias, "actual": actual_val, "expected": expected_val},
                        ))

        # Check required derived measures
        for dm in contract.required_derived_measures:
            d_alias = dm.output_alias
            num_alias = dm.numerator_source_alias
            den_alias = dm.denominator_source_alias

            if d_alias not in result_df.columns:
                results.append(ValidationResult(
                    is_valid=False,
                    error_type="SCHEMA_MISMATCH",
                    error_message=f"Expected derived measure alias '{d_alias}' missing from result columns",
                ))
                continue

            # ADV-04: Calculate expected derived measure strictly from independent Pandas expected_measures
            num_exp = expected_measures.get(num_alias)
            den_exp = expected_measures.get(den_alias)

            # Preserve NULL semantics:
            # - Zero valid orders (den_exp is None or den_exp == 0.0) -> AOV NULL
            # - Revenue NULL (num_exp is None) -> AOV NULL
            # - Otherwise -> num_exp / den_exp
            if num_exp is None or den_exp is None or den_exp == 0.0:
                expected_derived = None
            else:
                expected_derived = num_exp / den_exp

            actual_d_raw = row0[d_alias]
            actual_derived = float(actual_d_raw) if pd.notna(actual_d_raw) else None

            if expected_derived is None:
                if actual_derived is not None and not (isinstance(actual_derived, float) and math.isnan(actual_derived)):
                    results.append(ValidationResult(
                        is_valid=False,
                        error_type="CALCULATION_MISMATCH",
                        error_message=f"Contract derived measure mismatch on '{d_alias}': expected NULL, got {actual_derived}",
                        audit_details={"alias": d_alias, "actual": actual_derived, "expected": None},
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        audit_details={"alias": d_alias, "actual": None, "expected": None},
                    ))
            else:
                if actual_derived is None or (isinstance(actual_derived, float) and math.isnan(actual_derived)):
                    results.append(ValidationResult(
                        is_valid=False,
                        error_type="CALCULATION_MISMATCH",
                        error_message=f"Contract derived measure mismatch on '{d_alias}': expected {expected_derived:,.4f}, got NULL",
                        audit_details={"alias": d_alias, "actual": None, "expected": expected_derived},
                    ))
                else:
                    diff = abs(actual_derived - expected_derived)
                    if diff > 1e-4:
                        results.append(ValidationResult(
                            is_valid=False,
                            error_type="CALCULATION_MISMATCH",
                            error_message=(
                                f"Contract derived measure mismatch on '{d_alias}': actual SQL={actual_derived:,.4f}, "
                                f"independent Pandas expected={expected_derived:,.4f} ({num_exp} / {den_exp}), Δ={diff:,.6f}"
                            ),
                            audit_details={"alias": d_alias, "actual": actual_derived, "expected": expected_derived, "diff": diff},
                        ))
                    else:
                        results.append(ValidationResult(
                            is_valid=True,
                            audit_details={"alias": d_alias, "actual": actual_derived, "expected": expected_derived},
                        ))

        passed = sum(1 for r in results if r.is_valid)
        total = len(results)
        is_valid = (total > 0 and all(r.is_valid for r in results))
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        unit_id = getattr(contract, "execution_unit_id", "unit")
        checksum = self._compute_checksum(unit_id, where_cls)

        return ValidationReport(
            step_id=step_id,
            table_name=self.table_name,
            metric_col="contract_execution",
            is_fully_valid=is_valid,
            checks_passed=passed,
            checks_total=total,
            results=results,
            execution_time_ms=round(elapsed_ms, 2),
            checksum_hash=checksum,
            missing_sales_diagnostics=missing_sales_diag,
        )

    def validate_result_dataframe(
        self,
        step_id: str,
        result_df: pd.DataFrame,
        metric_col: str,
        group_col: Optional[str] = None,
    ) -> ValidationReport:
        """
        Convenience wrapper: validates a Pandas DataFrame result by registering
        it as a temporary DuckDB table and running all checks.
        """
        temp_table = f"_val_temp_{step_id.replace('-', '_')}"
        try:
            self.con.register(temp_table, result_df)
            report = self.validate_aggregation(
                step_id=step_id,
                metric_col=metric_col,
                group_col=group_col,
                skip_dual_path=True,  # DataFrame already IS the pandas path
            )
            # Run domain boundary check on the result_df directly
            if metric_col in result_df.columns:
                violations = self._check_df_boundaries(result_df, metric_col)
                report.results.extend(violations)
                report.checks_total += len(violations)
                report.checks_passed += sum(1 for r in violations if r.is_valid)
                report.is_fully_valid = all(r.is_valid for r in report.results)
            return report
        finally:
            try:
                self.con.unregister(temp_table)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_column_names(self) -> List[str]:
        # Use DESCRIBE — works with views and quoted/special table names.
        rows = self.con.execute(f'DESCRIBE "{self.table_name}"').fetchall()
        return [r[0] for r in rows]  # DESCRIBE: col 0 = column_name

    def _compute_checksum(self, metric_col: str, where_clause: str) -> str:
        payload = json.dumps({
            "table": self.table_name,
            "metric": metric_col,
            "filter": where_clause,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _check_df_boundaries(
        self, df: pd.DataFrame, metric_col: str
    ) -> List[ValidationResult]:
        """Fast vectorised boundary check directly on a DataFrame (no SQL round-trip)."""
        results: List[ValidationResult] = []
        col_lower = metric_col.lower()
        series = df[metric_col].dropna()

        # Non-negativity
        from src.validation.invariant_rules import NON_NEGATIVE_COLUMNS
        if any(kw in col_lower for kw in NON_NEGATIVE_COLUMNS):
            neg_count = int((series < 0).sum())
            if neg_count > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    error_type="BOUNDARY_ERROR",
                    error_message=(
                        f"'{metric_col}' has {neg_count} negative value(s) in result set. "
                        f"Min found: {series.min():.4f}"
                    ),
                    audit_details={"negative_count": neg_count, "min_value": float(series.min())},
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    audit_details={"rule": "NON_NEGATIVITY", "column": metric_col, "violations": 0},
                ))

        return results
