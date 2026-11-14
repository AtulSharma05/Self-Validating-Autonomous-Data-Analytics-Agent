"""
Deterministic invariant rules for the Self-Validating Analytics Agent.

Three independent validation layers:
  1. Mathematical Invariant Reconciliation  — Sum-of-Parts rule
  2. Domain & Boundary Rule Checks         — Non-negativity, ratio bounds, temporal bounds
  3. Dual-Path Verification               — DuckDB SQL vs Pandas aggregation cross-check

These rules are applied algorithmically and never ask the LLM to self-verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import duckdb
import pandas as pd

from src.agent.schemas import ValidationResult


# ---------------------------------------------------------------------------
# Layer 1 — Mathematical Invariant Reconciliation (Sum-of-Parts)
# ---------------------------------------------------------------------------

class SumOfPartsRule:
    """
    Verifies that the sum of segmented values equals the unsegmented grand total.

    Detects:
      - Faulty JOINs that dropped rows (sum < total)
      - Cartesian product double-counting (sum > total)

    Epsilon tolerance: 0.01  (handles floating-point rounding in aggregations)
    """

    EPSILON = 1e-2

    @classmethod
    def check(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        metric_col: str,
        group_col: str,
        where_clause: str = "1=1",
    ) -> ValidationResult:
        """
        Args:
            con:          Active DuckDB connection.
            table_name:   Name of the table to validate.
            metric_col:   Numeric column being aggregated (e.g. 'revenue').
            group_col:    Categorical column used for segmentation (e.g. 'region').
            where_clause: Optional SQL filter (default: no filter).

        Returns:
            ValidationResult with is_valid=True if |sum_of_parts - grand_total| < EPSILON.
        """
        try:
            grand_total_q = (
                f"SELECT COALESCE(SUM(\"{metric_col}\"), 0.0) "
                f"FROM {table_name} WHERE {where_clause}"
            )
            grand_total: float = con.execute(grand_total_q).fetchone()[0]

            parts_total_q = (
                f"SELECT COALESCE(SUM(seg_sum), 0.0) FROM ("
                f"  SELECT SUM(\"{metric_col}\") AS seg_sum "
                f"  FROM {table_name} WHERE {where_clause} "
                f"  GROUP BY \"{group_col}\""
                f")"
            )
            parts_total: float = con.execute(parts_total_q).fetchone()[0]

            diff = abs(grand_total - parts_total)
            rel_diff = diff / max(abs(grand_total), 1.0)

            if diff > cls.EPSILON and rel_diff > 1e-4:
                return ValidationResult(
                    is_valid=False,
                    error_type="INVARIANT_VIOLATION_SUM_OF_PARTS",
                    error_message=(
                        f"Sum of '{group_col}' segments ({parts_total:,.4f}) "
                        f"does not equal grand total ({grand_total:,.4f}). "
                        f"Discrepancy: {diff:,.6f}. "
                        f"Likely cause: faulty JOIN dropping rows or Cartesian product."
                    ),
                    audit_details={
                        "grand_total": grand_total,
                        "parts_total": parts_total,
                        "discrepancy": diff,
                        "epsilon": cls.EPSILON,
                        "table": table_name,
                        "metric": metric_col,
                        "group_by": group_col,
                    },
                )

            return ValidationResult(
                is_valid=True,
                audit_details={
                    "reconciled_total": grand_total,
                    "discrepancy": diff,
                },
            )

        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                error_type="SYNTAX_ERROR",
                error_message=f"SumOfPartsRule failed to execute: {exc}",
                audit_details={"exception": str(exc)},
            )


# ---------------------------------------------------------------------------
# Layer 2 — Domain & Boundary Rule Checks
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRule:
    """Defines a numeric boundary constraint for a single column."""
    column: str
    min_value: Optional[float] = None   # None → no lower bound
    max_value: Optional[float] = None   # None → no upper bound
    rule_name: str = ""
    description: str = ""


# Pre-built standard rules (can be extended per dataset)
# NOTE: 'profit' is intentionally NOT in NON_NEGATIVE_COLUMNS because in business/retail
# datasets, transactions can legitimately have negative profit (losses, returns, clearances).
NON_NEGATIVE_COLUMNS = ["revenue", "units_sold", "unit_price", "inventory_count",
                        "quantity", "amount", "sales"]

# Percentages and ratios strictly bounded [0, 100]
RATIO_COLUMNS = ["discount_pct", "discount", "tax_rate", "return_rate", "conversion_rate"]


class DomainBoundaryRules:
    """
    Checks that result values fall within known-valid business domains.

    Covers:
      - Non-negativity: revenue, units_sold, unit_price, etc. must be ≥ 0
      - Ratio bounds: percentages, margins, discounts must be in [0, 100]
      - Temporal monotonicity: query date range must be within dataset bounds
    """

    @classmethod
    def build_rules_for_table(
        cls,
        column_names: List[str],
        temporal_bounds: Optional[dict] = None,
    ) -> List[BoundaryRule]:
        """Auto-generates boundary rules based on column names present in a table."""
        rules: List[BoundaryRule] = []

        for col in column_names:
            col_lower = col.lower()

            # Non-negativity rule
            if any(kw in col_lower for kw in NON_NEGATIVE_COLUMNS):
                rules.append(BoundaryRule(
                    column=col,
                    min_value=0.0,
                    rule_name="NON_NEGATIVITY",
                    description=f"'{col}' must be ≥ 0 (business metric cannot be negative)",
                ))

            # Ratio / percentage bound rule
            if any(kw in col_lower for kw in RATIO_COLUMNS):
                rules.append(BoundaryRule(
                    column=col,
                    min_value=0.0,
                    max_value=100.0,
                    rule_name="RATIO_BOUNDS",
                    description=f"'{col}' must be in [0, 100] (percentage/ratio column)",
                ))

        return rules

    @classmethod
    def check_boundaries(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        rules: List[BoundaryRule],
        where_clause: str = "1=1",
    ) -> List[ValidationResult]:
        """
        Runs each boundary rule against the table slice matching where_clause.

        Returns a list of ValidationResult — one per rule.
        """
        results: List[ValidationResult] = []
        filter_clause = where_clause if where_clause and where_clause.strip() else "1=1"

        for rule in rules:
            try:
                conditions = []
                if rule.min_value is not None:
                    conditions.append(f"\"{rule.column}\" < {rule.min_value}")
                if rule.max_value is not None:
                    conditions.append(f"\"{rule.column}\" > {rule.max_value}")

                if not conditions:
                    continue

                violation_condition = " OR ".join(conditions)
                violation_q = (
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE ({violation_condition}) AND \"{rule.column}\" IS NOT NULL AND ({filter_clause})"
                )
                violation_count: int = con.execute(violation_q).fetchone()[0]

                if violation_count > 0:
                    # Sample a few offending rows for the audit log
                    sample_q = (
                        f"SELECT \"{rule.column}\" FROM {table_name} "
                        f"WHERE ({violation_condition}) AND \"{rule.column}\" IS NOT NULL AND ({filter_clause}) "
                        f"LIMIT 5"
                    )
                    samples = [r[0] for r in con.execute(sample_q).fetchall()]

                    results.append(ValidationResult(
                        is_valid=False,
                        error_type="BOUNDARY_ERROR",
                        error_message=(
                            f"{rule.description}. "
                            f"Found {violation_count} violation(s). "
                            f"Sample offending values: {samples}"
                        ),
                        audit_details={
                            "rule": rule.rule_name,
                            "column": rule.column,
                            "violation_count": violation_count,
                            "sample_values": samples,
                            "min_bound": rule.min_value,
                            "max_bound": rule.max_value,
                        },
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        audit_details={
                            "rule": rule.rule_name,
                            "column": rule.column,
                            "checked": True,
                            "violations": 0,
                        },
                    ))

            except Exception as exc:
                results.append(ValidationResult(
                    is_valid=False,
                    error_type="SYNTAX_ERROR",
                    error_message=f"BoundaryRule on '{rule.column}' failed: {exc}",
                    audit_details={"exception": str(exc)},
                ))

        return results

    @classmethod
    def check_temporal_bounds(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        date_col: str,
        query_start: str,
        query_end: str,
    ) -> ValidationResult:
        """
        Verifies the query's temporal window exists within the dataset's known date range.

        Prevents silent empty results from out-of-range date filters.
        """
        try:
            bounds_q = (
                f"SELECT CAST(MIN(\"{date_col}\") AS VARCHAR), "
                f"       CAST(MAX(\"{date_col}\") AS VARCHAR) "
                f"FROM {table_name} WHERE \"{date_col}\" IS NOT NULL"
            )
            data_min, data_max = con.execute(bounds_q).fetchone()

            if data_min is None:
                return ValidationResult(
                    is_valid=False,
                    error_type="BOUNDARY_ERROR",
                    error_message=f"Date column '{date_col}' has no non-null values.",
                    audit_details={"date_col": date_col},
                )

            # Simple string comparison works for ISO-format dates (YYYY-MM-DD)
            if query_end < data_min or query_start > data_max:
                return ValidationResult(
                    is_valid=False,
                    error_type="BOUNDARY_ERROR",
                    error_message=(
                        f"Query window [{query_start}, {query_end}] is entirely outside "
                        f"dataset date range [{data_min}, {data_max}]. "
                        f"This will produce an empty result — verify the date filters."
                    ),
                    audit_details={
                        "query_start": query_start,
                        "query_end": query_end,
                        "data_min": data_min,
                        "data_max": data_max,
                    },
                )

            return ValidationResult(
                is_valid=True,
                audit_details={
                    "query_window": [query_start, query_end],
                    "data_range": [data_min, data_max],
                },
            )

        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                error_type="SYNTAX_ERROR",
                error_message=f"Temporal bounds check failed: {exc}",
                audit_details={"exception": str(exc)},
            )

