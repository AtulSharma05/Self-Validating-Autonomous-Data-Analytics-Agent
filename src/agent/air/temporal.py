"""Temporal boundary resolution using half-open intervals [start, next_period_start)."""

from __future__ import annotations
import re
from typing import Optional, Tuple
from datetime import datetime, timezone

from src.agent.air.schemas import ColumnRef, ComparisonOperator, DataType, Predicate


class TemporalBoundaryResolver:
    """
    Resolves temporal periods into half-open intervals: [start_date, next_period_start).
    This strictly guarantees that records occurring on the final calendar day
    (including late-night timestamps up to 23:59:59.999) are never excluded
    on DATE, DATETIME, and TIMESTAMP columns.
    """

    QUARTER_MAP = {
        1: ("01-01", "04-01"),
        2: ("04-01", "07-01"),
        3: ("07-01", "10-01"),
        4: ("10-01", "01-01"),  # rolls to next year
    }

    @classmethod
    def resolve_period_to_predicates(
        cls,
        period_str: str,
        time_column: ColumnRef,
        timezone_policy: str = "UTC",
    ) -> Tuple[Predicate, Predicate]:
        """
        Translates a temporal shorthand (e.g., '2023-Q2', '2017', '2023-08')
        into two atomic half-open predicates:
            col >= period_start AND col < next_period_start

        Args:
            period_str: Human/legacy period notation
            time_column: Target temporal ColumnRef
            timezone_policy: Policy for timezone interpretation ('UTC', 'LOCAL', or 'STRICT_UTC')

        Returns:
            (Predicate(GTE, start), Predicate(LT, next_start))

        Raises:
            ValueError: If the period cannot be safely and unambiguously interpreted.
        """
        clean = period_str.strip().upper()

        if timezone_policy not in {"UTC", "LOCAL", "STRICT_UTC"}:
            raise ValueError(f"Unsupported timezone_policy: '{timezone_policy}'. Use 'UTC', 'LOCAL', or 'STRICT_UTC'.")

        # 1. Quarter Pattern e.g. "2023-Q2", "2023_Q2", "Q2 2023"
        m_q1 = re.search(r'\b(\d{4})[-_/\s]?Q([1-4])\b', clean)
        m_q2 = re.search(r'\bQ([1-4])[-_/\s]?(\d{4})\b', clean)
        if m_q1 or m_q2:
            year = int(m_q1.group(1)) if m_q1 else int(m_q2.group(2))
            qtr = int(m_q1.group(2)) if m_q1 else int(m_q2.group(1))

            start_md, next_md = cls.QUARTER_MAP[qtr]
            start_date = f"{year}-{start_md}"
            next_year = year + 1 if qtr == 4 else year
            next_start_date = f"{next_year}-{next_md}"

            return (
                Predicate(column=time_column, operator=ComparisonOperator.GTE, literal_value=start_date),
                Predicate(column=time_column, operator=ComparisonOperator.LT, literal_value=next_start_date),
            )

        # 2. Year Pattern e.g. "2017"
        m_y = re.search(r'^\b(\d{4})\b$', clean)
        if m_y:
            year = int(m_y.group(1))
            start_date = f"{year}-01-01"
            next_start_date = f"{year + 1}-01-01"
            return (
                Predicate(column=time_column, operator=ComparisonOperator.GTE, literal_value=start_date),
                Predicate(column=time_column, operator=ComparisonOperator.LT, literal_value=next_start_date),
            )

        # 3. Year-Month Pattern e.g. "2023-05"
        m_m = re.search(r'^\b(\d{4})[-_](\d{2})\b$', clean)
        if m_m:
            year = int(m_m.group(1))
            month = int(m_m.group(2))
            if not (1 <= month <= 12):
                raise ValueError(f"Invalid month value '{month}' in period '{period_str}'.")

            start_date = f"{year:04d}-{month:02d}-01"
            next_year = year + 1 if month == 12 else year
            next_month = 1 if month == 12 else month + 1
            next_start_date = f"{next_year:04d}-{next_month:02d}-01"
            return (
                Predicate(column=time_column, operator=ComparisonOperator.GTE, literal_value=start_date),
                Predicate(column=time_column, operator=ComparisonOperator.LT, literal_value=next_start_date),
            )

        raise ValueError(
            f"Cannot reliably interpret temporal period string: '{period_str}'. "
            f"Expected explicit quarter (e.g. '2023-Q2'), year ('2017'), or month ('2023-05')."
        )

