"""Automated structural and statistical data profiler using DuckDB aggregations."""

from typing import List, Optional
import duckdb

from src.agent.schemas import ColumnProfile, DatasetProfile, TableProfile
from src.ingestion.loader import sanitize_identifier


NUMERIC_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
    "FLOAT", "DOUBLE", "DECIMAL", "REAL"
}

TEMPORAL_TYPES = {
    "DATE", "TIME", "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS"
}


class AutomatedDataProfiler:
    """Computes concise statistical and structural metadata profiles directly from DuckDB tables."""

    @classmethod
    def profile_dataset(cls, con: duckdb.DuckDBPyConnection, session_id: str = "default") -> DatasetProfile:
        """Profiles all registered tables in the given DuckDB connection."""
        tables_res = con.execute("SHOW TABLES").fetchall()
        table_names = [r[0] for r in tables_res if not r[0].startswith("_temp")]

        profiles = [cls.profile_table(con, tbl) for tbl in table_names]
        return DatasetProfile(
            session_id=session_id,
            tables=profiles,
            total_tables=len(profiles)
        )

    @classmethod
    def profile_table(cls, con: duckdb.DuckDBPyConnection, table_name: str) -> TableProfile:
        """Generates a comprehensive structural and statistical profile for a single table."""
        sanitized = sanitize_identifier(table_name)
        
        # 1. Basic Table Dimensions
        row_count = con.execute(f"SELECT COUNT(*) FROM {sanitized}").fetchone()[0]
        # Use DESCRIBE (not PRAGMA table_info) — works with views and quoted names.
        col_info = con.execute(f'DESCRIBE "{sanitized}"').fetchall()
        
        columns: List[ColumnProfile] = []
        pk_candidates = []
        temporal_cols = []
        numeric_cols = []
        categorical_cols = []

        if row_count == 0:
            for col_name, col_type, *_ in col_info:
                columns.append(ColumnProfile(name=col_name, data_type=col_type.upper()))
            return TableProfile(
                table_name=sanitized,
                row_count=0,
                column_count=len(columns),
                columns=columns
            )

        # 2. Per-column profile extraction
        for col_name, raw_type, *_ in col_info:
            col_type = raw_type.upper().split("(")[0].strip()
            is_num = any(t in col_type for t in NUMERIC_TYPES)
            is_temp = any(t in col_type for t in TEMPORAL_TYPES)

            # Basic counts
            stats_query = f"""
                SELECT 
                    COUNT(*) - COUNT("{col_name}") AS null_cnt,
                    COUNT(DISTINCT "{col_name}") AS distinct_cnt
                FROM {sanitized}
            """
            null_cnt, distinct_cnt = con.execute(stats_query).fetchone()
            null_pct = (null_cnt / row_count) * 100.0 if row_count > 0 else 0.0

            # Determine categorical nature
            is_cat = (distinct_cnt <= 30 and not is_num) or (distinct_cnt <= 10 and is_num)

            # Sample values / distinct list
            sample_query = f"""
                SELECT DISTINCT "{col_name}" 
                FROM {sanitized} 
                WHERE "{col_name}" IS NOT NULL 
                LIMIT 5
            """
            sample_rows = con.execute(sample_query).fetchall()
            sample_values = [r[0] for r in sample_rows]

            col_prof = ColumnProfile(
                name=col_name,
                data_type=col_type,
                is_numeric=is_num,
                is_temporal=is_temp,
                is_categorical=is_cat,
                total_count=row_count,
                null_count=null_cnt,
                null_percentage=round(null_pct, 2),
                distinct_count=distinct_cnt,
                sample_values=sample_values
            )

            # Compute numerical distribution metrics
            if is_num and (row_count - null_cnt) > 0:
                num_stats_query = f"""
                    SELECT 
                        CAST(MIN("{col_name}") AS DOUBLE),
                        CAST(MAX("{col_name}") AS DOUBLE),
                        CAST(AVG("{col_name}") AS DOUBLE),
                        CAST(COALESCE(STDDEV("{col_name}"), 0.0) AS DOUBLE),
                        CAST(quantile_cont("{col_name}", 0.25) AS DOUBLE),
                        CAST(quantile_cont("{col_name}", 0.50) AS DOUBLE),
                        CAST(quantile_cont("{col_name}", 0.75) AS DOUBLE)
                    FROM {sanitized}
                    WHERE "{col_name}" IS NOT NULL
                """
                mn, mx, mean, std, q25, q50, q75 = con.execute(num_stats_query).fetchone()
                col_prof.min_value = round(mn, 2) if mn is not None else None
                col_prof.max_value = round(mx, 2) if mx is not None else None
                col_prof.mean_value = round(mean, 2) if mean is not None else None
                col_prof.stddev_value = round(std, 2) if std is not None else None
                col_prof.q25 = round(q25, 2) if q25 is not None else None
                col_prof.q50 = round(q50, 2) if q50 is not None else None
                col_prof.q75 = round(q75, 2) if q75 is not None else None
                numeric_cols.append(col_name)

            # Compute temporal range
            elif is_temp and (row_count - null_cnt) > 0:
                temp_stats_query = f"""
                    SELECT 
                        CAST(MIN("{col_name}") AS VARCHAR),
                        CAST(MAX("{col_name}") AS VARCHAR)
                    FROM {sanitized}
                    WHERE "{col_name}" IS NOT NULL
                """
                min_d, max_d = con.execute(temp_stats_query).fetchone()
                col_prof.min_date = str(min_d) if min_d else None
                col_prof.max_date = str(max_d) if max_d else None
                temporal_cols.append(col_name)

            if is_cat:
                categorical_cols.append(col_name)

            if null_cnt == 0 and distinct_cnt == row_count:
                pk_candidates.append(col_name)

            columns.append(col_prof)

        return TableProfile(
            table_name=sanitized,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            primary_key_candidates=pk_candidates,
            temporal_columns=temporal_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols
        )
