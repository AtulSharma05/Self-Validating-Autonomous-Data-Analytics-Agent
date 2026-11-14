"""Data Ingestion Engine for tabular datasets using in-memory DuckDB."""

import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import duckdb
import pandas as pd

from src.config import settings


def sanitize_identifier(name: str) -> str:
    """Sanitizes table and column names to prevent SQL injection and syntax errors."""
    clean = re.sub(r"[^\w]+", "_", name.strip().lower())
    clean = re.sub(r"^_+|_+$", "", clean)
    if not clean or clean[0].isdigit():
        clean = f"t_{clean}"
    return clean


class DataIngestionEngine:
    """Manages tabular dataset ingestion into an isolated in-memory DuckDB session."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.con = duckdb.connect(database=":memory:")
        # Configure memory limits
        self.con.execute(f"SET max_memory = '{settings.duckdb_max_memory}'")
        self.loaded_tables: Dict[str, int] = {}

    def load_dataset(self, file_path: str, table_name: Optional[str] = None) -> Dict[str, any]:
        """Loads a CSV, Parquet, or Excel file into an in-memory DuckDB table.
        
        Args:
            file_path: Path to dataset file.
            table_name: Optional explicit table name. Defaults to sanitized file stem.
            
        Returns:
            Dict containing registered table_name and row_count.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at {path}")

        raw_name = table_name or path.stem
        target_table = sanitize_identifier(raw_name)
        ext = path.suffix.lower()

        if ext == ".csv":
            try:
                # DuckDB zero-copy auto-detect
                self.con.execute(
                    f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM read_csv_auto('{path}', normalize_names=True)"
                )
            except Exception as e:
                err_msg = str(e).lower()
                if any(k in err_msg for k in ["unicode", "encoding", "byte sequence", "not utf-8", "csv error"]):
                    loaded = False
                    # Tier 2: DuckDB native latin-1
                    try:
                        self.con.execute(
                            f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM read_csv_auto('{path}', normalize_names=True, encoding='latin-1')"
                        )
                        loaded = True
                    except Exception:
                        pass

                    # Tier 3: Pandas with Windows CP1252 / Latin-1 / UTF-8-sig
                    if not loaded:
                        for enc in ["cp1252", "latin1", "utf-8-sig"]:
                            try:
                                df = pd.read_csv(path, encoding=enc)
                                df.columns = [sanitize_identifier(str(col)) for col in df.columns]
                                self.con.register(f"_temp_{target_table}", df)
                                self.con.execute(f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM _temp_{target_table}")
                                self.con.unregister(f"_temp_{target_table}")
                                loaded = True
                                break
                            except Exception:
                                pass

                    # Tier 4: Pandas with character replacement
                    if not loaded:
                        df = pd.read_csv(path, encoding="latin1", encoding_errors="replace")
                        df.columns = [sanitize_identifier(str(col)) for col in df.columns]
                        self.con.register(f"_temp_{target_table}", df)
                        self.con.execute(f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM _temp_{target_table}")
                        self.con.unregister(f"_temp_{target_table}")
                else:
                    raise
        elif ext == ".parquet":
            self.con.execute(
                f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM read_parquet('{path}')"
            )
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            # Normalize column names in DataFrame
            df.columns = [sanitize_identifier(str(col)) for col in df.columns]
            self.con.register(f"_temp_{target_table}", df)
            self.con.execute(f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM _temp_{target_table}")
            self.con.unregister(f"_temp_{target_table}")
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: .csv, .parquet, .xlsx, .xls")

        row_count = self.con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
        self.loaded_tables[target_table] = row_count

        # Automatically normalize any string date columns to native DuckDB DATE
        self._normalize_date_columns(target_table)

        return {
            "table_name": target_table,
            "row_count": row_count,
            "file_source": str(path)
        }

    def _normalize_date_columns(self, target_table: str):
        """Detects date/time VARCHAR columns and converts them to native DATE in DuckDB."""
        try:
            col_info = self.con.execute(f'DESCRIBE "{target_table}"').fetchall()
            for r in col_info:
                col_name, col_type = r[0], r[1].upper()
                if any(k in col_name.lower() for k in ["date", "time", "timestamp"]) and "VARCHAR" in col_type:
                    # Test if column values can be parsed as dates
                    parse_test_q = f'''
                        SELECT 
                            COUNT(*) AS total_non_null,
                            COUNT(COALESCE(
                                TRY_CAST("{col_name}" AS DATE),
                                TRY_STRPTIME("{col_name}", '%m/%d/%Y'),
                                TRY_STRPTIME("{col_name}", '%d/%m/%Y'),
                                TRY_STRPTIME("{col_name}", '%Y/%m/%d'),
                                TRY_STRPTIME("{col_name}", '%Y-%m-%d')
                            )) AS valid_parsed
                        FROM "{target_table}"
                        WHERE "{col_name}" IS NOT NULL
                    '''
                    total_non_null, valid_parsed = self.con.execute(parse_test_q).fetchone()
                    if total_non_null > 0 and (valid_parsed / total_non_null) >= 0.90:
                        self.con.execute(f'''
                            ALTER TABLE "{target_table}" ALTER COLUMN "{col_name}" TYPE DATE USING COALESCE(
                                TRY_CAST("{col_name}" AS DATE),
                                TRY_STRPTIME("{col_name}", '%m/%d/%Y'),
                                TRY_STRPTIME("{col_name}", '%d/%m/%Y'),
                                TRY_STRPTIME("{col_name}", '%Y/%m/%d'),
                                TRY_STRPTIME("{col_name}", '%Y-%m-%d')
                            )
                        ''')
        except Exception:
            pass  # Non-fatal if schema alteration fails

    def list_tables(self) -> List[str]:
        """Returns list of currently loaded table names."""
        rows = self.con.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]

    def get_row_count(self, table_name: str) -> int:
        """Returns row count for a given table."""
        sanitized = sanitize_identifier(table_name)
        return self.con.execute(f"SELECT COUNT(*) FROM {sanitized}").fetchone()[0]

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns results as a pandas DataFrame."""
        return self.con.execute(query).df()

    def close(self):
        """Closes the in-memory DuckDB connection."""
        self.con.close()
