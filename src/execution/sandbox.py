"""
Sandboxed Execution Engine — Tier 1 (In-Process DuckDB + Restricted Python exec).

Architecture:
  Tier 1 (implemented here):
    - SQL queries run inside an in-memory DuckDB connection with:
        * Hard timeout via concurrent.futures.ThreadPoolExecutor
        * Memory cap set at DuckDB engine level
        * No network access (DuckDB in-memory has no I/O)
    - Python snippets run via exec() in an isolated namespace with:
        * Hard timeout via concurrent.futures.ThreadPoolExecutor
        * Restricted builtins (no open, no __import__, no os)
        * Access to: pandas, numpy, duckdb connection — that's it

  Tier 2 (Docker subprocess — future):
    - For arbitrary untrusted code with network disabled and memory capped.
    - Stub left in place for future integration.

Result contract:
  Every execution produces an ExecutionResult with:
    - success, result_df, scalar_result, stdout, stderr, timing, error info
"""

from __future__ import annotations

import io
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from src.config import settings
from src.execution.error_classifier import ClassifiedError, ErrorClassifier
from src.agent.air.contract import ExecutionContract, ExecutionReceipt


# ---------------------------------------------------------------------------
# ExecutionResult — the standard output of every execution
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Standardised result container for every sandboxed execution.

    Fields:
        success:          True if the code ran without exception and produced output.
        result_df:        The primary tabular result (may be None for scalar queries).
        scalar_result:    Single numeric result (populated when query returns 1 row × 1 col).
        stdout:           Captured standard output from Python exec.
        stderr:           Captured standard error / traceback.
        execution_time_ms: Wall-clock time in milliseconds.
        query_executed:   The exact SQL or code string that was run.
        row_count:        Number of rows in result_df (0 if None).
        error:            Classified error (None if success=True).
    """
    success: bool
    result_df: Optional[pd.DataFrame] = None
    scalar_result: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    query_executed: str = ""
    row_count: int = 0
    error: Optional[ClassifiedError] = None

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0 or (self.result_df is not None and self.result_df.empty)

    def summary(self) -> str:
        if self.success:
            if self.scalar_result is not None:
                return f"✅ Scalar result: {self.scalar_result:,.4f} ({self.execution_time_ms:.1f}ms)"
            return f"✅ {self.row_count} rows returned ({self.execution_time_ms:.1f}ms)"
        return f"❌ {self.error.error_type.value}: {self.error.raw_message[:120]}"


# ---------------------------------------------------------------------------
# Sandboxed SQL Executor
# ---------------------------------------------------------------------------

class SandboxedSQLExecutor:
    """
    Executes DuckDB SQL queries with hard timeout and captures results.

    Every query runs in a ThreadPoolExecutor so the main thread can enforce
    a wall-clock timeout regardless of what the DuckDB engine is doing.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        timeout_seconds: Optional[int] = None,
    ):
        self.con = con
        self.timeout = timeout_seconds or settings.execution_timeout_seconds

    def execute(
        self,
        query: str,
        receipt: Optional[ExecutionReceipt] = None,
        contract: Optional[ExecutionContract] = None,
    ) -> ExecutionResult:
        """
        Execute a DuckDB SQL query safely.
        When an ExecutionContract is provided, enforces that a valid ExecutionReceipt
        matching the exact code and contract is verified before DuckDB execution.

        Args:
            query:    The SQL string to execute.
            receipt:  Immutable ExecutionReceipt issued by SQLASTConformanceVerifier.
            contract: The active ExecutionContract.

        Returns:
            ExecutionResult with the query output or classified error.
        """
        query = query.strip()

        # Gatekeeper: Refuse execution if contract is present but receipt is missing or mismatched
        if contract is not None:
            if receipt is None:
                classified = ErrorClassifier.classify_message(
                    "Sandbox Gatekeeper Violation: Missing ExecutionReceipt. Query execution blocked."
                )
                return ExecutionResult(
                    success=False,
                    query_executed=query,
                    error=classified,
                    stderr="Sandbox Gatekeeper: Missing ExecutionReceipt. Unvalidated query cannot execute in DuckDB.",
                )
            ok, err = receipt.matches(query, contract)
            if not ok:
                classified = ErrorClassifier.classify_message(
                    f"Sandbox Gatekeeper Violation: {err}"
                )
                return ExecutionResult(
                    success=False,
                    query_executed=query,
                    error=classified,
                    stderr=f"Sandbox Gatekeeper: {err}",
                )

        t_start = time.perf_counter()

        def _run():
            return self.con.execute(query).df()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                try:
                    df: pd.DataFrame = future.result(timeout=self.timeout)
                except FuturesTimeout:
                    future.cancel()
                    raise TimeoutError(
                        f"Query timed out after {self.timeout}s. "
                        "Simplify or add stricter filters."
                    )

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            # Detect empty result
            if df.empty:
                classified = ErrorClassifier.classify_empty_result(query)
                return ExecutionResult(
                    success=False,
                    result_df=df,
                    row_count=0,
                    execution_time_ms=round(elapsed_ms, 2),
                    query_executed=query,
                    error=classified,
                )

            # Extract scalar if single-cell result
            scalar = None
            if df.shape == (1, 1):
                val = df.iloc[0, 0]
                if pd.isna(val):
                    # Single-cell is NULL or NaN (e.g. AVG(), SUM() over 0 matching rows)
                    classified = ErrorClassifier.classify_empty_result(query)
                    return ExecutionResult(
                        success=False,
                        result_df=df,
                        scalar_result=None,
                        row_count=0,
                        execution_time_ms=round(elapsed_ms, 2),
                        query_executed=query,
                        error=classified,
                    )
                try:
                    scalar = float(val)
                except (TypeError, ValueError):
                    scalar = None

            return ExecutionResult(
                success=True,
                result_df=df,
                scalar_result=scalar,
                row_count=len(df),
                execution_time_ms=round(elapsed_ms, 2),
                query_executed=query,
            )

        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return ExecutionResult(
                success=False,
                stderr=str(exc),
                execution_time_ms=round(elapsed_ms, 2),
                query_executed=query,
                error=ErrorClassifier.classify_exception(exc),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            tb = traceback.format_exc()
            return ExecutionResult(
                success=False,
                stderr=tb,
                execution_time_ms=round(elapsed_ms, 2),
                query_executed=query,
                error=ErrorClassifier.classify_exception(exc),
            )


# ---------------------------------------------------------------------------
# Sandboxed Python Executor
# ---------------------------------------------------------------------------

# Builtins allowed in sandboxed Python exec
_SAFE_BUILTINS: Dict[str, Any] = {
    name: getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__, name, None)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "int", "isinstance", "len", "list", "map", "max",
        "min", "print", "range", "round", "set", "sorted", "str",
        "sum", "tuple", "type", "zip", "None", "True", "False",
    ]
}
# Remove None values in case some builtins aren't available
_SAFE_BUILTINS = {k: v for k, v in _SAFE_BUILTINS.items() if v is not None}


class SandboxedPythonExecutor:
    """
    Executes LLM-generated Python snippets in an isolated namespace.

    Security constraints:
      - No file I/O (open, os, pathlib are not in the namespace)
      - No network access (requests, urllib not in namespace)
      - Only pandas, numpy, and the DuckDB connection are injected
      - Hard wall-clock timeout via ThreadPoolExecutor

    Convention: Generated code must assign its primary result to `result_df`
    (a pandas DataFrame) or `result_scalar` (a numeric value).
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        timeout_seconds: Optional[int] = None,
    ):
        self.con = con
        self.timeout = timeout_seconds or settings.execution_timeout_seconds

    def execute(self, code: str, extra_namespace: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute a Python code snippet in a sandboxed namespace.

        Args:
            code:            Python source code as a string.
            extra_namespace: Additional variables to inject (e.g. a pre-loaded DataFrame).

        Returns:
            ExecutionResult with captured stdout and result_df / scalar_result.
        """
        code = code.strip()
        t_start = time.perf_counter()

        # Build the isolated namespace
        import numpy as np
        namespace: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "con": self.con,
            "result_df": None,
            "result_scalar": None,
        }
        if extra_namespace:
            namespace.update(extra_namespace)

        stdout_capture = io.StringIO()

        def _run():
            # Inject a print shim into the namespace so stdout is captured
            def _print(*args, **kwargs):
                kwargs.pop("file", None)
                print(*args, file=stdout_capture, **kwargs)

            namespace["print"] = _print
            exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102

        try:
            # Use a daemon executor — threads are daemon so they die with the process;
            # we do NOT use `with` block because __exit__ would wait for the thread
            # (which may be stuck in an infinite loop) to finish.
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run)
            executor.shutdown(wait=False)   # release the executor without blocking
            try:
                future.result(timeout=self.timeout)
            except FuturesTimeout:
                # Thread is still running but we move on; it will be GC'd eventually
                raise TimeoutError(
                    f"Python execution timed out after {self.timeout}s."
                )

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = ""  # stderr not captured separately; errors go to result.error

            result_df: Optional[pd.DataFrame] = namespace.get("result_df")
            result_scalar = namespace.get("result_scalar")

            # Coerce scalar if result_df is a 1×1 DataFrame
            if result_df is not None and isinstance(result_df, pd.DataFrame):
                if result_df.shape == (1, 1) and result_scalar is None:
                    try:
                        result_scalar = float(result_df.iloc[0, 0])
                    except (TypeError, ValueError):
                        pass

            if isinstance(result_scalar, (int, float)):
                result_scalar = float(result_scalar)
            else:
                result_scalar = None

            # Safety net: If python executed without defining result_df or result_scalar,
            # check if the code was actually a JSON payload containing an SQL query
            if result_df is None and result_scalar is None:
                import json
                try:
                    data = json.loads(code)
                    if isinstance(data, dict):
                        for k in ["sql", "query", "sql_query"]:
                            if k in data and isinstance(data[k], str) and data[k].strip():
                                sql_str = data[k].strip()
                                df_res = self.con.execute(sql_str).df()
                                return ExecutionResult(
                                    success=True,
                                    result_df=df_res,
                                    row_count=len(df_res),
                                    execution_time_ms=round(elapsed_ms, 2),
                                    query_executed=sql_str,
                                )
                except Exception:
                    pass

            row_count = len(result_df) if isinstance(result_df, pd.DataFrame) else 0

            # Empty result detection
            if isinstance(result_df, pd.DataFrame) and result_df.empty and result_scalar is None:
                classified = ErrorClassifier.classify_empty_result(code[:200])
                return ExecutionResult(
                    success=False,
                    result_df=result_df,
                    stdout=captured_stdout,
                    stderr=captured_stderr,
                    row_count=0,
                    execution_time_ms=round(elapsed_ms, 2),
                    query_executed=code,
                    error=classified,
                )

            return ExecutionResult(
                success=True,
                result_df=result_df if isinstance(result_df, pd.DataFrame) else None,
                scalar_result=result_scalar,
                stdout=captured_stdout,
                stderr=captured_stderr,
                row_count=row_count,
                execution_time_ms=round(elapsed_ms, 2),
                query_executed=code,
            )

        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return ExecutionResult(
                success=False,
                stderr=str(exc),
                execution_time_ms=round(elapsed_ms, 2),
                query_executed=code,
                error=ErrorClassifier.classify_exception(exc),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            tb = traceback.format_exc()
            return ExecutionResult(
                success=False,
                stderr=tb,
                stdout=stdout_capture.getvalue() if stdout_capture else "",
                execution_time_ms=round(elapsed_ms, 2),
                query_executed=code,
                error=ErrorClassifier.classify_exception(exc),
            )
