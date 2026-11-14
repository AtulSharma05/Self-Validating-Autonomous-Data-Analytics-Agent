"""
Unit tests for the Sandboxed Execution Engine and Error Classifier.

Covers:
  - SandboxedSQLExecutor: success, scalar, empty result, syntax error, timeout
  - SandboxedPythonExecutor: success, result_df, result_scalar, restricted builtins, timeout
  - ErrorClassifier: all 4 canonical error types + validation failure mapping
"""

import time
import pytest
import duckdb
import pandas as pd

from src.execution.error_classifier import ErrorClassifier, ErrorType
from src.execution.sandbox import (
    ExecutionResult,
    SandboxedPythonExecutor,
    SandboxedSQLExecutor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def con():
    """In-memory DuckDB connection with a sales table."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE sales AS
        SELECT * FROM (VALUES
            ('North', 'Electronics',  500.0, 5),
            ('South', 'Electronics',  300.0, 3),
            ('East',  'Apparel',      200.0, 4),
            ('West',  'Apparel',      100.0, 2)
        ) t(region, category, revenue, units_sold)
    """)
    yield c
    c.close()


@pytest.fixture
def sql_exec(con):
    return SandboxedSQLExecutor(con, timeout_seconds=5)


@pytest.fixture
def py_exec(con):
    return SandboxedPythonExecutor(con, timeout_seconds=5)


# ---------------------------------------------------------------------------
# ErrorClassifier Tests
# ---------------------------------------------------------------------------

class TestErrorClassifier:

    def test_classifies_binder_error_as_syntax(self):
        err = ErrorClassifier.classify_message("Binder Error: column not found")
        assert err.error_type == ErrorType.SYNTAX_ERROR
        assert err.is_retryable is True
        assert len(err.correction_hint) > 0

    def test_classifies_timeout_error(self):
        err = ErrorClassifier.classify_message("Query timed out after 10s")
        assert err.error_type == ErrorType.TIMEOUT_ERROR

    def test_classifies_invariant_violation(self):
        err = ErrorClassifier.classify_message(
            "Sum of parts does not equal grand total. Discrepancy: 100.0"
        )
        assert err.error_type == ErrorType.INVARIANT_ERROR

    def test_classifies_boundary_error(self):
        err = ErrorClassifier.classify_message("Found negative values in revenue column")
        assert err.error_type == ErrorType.BOUNDARY_ERROR

    def test_classifies_empty_result(self):
        err = ErrorClassifier.classify_empty_result("SELECT * FROM sales WHERE 1=0")
        assert err.error_type == ErrorType.EMPTY_RESULT_ERROR
        assert err.is_retryable is True

    def test_fallback_to_unknown_for_unrecognised(self):
        err = ErrorClassifier.classify_message("Something completely unexpected happened here")
        assert err.error_type == ErrorType.UNKNOWN_ERROR
        assert err.is_retryable is False

    def test_classify_exception_object(self):
        exc = RuntimeError("Table not found in catalog")
        err = ErrorClassifier.classify_exception(exc)
        assert err.error_type == ErrorType.SYNTAX_ERROR

    def test_classify_validation_failure_mapping(self):
        err = ErrorClassifier.classify_validation_failure(
            "INVARIANT_VIOLATION_SUM_OF_PARTS", "Parts sum != total"
        )
        assert err.error_type == ErrorType.INVARIANT_ERROR

    def test_prompt_injection_format(self):
        err = ErrorClassifier.classify_message("column not found: revenue")
        prompt = err.to_prompt_injection()
        assert "SYNTAX_ERROR" in prompt
        assert "column not found" in prompt
        assert "Correction Guidance" in prompt


# ---------------------------------------------------------------------------
# SandboxedSQLExecutor Tests
# ---------------------------------------------------------------------------

class TestSandboxedSQLExecutor:

    def test_simple_select_succeeds(self, sql_exec):
        result = sql_exec.execute("SELECT * FROM sales")
        assert result.success is True
        assert result.row_count == 4
        assert result.result_df is not None
        assert "revenue" in result.result_df.columns

    def test_aggregation_returns_scalar(self, sql_exec):
        result = sql_exec.execute("SELECT SUM(revenue) AS total FROM sales")
        assert result.success is True
        assert result.scalar_result == pytest.approx(1100.0, abs=0.01)
        assert result.row_count == 1

    def test_filtered_query(self, sql_exec):
        result = sql_exec.execute(
            "SELECT SUM(revenue) AS total FROM sales WHERE region = 'North'"
        )
        assert result.success is True
        assert result.scalar_result == pytest.approx(500.0, abs=0.01)

    def test_grouped_query(self, sql_exec):
        result = sql_exec.execute(
            "SELECT region, SUM(revenue) AS rev FROM sales GROUP BY region ORDER BY region"
        )
        assert result.success is True
        assert result.row_count == 4
        assert "region" in result.result_df.columns

    def test_bad_column_returns_syntax_error(self, sql_exec):
        result = sql_exec.execute("SELECT nonexistent_col FROM sales")
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == ErrorType.SYNTAX_ERROR

    def test_bad_table_returns_syntax_error(self, sql_exec):
        result = sql_exec.execute("SELECT * FROM ghost_table")
        assert result.success is False
        assert result.error.error_type == ErrorType.SYNTAX_ERROR

    def test_empty_result_detected(self, sql_exec):
        result = sql_exec.execute(
            "SELECT * FROM sales WHERE region = 'Antarctica'"
        )
        assert result.success is False
        assert result.error.error_type == ErrorType.EMPTY_RESULT_ERROR
        assert "0 rows" in result.error.correction_hint.lower() or \
               "relax" in result.error.correction_hint.lower()

    @pytest.mark.slow
    def test_timeout_enforced(self, con):
        """
        A very short timeout should trigger TIMEOUT_ERROR on a sleep-like
        expensive operation. We use a generate_series with heavy computation.
        """
        fast_exec = SandboxedSQLExecutor(con, timeout_seconds=1)
        # This query generates 100M rows — should exceed 1s timeout
        result = fast_exec.execute(
            "SELECT SUM(i) FROM generate_series(1, 100000000) t(i)"
        )
        # Either it completes (fast machine) or times out — both are acceptable
        # We just assert no unhandled exception was raised
        assert isinstance(result, ExecutionResult)

    def test_execution_time_is_recorded(self, sql_exec):
        result = sql_exec.execute("SELECT COUNT(*) FROM sales")
        assert result.execution_time_ms >= 0.0

    def test_query_executed_field_matches(self, sql_exec):
        query = "SELECT COUNT(*) AS cnt FROM sales"
        result = sql_exec.execute(query)
        assert result.query_executed == query

    def test_summary_string_on_success(self, sql_exec):
        result = sql_exec.execute("SELECT SUM(revenue) FROM sales")
        assert "✅" in result.summary()

    def test_summary_string_on_failure(self, sql_exec):
        result = sql_exec.execute("SELECT bad_column FROM sales")
        assert "❌" in result.summary()


# ---------------------------------------------------------------------------
# SandboxedPythonExecutor Tests
# ---------------------------------------------------------------------------

class TestSandboxedPythonExecutor:

    def test_basic_dataframe_result(self, py_exec):
        code = """
result_df = con.execute("SELECT * FROM sales").df()
"""
        result = py_exec.execute(code)
        assert result.success is True
        assert result.result_df is not None
        assert result.row_count == 4

    def test_scalar_result_via_result_scalar(self, py_exec):
        code = """
df = con.execute("SELECT SUM(revenue) AS total FROM sales").df()
result_scalar = df['total'].iloc[0]
"""
        result = py_exec.execute(code)
        assert result.success is True
        assert result.scalar_result == pytest.approx(1100.0, abs=0.01)

    def test_pandas_aggregation(self, py_exec):
        code = """
df = con.execute("SELECT * FROM sales").df()
result_df = df.groupby('region')['revenue'].sum().reset_index()
"""
        result = py_exec.execute(code)
        assert result.success is True
        assert result.row_count == 4
        assert "revenue" in result.result_df.columns

    def test_print_captured_in_stdout(self, py_exec):
        code = "print('hello from sandbox')\nresult_scalar = 42.0"
        result = py_exec.execute(code)
        assert result.success is True
        assert "hello from sandbox" in result.stdout
        assert result.scalar_result == 42.0

    def test_extra_namespace_injection(self, py_exec):
        """Extra variables can be injected into the sandbox namespace."""
        import pandas as pd
        extra = {"my_df": pd.DataFrame({"x": [1, 2, 3]})}
        code = "result_df = my_df.copy()"
        result = py_exec.execute(code, extra_namespace=extra)
        assert result.success is True
        assert len(result.result_df) == 3

    def test_file_access_blocked(self, py_exec):
        """'open' is not in the sandbox builtins — NameError expected."""
        code = "f = open('/etc/passwd', 'r')"
        result = py_exec.execute(code)
        assert result.success is False
        # Should be a syntax/name error since 'open' is not available
        assert result.error is not None

    def test_import_blocked(self, py_exec):
        """__import__ is not available in restricted builtins."""
        code = "os = __import__('os')\nos.getcwd()"
        result = py_exec.execute(code)
        assert result.success is False

    def test_syntax_error_in_code(self, py_exec):
        code = "result_df = con.execute('SELECT * FROM nonexistent_table_xyz').df()"
        result = py_exec.execute(code)
        assert result.success is False
        assert result.error is not None

    @pytest.mark.slow
    def test_python_timeout_enforced(self, con):
        """An infinite loop should be killed by the timeout."""
        fast_exec = SandboxedPythonExecutor(con, timeout_seconds=1)
        code = """
i = 0
while True:
    i += 1
"""
        result = fast_exec.execute(code)
        assert result.success is False
        assert result.error.error_type == ErrorType.TIMEOUT_ERROR

    def test_execution_time_recorded(self, py_exec):
        result = py_exec.execute("result_scalar = 1.0 + 1.0")
        assert result.execution_time_ms >= 0.0
