"""
End-to-end integration test verifying Phases 1, 2, and 3 working together seamlessly.

Flow tested:
  1. Data Ingestion (DuckDB load of real synthetic retail dataset)
  2. Profiling (Statistical and structural metadata extraction)
  3. Objective Extraction (translating user query to AnalysisObjective schema)
  4. Code Generation & SQLGlot Pre-validation (generating safe DuckDB SQL)
  5. Sandboxed SQL Execution (running query inside isolated DuckDB engine)
  6. 3-Layer Deterministic Validation (Sum-of-parts, Domain Boundaries, Dual-path cross-check)
  7. Error Classifier & Self-Correction Feedback Loop (handling edge-case failures)
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler
from src.agent.objective_extractor import ObjectiveExtractor
from src.agent.codegen import CodeGenerator, SQLValidator
from src.execution.sandbox import SandboxedSQLExecutor
from src.execution.error_classifier import ErrorClassifier, ErrorType
from src.validation.validator import DeterministicValidator

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


def test_complete_phases_1_to_3_pipeline():
    # -------------------------------------------------------------
    # 1. Ingestion (Feature 1)
    # -------------------------------------------------------------
    engine = DataIngestionEngine(session_id="e2e_session")
    meta = engine.load_dataset(str(DATA_PATH))
    assert meta["table_name"] == "sample_retail_sales"
    assert meta["row_count"] > 900

    # -------------------------------------------------------------
    # 2. Automated Profiling (Feature 2)
    # -------------------------------------------------------------
    profile = AutomatedDataProfiler.profile_dataset(engine.con, session_id="e2e_session")
    assert profile.total_tables == 1
    table_prof = profile.tables[0]
    assert table_prof.table_name == "sample_retail_sales"
    assert "revenue" in table_prof.numeric_columns
    assert "region" in table_prof.categorical_columns
    assert "order_date" in table_prof.temporal_columns

    compact_schema = profile.to_compact_llm_context()
    assert len(compact_schema) > 100

    # -------------------------------------------------------------
    # 3. Objective Extraction (Feature 3)
    # -------------------------------------------------------------
    mock_extractor_llm = MagicMock()
    mock_obj_response = MagicMock()
    mock_obj_response.content = """{
      "analysis_type": "COMPARATIVE",
      "target_metric": "SUM(revenue)",
      "target_table": "sample_retail_sales",
      "time_column": "order_date",
      "test_period": "2023",
      "baseline_period": null,
      "filter_conditions": [],
      "candidate_dimensions": ["region", "product_category"]
    }"""
    mock_extractor_llm.invoke.return_value = mock_obj_response

    extractor = ObjectiveExtractor(llm=mock_extractor_llm)
    objective = extractor.extract(profile, "Compare total revenue across regions in 2023")
    assert objective.target_table == "sample_retail_sales"
    assert objective.target_metric == "SUM(revenue)"
    assert "region" in objective.candidate_dimensions

    # -------------------------------------------------------------
    # 4. Code Generation & SQLGlot AST Validation (Feature 5)
    # -------------------------------------------------------------
    mock_codegen_llm = MagicMock()
    mock_code_response = MagicMock()
    mock_code_response.content = """```sql
    SELECT "region", CAST(SUM("revenue") AS DOUBLE) AS region_revenue
    FROM sample_retail_sales
    GROUP BY "region"
    ORDER BY region_revenue DESC
    ```"""
    mock_codegen_llm.invoke.return_value = mock_code_response

    codegen = CodeGenerator(llm=mock_codegen_llm)
    gen_code = codegen.generate(profile, objective, "Calculate revenue by region")

    assert gen_code.code_type == "sql"
    assert gen_code.is_sql_valid is True
    assert gen_code.is_ready is True

    # -------------------------------------------------------------
    # 5. Sandboxed Execution (Feature 6)
    # -------------------------------------------------------------
    sandbox = SandboxedSQLExecutor(engine.con, timeout_seconds=5)
    exec_result = sandbox.execute(gen_code.sql)

    assert exec_result.success is True
    assert exec_result.row_count == 4
    assert exec_result.result_df is not None
    assert "region_revenue" in exec_result.result_df.columns
    assert exec_result.execution_time_ms > 0

    # -------------------------------------------------------------
    # 6. Deterministic Validation (Feature 7)
    # -------------------------------------------------------------
    validator = DeterministicValidator(engine.con, "sample_retail_sales")
    val_report = validator.validate_aggregation(
        step_id="e2e_step_01",
        metric_col="revenue",
        agg_func="SUM",
        group_col="region",
    )

    assert val_report.is_fully_valid is True
    assert val_report.checks_passed == val_report.checks_total
    assert val_report.pass_rate == 1.0
    assert len(val_report.checksum_hash) == 16

    # -------------------------------------------------------------
    # 7. Self-Correction & Error Classifier Trigger Check (Feature 8)
    # -------------------------------------------------------------
    # Simulate a deliberate mistake (nonexistent column) to verify error classifier & prompt injection
    bad_exec = sandbox.execute('SELECT "unknown_column" FROM sample_retail_sales')
    assert bad_exec.success is False
    assert bad_exec.error.error_type == ErrorType.SYNTAX_ERROR
    prompt_injection = bad_exec.error.to_prompt_injection()
    assert "SYNTAX_ERROR" in prompt_injection
    assert "Correction Guidance" in prompt_injection

    engine.close()

