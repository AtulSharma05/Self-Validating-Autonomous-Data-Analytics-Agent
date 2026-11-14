"""
Unit and integration tests for the Benchmark Suite and 5-Way Ablation Study (Phase 9 / Feature 13).
"""

from pathlib import Path
import pytest
import duckdb

from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler
from tests.benchmark.evaluator import AblationEvaluator, BenchmarkItem
from tests.benchmark.runner import BenchmarkRunner

BENCHMARK_JSON = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "benchmark_questions.json"
DATASET_CSV = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


@pytest.fixture
def test_con():
    engine = DataIngestionEngine(session_id="benchmark_test_sess")
    engine.load_dataset(str(DATASET_CSV))
    yield engine.con
    engine.close()


def test_benchmark_questions_file_validity():
    runner = BenchmarkRunner(questions_path=BENCHMARK_JSON)
    assert len(runner.items) >= 10

    for item in runner.items:
        assert item.id.startswith("Q")
        assert len(item.question) > 5
        assert len(item.ground_truth_sql) > 5
        assert item.expected_value is not None


def test_ablation_evaluator_all_systems(test_con):
    profile = AutomatedDataProfiler.profile_dataset(test_con)
    item = BenchmarkItem(
        id="Q_TEST",
        category="DESCRIPTIVE",
        difficulty="Easy",
        question="Total revenue?",
        target_table="sample_retail_sales",
        metric="revenue",
        ground_truth_sql="SELECT CAST(SUM(revenue) AS DOUBLE) FROM sample_retail_sales",
        expected_value=566489.94,
    )

    res_a = AblationEvaluator.run_system_a(test_con, item)
    assert res_a.system_name == "System A: Vanilla LLM"
    assert res_a.is_hallucination is True

    res_b = AblationEvaluator.run_system_b(test_con, item)
    assert res_b.is_accurate is True
    assert res_b.validation_passed is False

    res_c = AblationEvaluator.run_system_c(test_con, item)
    assert res_c.is_accurate is True
    assert res_c.validation_passed is True

    res_d = AblationEvaluator.run_system_d(test_con, item)
    assert res_d.is_accurate is True
    assert res_d.validation_passed is True

    res_e = AblationEvaluator.run_system_e(test_con, item, profile)
    assert res_e.is_accurate is True
    assert res_e.ecs_score is not None
    assert res_e.ecs_score >= 75.0


def test_benchmark_runner_execution_and_markdown_report():
    runner = BenchmarkRunner(questions_path=BENCHMARK_JSON, data_path=DATASET_CSV)
    results = runner.run_benchmark(limit=3)

    assert results["total_questions"] == 3
    assert results["total_trials"] == 15  # 3 questions * 5 systems

    summary = results["summary_table"]
    assert len(summary) == 5  # 5 systems

    system_names = [s["System"] for s in summary]
    assert any("System A" in name for name in system_names)
    assert any("System E" in name for name in system_names)

    md_report = runner.generate_markdown_report(results)
    assert "# 5-Way Ablation Benchmark Evaluation Report" in md_report
    assert "System E" in md_report

