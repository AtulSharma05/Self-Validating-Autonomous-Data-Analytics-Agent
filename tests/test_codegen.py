"""
Unit tests for ObjectiveExtractor and CodeGenerator (Phase 3).
Uses mocked LLMs to test prompt parsing, JSON extraction, and SQLGlot validation without needing a live Ollama instance.
"""

from unittest.mock import MagicMock
import pytest

from src.agent.codegen import CodeGenerator, SQLValidator
from src.agent.objective_extractor import ObjectiveExtractor
from src.agent.schemas import AnalysisObjective, AnalysisType, DatasetProfile, TableProfile, ColumnProfile


@pytest.fixture
def sample_profile():
    return DatasetProfile(
        session_id="test_sess",
        total_tables=1,
        tables=[
            TableProfile(
                table_name="sales",
                row_count=1000,
                column_count=4,
                columns=[
                    ColumnProfile(name="order_date", data_type="DATE", is_temporal=True, min_date="2023-01-01", max_date="2023-12-31"),
                    ColumnProfile(name="region", data_type="VARCHAR", is_categorical=True, sample_values=["North", "South", "East", "West"], distinct_count=4),
                    ColumnProfile(name="product_category", data_type="VARCHAR", is_categorical=True, sample_values=["Electronics", "Apparel"], distinct_count=2),
                    ColumnProfile(name="revenue", data_type="DOUBLE", is_numeric=True, min_value=10.0, max_value=500.0, mean_value=250.0),
                ],
                categorical_columns=["region", "product_category"],
                numeric_columns=["revenue"],
                temporal_columns=["order_date"]
            )
        ]
    )


def test_sqlglot_validation_valid():
    sql = 'SELECT "region", SUM("revenue") FROM sales GROUP BY "region"'
    is_valid, err = SQLValidator.validate(sql)
    assert is_valid is True
    assert err is None


def test_sqlglot_validation_invalid():
    sql = 'SELECT FROM WHERE sales'
    is_valid, err = SQLValidator.validate(sql)
    assert is_valid is False
    assert err is not None


def test_objective_extractor_parsing(sample_profile):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """```json
    {
      "analysis_type": "DIAGNOSTIC",
      "target_metric": "SUM(revenue)",
      "target_table": "sales",
      "time_column": "order_date",
      "test_period": "2023-Q3",
      "baseline_period": "2023-Q2",
      "filter_conditions": [],
      "candidate_dimensions": ["region", "product_category"]
    }
    ```"""
    mock_llm.invoke.return_value = mock_response

    extractor = ObjectiveExtractor(llm=mock_llm)
    objective = extractor.extract(sample_profile, "Why did revenue drop in Q3 compared to Q2?")

    assert objective.analysis_type == AnalysisType.DIAGNOSTIC
    assert objective.target_table == "sales"
    assert objective.target_metric == "SUM(revenue)"
    assert "region" in objective.candidate_dimensions


def test_codegen_sql_parsing(sample_profile):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """```sql
    SELECT "region", CAST(SUM("revenue") AS DOUBLE) AS total_revenue
    FROM sales
    GROUP BY "region"
    ```"""
    mock_llm.invoke.return_value = mock_response

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sales"
    )

    generator = CodeGenerator(llm=mock_llm)
    gen_code = generator.generate(sample_profile, objective, "Calculate total revenue by region")

    assert gen_code.code_type == "sql"
    assert gen_code.is_sql_valid is True
    assert gen_code.is_ready is True
    assert "total_revenue" in gen_code.executable_code


def test_codegen_python_parsing(sample_profile):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '''```python
result_df = con.execute("""
    SELECT "region", CAST(SUM("revenue") AS DOUBLE) as total_rev
    FROM sales
    GROUP BY "region"
""").df()
```'''
    mock_llm.invoke.return_value = mock_response

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sales"
    )

    generator = CodeGenerator(llm=mock_llm)
    gen_code = generator.generate(sample_profile, objective, "Calculate total revenue by region via python")

    assert gen_code.code_type == "python"
    assert gen_code.is_sql_valid is True
    assert gen_code.is_ready is True
    assert "result_df" in gen_code.executable_code


def test_codegen_json_structured_parsing(sample_profile):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '''{
        "thoughts": "The task is to calculate the total revenue for each region.",
        "sql": "SELECT \\"region\\", SUM(CAST(\\"revenue\\" AS DOUBLE)) AS \\"total_revenue\\" FROM sales GROUP BY \\"region\\""
    }'''
    mock_llm.invoke.return_value = mock_response

    objective = AnalysisObjective(
        analysis_type=AnalysisType.DESCRIPTIVE,
        target_metric="SUM(revenue)",
        target_table="sales"
    )

    generator = CodeGenerator(llm=mock_llm)
    gen_code = generator.generate(sample_profile, objective, "Calculate total revenue for each region")

    assert gen_code.code_type == "sql"
    assert gen_code.is_sql_valid is True
    assert gen_code.is_ready is True
    assert "total_revenue" in gen_code.sql
    assert "task" in gen_code.explanation

