"""
Unit & integration tests for the FastAPI REST API backend (Phase 7).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from src.api.main import app
from src.api.session import session_manager

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


@pytest.fixture
def client():
    session_manager.clear_all()
    with TestClient(app) as c:
        yield c
    session_manager.clear_all()


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_provider" in data
    assert "active_sessions_count" in data
    assert "ollama_connected" in data
    assert "ollama_status" in data


def test_ollama_status_endpoint(client):
    res = client.get("/api/ollama/status")
    assert res.status_code == 200
    data = res.json()
    assert "connected" in data
    assert "target_model" in data
    assert "base_url" in data


def test_upload_dataset_csv(client):
    with open(DATA_PATH, "rb") as f:
        res = client.post(
            "/api/upload",
            files={"file": ("sample_retail_sales.csv", f, "text/csv")},
            data={"session_id": "api_test_sess"},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "api_test_sess"
    assert data["table_name"] == "sample_retail_sales"
    assert data["row_count"] > 900
    assert "profile" in data
    assert data["profile"]["total_tables"] == 1


def test_get_profile_endpoint(client):
    # Upload first
    with open(DATA_PATH, "rb") as f:
        client.post(
            "/api/upload",
            files={"file": ("sample_retail_sales.csv", f, "text/csv")},
            data={"session_id": "prof_sess"},
        )

    res = client.get("/api/profile/prof_sess")
    assert res.status_code == 200
    data = res.json()
    assert data["total_tables"] == 1
    assert data["tables"][0]["table_name"] == "sample_retail_sales"


def test_upload_invalid_format_fails(client):
    res = client.post(
        "/api/upload",
        files={"file": ("bad_file.txt", b"invalid data", "text/plain")},
    )
    assert res.status_code == 400
    assert "Unsupported format" in res.json()["detail"]


@patch("src.agent.objective_extractor._build_llm")
@patch("src.agent.planner._build_llm")
@patch("src.agent.codegen._build_llm")
def test_analyze_endpoint_e2e(mock_cg_llm, mock_pl_llm, mock_oe_llm, client):
    # Setup LLM mocks
    mock_llm = MagicMock()
    # Objective mock
    mock_oe_llm.return_value = mock_llm
    mock_pl_llm.return_value = mock_llm
    mock_cg_llm.return_value = mock_llm

    obj_resp = MagicMock()
    obj_resp.content = """{
      "analysis_type": "DESCRIPTIVE",
      "target_metric": "SUM(revenue)",
      "target_table": "sample_retail_sales",
      "time_column": "order_date",
      "test_period": null,
      "baseline_period": null,
      "filter_conditions": [],
      "candidate_dimensions": ["region"]
    }"""

    code_resp = MagicMock()
    code_resp.content = """```sql
    SELECT CAST(SUM("revenue") AS DOUBLE) AS total_revenue
    FROM sample_retail_sales
    ```"""

    def mock_invoke(messages):
        # Return obj_resp if it's the objective extraction prompt
        for m in messages:
            content = getattr(m, "content", "")
            if "analysis_type" in content or "AnalysisObjective" in content or "User Question" in content:
                return obj_resp
        return code_resp

    mock_llm.invoke.side_effect = mock_invoke

    # Upload dataset
    with open(DATA_PATH, "rb") as f:
        client.post(
            "/api/upload",
            files={"file": ("sample_retail_sales.csv", f, "text/csv")},
            data={"session_id": "analyze_sess"},
        )

    # Analyze
    res = client.post(
        "/api/analyze",
        json={"session_id": "analyze_sess", "question": "What is total sales?"},
    )

    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "analyze_sess"
    assert data["status"] == "COMPLETED"
    assert len(data["evidence_records"]) >= 1
    assert data["ecs_report"] is not None
    assert "score" in data["ecs_report"]
    assert "Analytical Findings" in data["final_summary"]

    # Test provenance endpoint
    prov_res = client.get("/api/provenance/analyze_sess")
    assert prov_res.status_code == 200
    prov_data = prov_res.json()
    assert prov_data["node_count"] >= 1
    assert prov_data["is_acyclic"] is True

    # Test session cleanup
    del_res = client.delete("/api/sessions/analyze_sess")
    assert del_res.status_code == 200
    assert "closed successfully" in del_res.json()["message"]
