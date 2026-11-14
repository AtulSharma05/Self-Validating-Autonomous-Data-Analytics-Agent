"""
Unit tests for LLM connection and health checking utilities.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.agent.connection import check_ollama_status


def test_check_ollama_status_connected_and_model_found():
    """Verifies successful Ollama connection when configured model is present."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "qwen2.5-coder:7b"},
            {"name": "llama3.1:8b"}
        ]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = check_ollama_status(target_model="qwen2.5-coder:7b")

    assert res["connected"] is True
    assert res["model_found"] is True
    assert "qwen2.5-coder:7b" in res["available_models"]
    assert res["error"] is None


def test_check_ollama_status_connected_but_model_missing():
    """Verifies Ollama is online but target model has not yet been pulled."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "mistral:7b"}
        ]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = check_ollama_status(target_model="qwen2.5-coder:7b")

    assert res["connected"] is True
    assert res["model_found"] is False
    assert res["available_models"] == ["mistral:7b"]
    assert res["error"] is None


def test_check_ollama_status_disconnected():
    """Verifies graceful handling when Ollama daemon is unreachable."""
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("Connection refused")):
        res = check_ollama_status(base_url="http://localhost:11434", target_model="qwen2.5-coder:7b")

    assert res["connected"] is False
    assert res["model_found"] is False
    assert res["available_models"] == []
    assert "Connection refused" in res["error"]

