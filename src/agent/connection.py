"""
Connection and health checking utilities for LLM providers (Ollama, Gemini, OpenAI).
"""

import json
from typing import Any, Dict, List, Optional
import urllib.request

from src.config import settings


def check_ollama_status(
    base_url: Optional[str] = None,
    target_model: Optional[str] = None,
    timeout: float = 1.5,
) -> Dict[str, Any]:
    """
    Checks whether the local Ollama server is running and accessible,
    and whether the configured target model is pulled and ready.

    Args:
        base_url: Ollama HTTP base URL (defaults to settings.ollama_base_url).
        target_model: Model name to look for (defaults to settings.ollama_model_name).
        timeout: Network timeout in seconds.

    Returns:
        Dict[str, Any]:
            - connected (bool): True if Ollama answered HTTP 200.
            - base_url (str): The target URL checked.
            - target_model (str): The configured model name.
            - model_found (bool): True if target_model is present in pulled models.
            - available_models (List[str]): List of available models on the host.
            - version (Optional[str]): Ollama daemon version if retrieved.
            - error (Optional[str]): Detailed error message if disconnected.
    """
    url = (base_url or settings.ollama_base_url).rstrip("/")
    model = (target_model or settings.ollama_model_name).strip()

    status: Dict[str, Any] = {
        "connected": False,
        "base_url": url,
        "target_model": model,
        "model_found": False,
        "available_models": [],
        "version": None,
        "error": None,
    }

    try:
        # Check /api/tags for model list
        req = urllib.request.Request(
            f"{url}/api/tags",
            headers={"User-Agent": "AnalyticsAgent/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                models_data = payload.get("models", [])
                model_names = [m.get("name", "") for m in models_data]
                status["connected"] = True
                status["available_models"] = model_names

                # Normalize model names for matching
                target_base = model.split(":")[0].lower()
                status["model_found"] = any(
                    m.lower() == model.lower()
                    or m.lower().startswith(f"{model.lower()}:")
                    or (m.split(":")[0].lower() == target_base and ":" not in model)
                    for m in model_names
                )
    except Exception as exc:
        status["connected"] = False
        status["error"] = str(exc)
        return status

    # Try optional version check
    try:
        v_req = urllib.request.Request(
            f"{url}/api/version",
            headers={"User-Agent": "AnalyticsAgent/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(v_req, timeout=timeout) as v_resp:
            if v_resp.status == 200:
                v_data = json.loads(v_resp.read().decode("utf-8"))
                status["version"] = v_data.get("version")
    except Exception:
        pass

    return status

