"""
Main FastAPI application entry point.
"""

from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.connection import check_ollama_status
from src.api.routes.analyze import router as analyze_router
from src.api.routes.upload import router as upload_router
from src.api.session import session_manager
from src.config import settings

app = FastAPI(
    title="Self-Validating Autonomous Data Analytics Agent",
    version="1.0.0",
    description="Neuro-symbolic autonomous agent for diagnostic tabular analytics with deterministic validation and evidence scoring.",
)

# Enable CORS for Streamlit / React / Vue dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register sub-routers
app.include_router(upload_router)
app.include_router(analyze_router)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """System health & configuration status."""
    ollama_info = check_ollama_status()
    return {
        "status": "healthy",
        "model_provider": settings.model_provider,
        "ollama_model": settings.ollama_model_name,
        "ollama_connected": ollama_info["connected"],
        "ollama_status": ollama_info,
        "active_sessions_count": len(session_manager.list_sessions()),
        "duckdb_max_memory": settings.duckdb_max_memory,
        "execution_timeout_seconds": settings.execution_timeout_seconds,
    }


@app.get("/api/ollama/status")
async def get_ollama_status() -> Dict[str, Any]:
    """Checks whether the Ollama instance is reachable and the model is loaded."""
    return check_ollama_status()


@app.get("/api/sessions")
async def list_active_sessions() -> Dict[str, Any]:
    """Lists all active in-memory session identifiers."""
    sessions = session_manager.list_sessions()
    return {
        "active_sessions": sessions,
        "count": len(sessions),
    }


@app.delete("/api/sessions/{session_id}")
async def close_session(session_id: str) -> Dict[str, str]:
    """Terminates an in-memory session and releases memory."""
    session_manager.close_session(session_id)
    return {"message": f"Session '{session_id}' closed successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

