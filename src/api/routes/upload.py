"""
Upload & profiling router for tabular datasets.
"""

from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.agent.schemas import DatasetProfile
from src.api.session import session_manager
from src.ingestion.profiler import AutomatedDataProfiler

router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Uploads a CSV, Parquet, or Excel file and registers it inside an in-memory DuckDB session.
    Automatically profiles the dataset for privacy-preserving analytical reasoning.
    """
    sid = session_manager.create_session(session_id)
    engine = session_manager.get_engine(sid)

    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".parquet", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Must be .csv, .parquet, or .xlsx",
        )

    # Save to temp file
    temp_dir = Path(tempfile.gettempdir()) / "analytics_uploads" / sid
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / file.filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest into DuckDB
        meta = engine.load_dataset(str(temp_file_path), table_name=table_name)

        # Profile dataset
        profile = AutomatedDataProfiler.profile_dataset(engine.con, session_id=sid)
        session_manager.set_profile(sid, profile)

        return {
            "session_id": sid,
            "table_name": meta["table_name"],
            "row_count": meta["row_count"],
            "total_tables": profile.total_tables,
            "profile": profile.model_dump(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        # Clean up temp file
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@router.get("/profile/{session_id}")
async def get_dataset_profile(session_id: str) -> Dict[str, Any]:
    """Retrieves the cached statistical profile schema for an active session."""
    profile = session_manager.get_profile(session_id)
    if not profile:
        # Check if engine exists, compute if needed
        if session_id in session_manager.list_sessions():
            engine = session_manager.get_engine(session_id)
            profile = AutomatedDataProfiler.profile_dataset(engine.con, session_id=session_id)
            session_manager.set_profile(session_id, profile)
        else:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return profile.model_dump()

