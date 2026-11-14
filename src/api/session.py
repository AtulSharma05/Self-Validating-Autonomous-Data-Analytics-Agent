"""
In-memory session manager for isolated DuckDB analytics sessions.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

from src.agent.schemas import DatasetProfile
from src.ingestion.loader import DataIngestionEngine


class SessionManager:
    """
    Thread-safe registry of active user analytics sessions.
    Each session retains its isolated in-memory DuckDB connection,
    statistical profile, and analysis provenance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions = {}
                cls._instance._profiles = {}
                cls._instance._results = {}
        return cls._instance

    def create_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())[:8]
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = DataIngestionEngine(session_id=sid)
        return sid

    def get_engine(self, session_id: str) -> DataIngestionEngine:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = DataIngestionEngine(session_id=session_id)
            return self._sessions[session_id]

    def set_profile(self, session_id: str, profile: DatasetProfile) -> None:
        with self._lock:
            self._profiles[session_id] = profile

    def get_profile(self, session_id: str) -> Optional[DatasetProfile]:
        with self._lock:
            return self._profiles.get(session_id)

    def store_result(self, session_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._results[session_id] = result

    def get_last_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._results.get(session_id)

    def close_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                try:
                    self._sessions[session_id].close()
                except Exception:
                    pass
                del self._sessions[session_id]
            self._profiles.pop(session_id, None)
            self._results.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear_all(self) -> None:
        with self._lock:
            for eng in self._sessions.values():
                try:
                    eng.close()
                except Exception:
                    pass
            self._sessions.clear()
            self._profiles.clear()
            self._results.clear()


session_manager = SessionManager()

