"""
Analysis & Provenance router for the Autonomous Analytics Agent.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import create_analytics_graph
from src.agent.state import AnalyticsAgentState
from src.api.session import session_manager
from src.ingestion.profiler import AutomatedDataProfiler

router = APIRouter(prefix="/api", tags=["analytics"])


class AnalyzeRequest(BaseModel):
    session_id: str = Field(description="Active session ID with pre-loaded data")
    question: str = Field(description="Natural-language analytical query")
    max_retries: Optional[int] = Field(default=3, description="Max self-correction iterations")


@router.post("/analyze")
async def analyze_question(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Submits a natural-language query to the autonomous agent.
    Orchestrates planning, code generation, sandboxed execution,
    deterministic validation, self-correction, RCA, and confidence scoring.
    """
    sid = request.session_id
    if sid not in session_manager.list_sessions():
        raise HTTPException(
            status_code=404,
            detail=f"Session '{sid}' not found. Please upload a dataset first.",
        )

    engine = session_manager.get_engine(sid)
    profile = session_manager.get_profile(sid)
    if not profile:
        profile = AutomatedDataProfiler.profile_dataset(engine.con, session_id=sid)
        session_manager.set_profile(sid, profile)

    # Compile and execute LangGraph
    try:
        graph = create_analytics_graph(engine.con)

        initial_state: AnalyticsAgentState = {
            "session_id": sid,
            "user_question": request.question,
            "profile": profile,
            "max_retries": request.max_retries or 3,
        }

        final_state = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline execution failed: {exc}")

    # Format serializable response
    objective_dict = (
        final_state.get("objective").model_dump()
        if final_state.get("objective")
        else None
    )

    plan_dict = (
        final_state.get("plan").model_dump()
        if final_state.get("plan")
        else None
    )

    evidence_list = [
        ev.model_dump() if hasattr(ev, "model_dump") else ev
        for ev in final_state.get("evidence_records", [])
    ]

    rca_dict = (
        final_state.get("rca_report").model_dump()
        if final_state.get("rca_report")
        else None
    )

    ecs_dict = (
        final_state.get("ecs_report").to_dict()
        if final_state.get("ecs_report") and hasattr(final_state.get("ecs_report"), "to_dict")
        else None
    )

    result_payload = {
        "session_id": sid,
        "user_question": request.question,
        "status": final_state.get("status", "COMPLETED"),
        "is_diagnostic": final_state.get("is_diagnostic", False),
        "objective": objective_dict,
        "plan": plan_dict,
        "evidence_records": evidence_list,
        "rca_report": rca_dict,
        "ecs_report": ecs_dict,
        "final_summary": final_state.get("final_summary"),
        "provenance_dag": final_state.get("provenance_dag"),
    }

    session_manager.store_result(sid, result_payload)
    return result_payload


@router.get("/provenance/{session_id}")
async def get_provenance_dag(session_id: str) -> Dict[str, Any]:
    """Retrieves the full NetworkX provenance DAG structure for a session."""
    last_res = session_manager.get_last_result(session_id)
    if not last_res or not last_res.get("provenance_dag"):
        # Return empty DAG skeleton
        return {
            "session_id": session_id,
            "node_count": 0,
            "edge_count": 0,
            "is_acyclic": True,
            "nodes": [],
            "edges": [],
        }

    return last_res["provenance_dag"]


@router.get("/results/{session_id}")
async def get_latest_results(session_id: str) -> Dict[str, Any]:
    """Retrieves the latest analysis results and reports for a session."""
    last_res = session_manager.get_last_result(session_id)
    if not last_res:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis results found for session '{session_id}'.",
        )
    return last_res

