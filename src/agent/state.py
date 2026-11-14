"""
State definition for the Autonomous Data Analytics Agent using LangGraph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    CalculationVerification,
    ClaimEvidence,
    CorrectionAttempt,
    DatasetProfile,
    EvidenceRecord,
    PlanStep,
    RCAReport,
    SemanticCoverageReport,
)
from src.reporting.confidence import ECSReport


class AnalyticsAgentState(TypedDict, total=False):
    """
    Typed dictionary representing the evolving state across LangGraph nodes.
    """
    # Session & metadata
    session_id: str
    user_question: str
    profile: DatasetProfile

    # Extracted objective & plan
    objective: Optional[AnalysisObjective]
    plan: Optional[AnalysisPlan]
    current_step_index: int

    # Current execution & validation artifacts
    current_step: Optional[PlanStep]
    current_code: Optional[str]
    current_code_type: Optional[str]  # "sql" or "python"
    current_exec_result: Optional[Dict[str, Any]]
    current_validation: Optional[Dict[str, Any]]
    
    # Self-correction state
    retry_count: int
    max_retries: int
    error_context: Optional[str]
    unresolved_discrepancy: bool
    correction_history: List[CorrectionAttempt]

    # Accumulated evidence & results
    evidence_records: List[EvidenceRecord]
    calculation_verifications: List[CalculationVerification]
    claim_evidences: List[ClaimEvidence]
    step_results: Dict[str, Any]
    final_summary: Optional[str]
    rca_report: Optional[RCAReport]
    ecs_report: Optional[ECSReport]
    provenance_dag: Optional[Dict[str, Any]]

    # Independent validation & coverage statuses
    semantic_coverage: Optional[SemanticCoverageReport]
    execution_correctness: str
    calculation_correctness: str
    semantic_coverage_status: str
    evidence_sufficiency: str
    evidence_integrity: str
    overall_outcome: str

    # Diagnostic flag
    is_diagnostic: bool
    status: str

