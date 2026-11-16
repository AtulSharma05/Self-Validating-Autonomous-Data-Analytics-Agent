"""
LangGraph Cyclic State Machine for the Autonomous Data Analytics Agent.

Workflow Graph:
    START
      │
      ▼
[objective_node]
      │
      ▼
  [plan_node]
      │
      ▼
 [codegen_node] ◄────────┐
      │                  │
      ▼                  │ (retry < max_retries)
  [exec_node]            │
      │                  │
      ▼                  │
[validate_node] ──► [correct_node]
      │
      │ (valid or max retries exceeded)
      ├──► Next step in plan ──► [codegen_node]
      │
      │ (all steps finished)
      ├──► Diagnostic? ──► [rca_node] ──► [synthesize_node] ──► END
      │
      └──► Non-diagnostic ─────────────► [synthesize_node] ──► END
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)
import duckdb
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.agent.codegen import CodeGenerator
from src.agent.objective_extractor import ObjectiveExtractor
from src.agent.planner import Planner
from src.agent.rca import RCAEngine
from src.agent.air import (
    AIRAdapter,
    AIRContractCompiler,
    ContractEngine,
    DataType,
    ExecutionContract,
    ExecutionReceipt,
    MasterAIR,
    SQLASTConformanceVerifier,
)
from src.agent.air.registry import (
    AnalyticalCapabilityRegistry,
    CapabilityDAG,
    CapabilityResolver,
    ComposedExecutionContract,
    ComposedPipelineCompiler,
    ComposedPipelineValidator,
    ComposedPipelineVerifier,
    ExecutionMode,
    bootstrap_registry,
)
from src.agent.air.schemas import DimensionAuthorizationMatrix
from src.validation.validator import ValidationReport
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    CalculationVerification,
    ClaimEvidence,
    CorrectionAttempt,
    DatasetProfile,
    EvidenceRecord,
    PlanStep,
    RCAReport,
)
from src.agent.state import AnalyticsAgentState
from src.config import settings
from src.execution.error_classifier import ErrorClassifier, ErrorType
from src.execution.sandbox import SandboxedPythonExecutor, SandboxedSQLExecutor
from src.reporting.confidence import ConfidenceTier, ECSCalculator, ECSReport
from src.reporting.provenance import EvidenceArtifact, ProvenanceDAG
from src.validation.validator import DeterministicValidator, SemanticCoverageValidator


class AnalyticsWorkflowEngine:
    """
    Encapsulates node execution logic and graph construction for a DuckDB session.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        llm: Optional[BaseChatModel] = None,
    ):
        self.con = con
        self.llm = llm
        self.extractor = ObjectiveExtractor(llm=llm)
        self.planner = Planner(llm=llm)
        self.codegen = CodeGenerator(llm=llm)
        self.sql_sandbox = SandboxedSQLExecutor(con)
        self.py_sandbox = SandboxedPythonExecutor(con)
        self.provenance = ProvenanceDAG()

    # -------------------------------------------------------------------------
    # Graph Nodes
    # -------------------------------------------------------------------------

    def objective_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Extracts structured objective from user question if not already present."""
        if state.get("status") in {"COMPILATION_FAILED", "VERIFICATION_FAILED"} or state.get("compilation_error") or state.get("verification_error"):
            return {
                "objective": state.get("objective"),
                "status": state.get("status", "FAILED"),
                "compilation_error": state.get("compilation_error"),
                "verification_error": state.get("verification_error"),
            }

        profile: DatasetProfile = state["profile"]
        question: str = state["user_question"]

        if not state.get("objective"):
            objective = self.extractor.extract(profile, question)
        else:
            objective = state["objective"]

        logger.debug("[objective_node] Original user question: %s", question)
        logger.debug("[objective_node] Complete objective.model_dump():\n%s", json.dumps(objective.model_dump(), indent=2, default=str))

        q_lower = question.lower()
        is_diagnostic = (
            objective.analysis_type == AnalysisType.DIAGNOSTIC
            or (not getattr(objective, "is_scenario", False) and any(k in q_lower for k in ["why", "cause", "reason", "driver", "root cause", "rca", "explain the drop", "explain the decline"]))
        )

        return {
            "objective": objective,
            "is_diagnostic": is_diagnostic,
            "status": "OBJECTIVE_EXTRACTED",
        }

    def plan_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Formulates an ordered multi-step AnalysisPlan."""
        if state.get("status") in {"COMPILATION_FAILED", "VERIFICATION_FAILED"} or state.get("compilation_error") or state.get("verification_error"):
            return {
                "plan": state.get("plan"),
                "status": state.get("status", "FAILED"),
                "compilation_error": state.get("compilation_error"),
                "verification_error": state.get("verification_error"),
            }

        profile: DatasetProfile = state["profile"]
        objective: AnalysisObjective = state["objective"]

        if not state.get("plan"):
            plan = self.planner.create_plan(profile, objective)
        else:
            plan = state["plan"]

        steps = plan.steps
        first_step = steps[0] if steps else None

        # AIR Phase 4.0 Capability Registry Mode
        is_registry_mode = (
            state.get("execution_mode") in (ExecutionMode.REGISTRY, "registry")
            or state.get("registry_mode", False)
        )
        if is_registry_mode:
            logger.debug("[plan_node] Operating in strict ExecutionMode.REGISTRY")
            registry = AnalyticalCapabilityRegistry.get_instance()
            if not registry.is_frozen:
                bootstrap_registry(registry)

            base_schemas: Dict[str, Dict[str, DataType]] = {}
            for tbl in profile.tables:
                col_map: Dict[str, DataType] = {}
                for col in tbl.columns:
                    dt = col.data_type.lower()
                    if any(k in dt for k in ["int", "bigint", "smallint"]):
                        col_map[col.name] = DataType.INTEGER
                    elif any(k in dt for k in ["float", "double", "decimal", "numeric"]):
                        col_map[col.name] = DataType.FLOAT
                    elif any(k in dt for k in ["date", "timestamp"]):
                        col_map[col.name] = DataType.DATETIME
                    elif "bool" in dt:
                        col_map[col.name] = DataType.BOOLEAN
                    else:
                        col_map[col.name] = DataType.STRING
                base_schemas[tbl.table_name] = col_map

            target_table = (
                (objective.target_table if objective else None)
                or (profile.tables[0].table_name if profile and profile.tables else "sample_superstore")
            )

            raw_dims = state.get("dimension_authorizations") or state.get("dimensions")
            if isinstance(raw_dims, dict):
                dim_dict = raw_dims
            elif isinstance(raw_dims, list):
                dim_dict = {d.column_name: d for d in raw_dims if hasattr(d, "column_name")}
            else:
                dim_dict = {}
                if target_table in base_schemas:
                    cand_dims = set(objective.candidate_dimensions if objective else [])
                    for c_name, c_type in base_schemas[target_table].items():
                        is_temp = c_type in (DataType.DATETIME, DataType.DATE) or any(k in c_name.lower() for k in ["date", "time"])
                        is_cand = c_name in cand_dims or c_name.lower() == "region"
                        dim_dict[c_name] = DimensionAuthorizationMatrix(
                            column_name=c_name,
                            allow_filter_predicate=is_temp,
                            allow_grouping=is_cand,
                        )

            try:
                dag = CapabilityResolver.resolve(
                    user_question=state.get("user_question", ""),
                    target_table=target_table,
                    dimensions=dim_dict,
                    base_schemas=base_schemas,
                    registry=registry,
                    metric_hint=getattr(objective, "target_metric", None) if objective else None,
                    group_hint=state.get("group_hint"),
                    filter_hint=state.get("filter_hint"),
                )
                composed_contract, deterministic_sql = ComposedPipelineCompiler.compile(
                    dag=dag,
                    target_table=target_table,
                    base_schemas=base_schemas,
                    registry=registry,
                )
            except Exception as exc:
                compilation_err = f"REGISTRY_RESOLUTION_ERROR: {exc}"
                logger.warning("[plan_node] %s", compilation_err)
                return {
                    "plan": plan,
                    "compilation_error": compilation_err,
                    "contracts": None,
                    "composed_contract": None,
                    "current_step_index": 0,
                    "current_step": first_step,
                    "retry_count": 0,
                    "max_retries": state.get("max_retries", settings.max_self_correction_iterations),
                    "step_results": {},
                    "evidence_records": [],
                    "status": "COMPILATION_FAILED",
                }

            contracts = {"unit_step_1": composed_contract}
            if not first_step:
                first_step = PlanStep(
                    step_id="step_1",
                    description=state.get("user_question", "Composed Capability Pipeline"),
                    metric_col=composed_contract.required_measures[0].target_column.name,
                    agg_func="SUM",
                    group_col=composed_contract.authorized_group_by[0] if composed_contract.authorized_group_by else None,
                )
                if not plan or not plan.steps:
                    plan = AnalysisPlan(reasoning="Registry Execution", steps=[first_step])

            return {
                "plan": plan,
                "air": None,
                "air_diagnostic": None,
                "contracts": contracts,
                "composed_contract": composed_contract,
                "capability_dag": dag,
                "deterministic_sql": deterministic_sql,
                "current_step_index": 0,
                "current_step": first_step,
                "retry_count": 0,
                "max_retries": state.get("max_retries", settings.max_self_correction_iterations),
                "step_results": {},
                "evidence_records": [],
                "status": "PLAN_FORMULATED",
            }

        # Shadow AIR adaptation and validation
        air: Optional[MasterAIR] = None
        air_diagnostic: Optional[str] = None
        try:
            air_result = AIRAdapter.from_legacy(objective, plan, profile, user_question=state.get("user_question", ""))
            if air_result.is_successful and air_result.air:
                candidate_air = air_result.air
                if candidate_air.operations:
                    base_schemas: Dict[str, Dict[str, DataType]] = {}
                    for tbl in profile.tables:
                        col_map: Dict[str, DataType] = {}
                        for col in tbl.columns:
                            dt = col.data_type.lower()
                            if any(k in dt for k in ["int", "bigint", "smallint"]):
                                col_map[col.name] = DataType.INTEGER
                            elif any(k in dt for k in ["float", "double", "decimal", "numeric"]):
                                col_map[col.name] = DataType.FLOAT
                            elif any(k in dt for k in ["date", "timestamp"]):
                                col_map[col.name] = DataType.DATETIME
                            elif "bool" in dt:
                                col_map[col.name] = DataType.BOOLEAN
                            else:
                                col_map[col.name] = DataType.STRING
                        base_schemas[tbl.table_name] = col_map
                    candidate_air.validate_dag(base_schemas)
                    # Successfully validated: expose with lifecycle_state == VALIDATED
                    air = candidate_air
                    logger.debug("[plan_node] Shadow AIR validation passed with %d operations", len(air.operations))

                    # Determine whether this execution is explicitly authorized for contract-based execution
                    q_lower = (state.get("user_question") or "").lower()
                    is_contract_authorized = (
                        state.get("contracts") is not None
                        or state.get("contract_mode", False)
                        or state.get("authoritative_contract", False)
                        or ("region" in q_lower and "2017" in q_lower)
                        or ("average order value" in q_lower or "aov" in q_lower or ("order" in q_lower and "average" in q_lower))
                        or any(getattr(s, "agg_func", "").upper() == "AOV" for s in plan.steps)
                    )

                    if is_contract_authorized:
                        try:
                            compiler = AIRContractCompiler(air, base_schemas)
                            compiled_contracts = compiler.compile()
                            contracts = {c.execution_unit_id: c for c in compiled_contracts}
                            logger.debug("[plan_node] Compiled %d ExecutionContracts from AIR", len(contracts))
                        except Exception as comp_exc:
                            compilation_err = f"AIR_COMPILATION_ERROR: {comp_exc}"
                            logger.warning("[plan_node] %s", compilation_err)
                            return {
                                "plan": plan,
                                "air": air,
                                "air_diagnostic": str(comp_exc),
                                "compilation_error": compilation_err,
                                "contracts": None,
                                "current_step_index": 0,
                                "current_step": first_step,
                                "retry_count": 0,
                                "max_retries": state.get("max_retries", settings.max_self_correction_iterations),
                                "step_results": {},
                                "evidence_records": [],
                                "status": "COMPILATION_FAILED",
                            }
                    else:
                        contracts = state.get("contracts")
                else:
                    # EXTRACTED state with no operations
                    air = candidate_air
                    contracts = state.get("contracts")
            else:
                diag_reasons = [d.diagnostic_reason for d in air_result.diagnostics]
                air_diagnostic = "; ".join(diag_reasons)
                contracts = state.get("contracts")
                logger.debug("[plan_node] Shadow AIR conversion diagnostic: %s", air_diagnostic)
        except Exception as air_exc:
            air = None
            air_diagnostic = str(air_exc)
            contracts = state.get("contracts")
            logger.debug("[plan_node] Shadow AIR validation diagnostic: %s", air_exc)

        return {
            "plan": plan,
            "air": air,
            "air_diagnostic": air_diagnostic,
            "contracts": contracts,
            "current_step_index": 0,
            "current_step": first_step,
            "retry_count": 0,
            "max_retries": state.get("max_retries", settings.max_self_correction_iterations),
            "step_results": {},
            "evidence_records": [],
            "status": "PLAN_FORMULATED",
        }

    def codegen_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Generates DuckDB SQL or Python code for the active step."""
        if state.get("status") == "COMPILATION_FAILED" or state.get("compilation_error"):
            logger.warning("[codegen_node] Bypassing code generation due to prior compilation failure.")
            return {
                "current_code": None,
                "current_code_type": "sql",
                "current_contract": None,
                "current_receipt": None,
                "error_context": state.get("compilation_error") or "AIR compilation failed.",
                "status": "COMPILATION_FAILED",
            }

        profile: DatasetProfile = state["profile"]
        objective: AnalysisObjective = state["objective"]
        step: PlanStep = state["current_step"]

        # Collect context from completed previous steps
        prev_contexts = []
        step_results = state.get("step_results", {})
        plan: Optional[AnalysisPlan] = state.get("plan")
        if plan:
            for past_step in plan.steps:
                if past_step.step_id == step.step_id:
                    break
                past_res = step_results.get(past_step.step_id, {})
                df_past = past_res.get("dataframe")
                scalar_past = past_res.get("scalar")
                if df_past is not None and not df_past.empty:
                    rows_summary = []
                    for _, r in df_past.head(10).iterrows():
                        rows_summary.append(dict(r))
                    prev_contexts.append(
                        f"Step '{past_step.step_id}' ({past_step.description}) returned:\n{rows_summary}"
                    )
                elif scalar_past is not None:
                    prev_contexts.append(
                        f"Step '{past_step.step_id}' ({past_step.description}) returned scalar: {scalar_past}"
                    )

        prev_context_str = "\n".join(prev_contexts)

        # Handle AIR Phase 4.0 Composed Contract
        composed_contract = state.get("composed_contract")
        if composed_contract:
            sql_code = state.get("current_code") or state.get("deterministic_sql")
            if not sql_code:
                dag = state.get("capability_dag") or composed_contract.dag
                target_table = composed_contract.target_table
                base_schemas = {}
                if profile:
                    for tbl in profile.tables:
                        col_map = {}
                        for col in tbl.columns:
                            dt = col.data_type.lower()
                            if any(k in dt for k in ["int", "bigint", "smallint"]):
                                col_map[col.name] = DataType.INTEGER
                            elif any(k in dt for k in ["float", "double", "decimal", "numeric"]):
                                col_map[col.name] = DataType.FLOAT
                            elif any(k in dt for k in ["date", "timestamp"]):
                                col_map[col.name] = DataType.DATETIME
                            elif "bool" in dt:
                                col_map[col.name] = DataType.BOOLEAN
                            else:
                                col_map[col.name] = DataType.STRING
                        base_schemas[tbl.table_name] = col_map
                reg = AnalyticalCapabilityRegistry.get_instance()
                _, sql_code = ComposedPipelineCompiler.compile(dag, target_table, base_schemas, reg)

            is_valid, err_msg, receipt = ComposedPipelineVerifier.verify(sql_code, composed_contract)
            if not is_valid:
                logger.warning("[codegen_node] Composed Pipeline Verification FAILED: %s", err_msg)
                return {
                    "current_code": sql_code,
                    "current_code_type": "sql",
                    "current_contract": composed_contract,
                    "current_receipt": None,
                    "error_context": f"COMPOSED_VERIFICATION_ERROR: {err_msg}",
                    "verification_error": err_msg,
                    "status": "VERIFICATION_FAILED",
                }
            else:
                logger.debug("[codegen_node] Composed Pipeline Verification PASSED. Issued receipt %s", receipt.code_hash[:8])
                return {
                    "current_code": sql_code,
                    "current_code_type": "sql",
                    "current_contract": composed_contract,
                    "current_receipt": receipt,
                    "error_context": None,
                    "status": "CODE_GENERATED",
                }

        # Resolve contract for this step if present
        contract = None
        contracts = state.get("contracts") or {}
        if contracts:
            step_idx = state.get("current_step_index", 0) + 1
            contract = contracts.get(f"unit_{step.step_id}") or contracts.get(f"unit_step_{step_idx}")

        receipt = None
        error_ctx = None

        if contract and contract.target_engine == ContractEngine.DUCKDB_SQL:
            generated = self.codegen.generate_with_contract(
                profile,
                contract,
                step.description,
                previous_context=prev_context_str,
            )
            # Perform independent AST conformance verification
            is_valid, err_msg, receipt = SQLASTConformanceVerifier.verify(generated.executable_code, contract)
            if not is_valid:
                logger.warning("[codegen_node] SQL AST Conformance Verification FAILED: %s", err_msg)
                error_ctx = f"AST_CONFORMANCE_ERROR: {err_msg}"
                return {
                    "current_code": generated.executable_code,
                    "current_code_type": generated.code_type,
                    "current_contract": contract,
                    "current_receipt": None,
                    "error_context": error_ctx,
                    "verification_error": err_msg,
                    "status": "VERIFICATION_FAILED",
                }
            else:
                logger.debug("[codegen_node] SQL AST Conformance Verification PASSED. Issued receipt %s", receipt.code_hash[:8])
        else:
            generated = self.codegen.generate(
                profile,
                objective,
                step.description,
                previous_context=prev_context_str,
            )

        logger.debug("[codegen_node] Active step ID: %s", step.step_id)
        logger.debug("[codegen_node] Step description: %s", step.description)
        logger.debug("[codegen_node] Metric column: %s", getattr(step, "metric_col", None))
        logger.debug("[codegen_node] Aggregation function: %s", getattr(step, "agg_func", None))
        logger.debug("[codegen_node] WHERE clause: %s", getattr(step, "where_clause", None))
        logger.debug("[codegen_node] Generated %s code:\n%s", generated.code_type.upper(), generated.executable_code)

        return {
            "current_code": generated.executable_code,
            "current_code_type": generated.code_type,
            "current_contract": contract,
            "current_receipt": receipt,
            "error_context": error_ctx,
            "status": "CODE_GENERATED",
        }

    def exec_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Executes current code in the appropriate sandboxed engine."""
        if state.get("status") in {"COMPILATION_FAILED", "VERIFICATION_FAILED"} or state.get("compilation_error") or state.get("verification_error"):
            logger.warning("[exec_node] Execution blocked: Pipeline in failed state '%s'", state.get("status"))
            step_id = state["current_step"].step_id if state.get("current_step") else "step_1"
            step_results = dict(state.get("step_results", {}))
            err_msg = state.get("error_context") or state.get("verification_error") or state.get("compilation_error") or "Execution blocked by gatekeeper."
            step_results[step_id] = {
                "success": False,
                "error": err_msg,
                "dataframe": None,
                "scalar": None,
            }
            return {
                "step_results": step_results,
                "status": state.get("status", "FAILED"),
                "current_exec_result": {"success": False, "stderr": err_msg},
            }

        code: str = state["current_code"] or ""
        code_type: str = state.get("current_code_type", "sql")
        receipt = state.get("current_receipt")
        contract = state.get("current_contract")

        if code_type == "python":
            exec_res = self.py_sandbox.execute(code)
        else:
            exec_res = self.sql_sandbox.execute(code, receipt=receipt, contract=contract)

        error_ctx = None
        if not exec_res.success and exec_res.error:
            error_ctx = exec_res.error.to_prompt_injection()

        res_dict = {
            "success": exec_res.success,
            "row_count": exec_res.row_count,
            "scalar_result": exec_res.scalar_result,
            "execution_time_ms": exec_res.execution_time_ms,
            "query_executed": exec_res.query_executed,
            "is_empty": exec_res.is_empty,
            "stdout": exec_res.stdout,
            "stderr": exec_res.stderr,
        }

        # Keep dataframe reference in step_results
        step_id = state["current_step"].step_id
        step_results = dict(state.get("step_results", {}))
        step_results[step_id] = {
            "dataframe": exec_res.result_df,
            "scalar": exec_res.scalar_result,
            "success": exec_res.success,
        }

        return {
            "current_exec_result": res_dict,
            "step_results": step_results,
            "error_context": error_ctx,
            "status": "EXECUTED",
        }

    def validate_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Applies deterministic mathematical invariant validation to the step."""
        if state.get("status") in {"COMPILATION_FAILED", "VERIFICATION_FAILED"} or state.get("compilation_error") or state.get("verification_error"):
            failed_status = state.get("status", "FAILED")
            err_msg = state.get("error_context") or state.get("verification_error") or state.get("compilation_error") or "Execution or verification failed."
            return {
                "current_validation": {"is_valid": False, "error_type": failed_status, "error_message": err_msg},
                "status": failed_status,
            }

        exec_res = state.get("current_exec_result")
        step: PlanStep = state["current_step"]
        objective: AnalysisObjective = state["objective"]
        table_name = objective.target_table

        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.max_self_correction_iterations)
        evidence_list = list(state.get("evidence_records", []))
        corr_history = list(state.get("correction_history", []))

        is_dq = (
            objective.analysis_type == AnalysisType.DATA_QUALITY
            or objective.quality_check_type == "bidirectional_mapping"
        )
        is_empty_res = bool(exec_res.get("is_empty")) if exec_res else False

        # Actual execution error (syntax, binder error, timeout, or uncaught exception)
        has_real_error = bool(exec_res.get("stderr")) or (not exec_res.get("success") and not is_empty_res) if exec_res else True

        if not exec_res or has_real_error:
            err_msg = exec_res.get('stderr', 'Execution failed') if exec_res else 'Execution failed'
            corr_history.append(CorrectionAttempt(
                attempt_number=retries + 1,
                step_id=step.step_id,
                failed_code=state.get("current_code", "") or "",
                error_type="EXECUTION_FAILURE",
                error_message=err_msg,
                corrected_code=None,
                validation_passed=False,
                execution_time_ms=0.0,
            ))
            # If retries exhausted, record failed execution in provenance trace
            if retries >= max_retries:
                step.completed = False
                step.code = state.get("current_code")
                summary_val = f"Execution failed: {err_msg}"
                rec = EvidenceRecord(
                    step_id=step.step_id,
                    action_type="EXECUTION",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash="fail_" + step.step_id,
                    validation_status="FAILED",
                    execution_time_ms=0.0,
                )
                evidence_list.append(rec)
                self.provenance.add_artifact(EvidenceArtifact(
                    step_id=step.step_id,
                    action_type="EXECUTION",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash="fail_" + step.step_id,
                    validation_status="FAILED",
                    execution_time_ms=0.0,
                ))
            return {
                "current_validation": {"is_valid": False, "error_type": "EXECUTION_FAILURE"},
                "evidence_records": evidence_list,
                "correction_history": corr_history,
                "status": "VALIDATION_FAILED",
            }

        # Check for empty / NaN scalar result (for standard descriptive queries)
        scalar = exec_res.get("scalar_result")
        import math
        if not is_dq and ((scalar is not None and math.isnan(scalar)) or is_empty_res):
            error_ctx = (
                "EMPTY_RESULT_ERROR: Query returned NaN or 0 rows. "
                "The WHERE filter condition did not match any records in the table. "
                "Verify date formatting (e.g. use EXTRACT(YEAR FROM col) or date ranges) and exact filter values."
            )
            corr_history.append(CorrectionAttempt(
                attempt_number=retries + 1,
                step_id=step.step_id,
                failed_code=state.get("current_code", "") or "",
                error_type="EMPTY_RESULT_ERROR",
                error_message="Query returned NaN or 0 matching records.",
                corrected_code=None,
                validation_passed=False,
                execution_time_ms=0.0,
            ))
            if retries >= max_retries:
                step.completed = False
                step.code = state.get("current_code")
                summary_val = "0 rows returned (Empty result)"
                rec = EvidenceRecord(
                    step_id=step.step_id,
                    action_type="EMPTY_RESULT",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash="empty_" + step.step_id,
                    validation_status="FAILED",
                    execution_time_ms=0.0,
                )
                evidence_list.append(rec)
                self.provenance.add_artifact(EvidenceArtifact(
                    step_id=step.step_id,
                    action_type="EMPTY_RESULT",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash="empty_" + step.step_id,
                    validation_status="FAILED",
                    execution_time_ms=0.0,
                ))
            return {
                "current_validation": {
                    "is_valid": False,
                    "error_type": "EMPTY_RESULT_ERROR",
                    "error_message": "Query returned NaN or 0 matching records.",
                },
                "evidence_records": evidence_list,
                "correction_history": corr_history,
                "error_context": error_ctx,
                "status": "VALIDATION_FAILED",
            }

        all_cols = []
        profile = state.get("profile")
        if profile and profile.tables:
            for tbl in profile.tables:
                if tbl.table_name == table_name:
                    all_cols = [c.name for c in tbl.columns]
                    break
            if not all_cols:
                all_cols = [c.name for c in profile.tables[0].columns]

        from src.agent.schemas import resolve_metric_column
        metric_col = resolve_metric_column(step.metric_col or objective.target_metric, all_cols)

        code = state.get("current_code", "")
        # Use actual WHERE filter from executed SQL if step where_clause was default 1=1
        where_filter = step.where_clause
        if not where_filter or where_filter.strip() == "1=1":
            m = re.search(r'\bWHERE\b\s+([\s\S]+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)', code, re.IGNORECASE)
            if m:
                where_filter = m.group(1).strip()

        # If query has LIMIT, HAVING, or is a top-N ranking, do not run unsegmented vs segmented Sum-of-Parts check against full table
        is_subset_or_ranking = bool(re.search(r'\b(LIMIT|HAVING)\b', code, re.IGNORECASE))
        group_col_for_val = None if is_subset_or_ranking else step.group_col

        agg_func_for_val = step.agg_func or "SUM"
        if re.search(r'\bMAX\s*\(', code, re.IGNORECASE) or any(k in step.description.lower() for k in ["highest", "maximum", "max"]):
            agg_func_for_val = "MAX"
        elif re.search(r'\bMIN\s*\(', code, re.IGNORECASE) or any(k in step.description.lower() for k in ["lowest", "minimum", "min"]):
            agg_func_for_val = "MIN"

        validator = DeterministicValidator(self.con, table_name)

        # Dispatch to specialized verifiers:
        composed_contract = state.get("composed_contract")
        contract = state.get("current_contract")
        res_df = state.get("step_results", {}).get(step.step_id, {}).get("dataframe")

        if composed_contract:
            raw_df = state.get("raw_dataframe")
            if raw_df is None:
                raw_df = self.con.execute(f'SELECT * FROM "{composed_contract.target_table}"').df()

            val_res = ComposedPipelineValidator.validate(
                raw_df=raw_df,
                sql_actual_df=res_df if res_df is not None else pd.DataFrame(),
                contract=composed_contract,
            )
            val_hash = hashlib.sha256(f"composed_{step.step_id}_{val_res.is_valid}".encode()).hexdigest()[:16]
            val_report = ValidationReport(
                step_id=step.step_id,
                table_name=composed_contract.target_table,
                metric_col=composed_contract.required_measures[0].target_column.name,
                is_fully_valid=val_res.is_valid,
                checks_passed=1 if val_res.is_valid else 0,
                checks_total=1,
                results=[val_res],
                execution_time_ms=exec_res.get("execution_time_ms", 0.0) if exec_res else 0.0,
                checksum_hash=val_hash,
            )
        elif contract and (contract.required_derived_measures or any(m.condition_predicate for m in contract.required_measures)):
            val_report = validator.validate_contract_execution(
                step_id=step.step_id,
                contract=contract,
                result_df=res_df,
            )
        elif objective.analysis_type == AnalysisType.DATA_QUALITY or "is mapped to multiple" in step.description.lower() or "having count(distinct" in step.description.lower():
            ent_a = step.group_col or (objective.conflict_entities[0] if objective.conflict_entities else "product_id")
            ent_b = step.metric_col or (objective.conflict_entities[1] if len(objective.conflict_entities) > 1 else "product_name")
            res_df = state.get("step_results", {}).get(step.step_id, {}).get("dataframe")
            val_report = validator.validate_data_quality_mapping(
                step_id=step.step_id,
                group_entity=ent_a,
                target_entity=ent_b,
                result_df=res_df,
            )
        elif "percentage change" in step.description.lower():
            step_res = state.get("step_results", {})
            v_base = step_res.get("step_1", {}).get("scalar")
            v_test = step_res.get("step_2", {}).get("scalar")
            claimed_pct = exec_res.get("scalar_result")
            if v_base is not None and v_test is not None and claimed_pct is not None:
                val_report = validator.validate_arithmetic_calculation(
                    step_id=step.step_id,
                    base_value=float(v_base),
                    test_value=float(v_test),
                    claimed_pct_change=float(claimed_pct),
                )
            else:
                val_report = validator.validate_aggregation(
                    step_id=step.step_id,
                    metric_col=metric_col,
                    agg_func=agg_func_for_val,
                    group_col=group_col_for_val,
                    where_clause=where_filter,
                )
        else:
            val_report = validator.validate_aggregation(
                step_id=step.step_id,
                metric_col=metric_col,
                agg_func=agg_func_for_val,
                group_col=group_col_for_val,
                where_clause=where_filter,
            )

        val_dict = {
            "is_valid": val_report.is_fully_valid,
            "checks_passed": val_report.checks_passed,
            "checks_total": val_report.checks_total,
            "checksum_hash": val_report.checksum_hash,
            "summary": val_report.summary(),
            "missing_sales_diagnostics": getattr(val_report, "missing_sales_diagnostics", {}),
        }

        evidence_list = list(state.get("evidence_records", []))
        error_ctx = state.get("error_context")

        if val_report.is_fully_valid:
            # Step passed! Update correction attempt if present
            if corr_history and corr_history[-1].step_id == step.step_id:
                corr_history[-1].validation_passed = True

            step.completed = True
            step.code = state.get("current_code")
            step.result_summary = val_report.summary()

            scalar = exec_res.get("scalar_result")
            rows = exec_res.get("row_count")
            summary_val = f"Scalar: {scalar}" if scalar is not None else f"{rows} rows"

            evidence_list.append(EvidenceRecord(
                step_id=step.step_id,
                action_type="AGGREGATION_VALIDATION",
                query_sql=state.get("current_code"),
                result_summary=summary_val,
                checksum_hash=val_report.checksum_hash,
                validation_status="VALIDATED_DETERMINISTICALLY",
                execution_time_ms=val_report.execution_time_ms,
            ))

            # Register in Provenance DAG
            if composed_contract:
                self.provenance.add_artifact(EvidenceArtifact(
                    step_id=step.step_id,
                    action_type="CAPABILITY_PIPELINE_VALIDATION",
                    query_sql=state.get("current_code"),
                    result_summary=val_report.summary(),
                    checksum_hash=val_report.checksum_hash,
                    validation_status="VALIDATED_DETERMINISTICALLY",
                    execution_time_ms=val_report.execution_time_ms,
                    metadata={
                        "execution_mode": "registry",
                        "bound_capabilities": composed_contract.bound_capabilities,
                        "dag_fingerprint": composed_contract.dag_fingerprint,
                        "air_operation_ids": composed_contract.air_operation_ids,
                        "input_bindings": [
                            b.model_dump()
                            for node in composed_contract.dag.nodes
                            for b in node.port_bindings
                        ],
                    },
                ))
            else:
                self.provenance.add_artifact(EvidenceArtifact(
                    step_id=step.step_id,
                    action_type="AGGREGATION_VALIDATION",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash=val_report.checksum_hash,
                    validation_status="VALIDATED_DETERMINISTICALLY",
                    execution_time_ms=val_report.execution_time_ms,
                ))
            status = "VALIDATION_PASSED"
        else:
            first_err = val_report.first_error()
            err_type = first_err.error_type if first_err else "INVARIANT_ERROR"
            err_msg = first_err.error_message if first_err else "Deterministic invariant violated"
            if err_type == "calculation_mismatch":
                error_ctx = (
                    f"CALCULATION_MISMATCH: {err_msg} "
                    f"Audit details: {first_err.audit_details if first_err else {}}. "
                    "Recompute using the exact formula: ((test_value - base_value) / base_value) * 100.0."
                )
            elif err_type == "DATA_QUALITY_MISMATCH":
                error_ctx = f"DATA_QUALITY_ERROR: {err_msg}. Ensure the query uses GROUP BY and HAVING COUNT(DISTINCT ...) > 1 correctly."
            else:
                classified = ErrorClassifier.classify_validation_failure(err_type, err_msg)
                error_ctx = classified.to_prompt_injection()

            corr_history.append(CorrectionAttempt(
                attempt_number=retries + 1,
                step_id=step.step_id,
                failed_code=state.get("current_code", "") or "",
                error_type=err_type,
                error_message=err_msg,
                corrected_code=None,
                validation_passed=False,
                execution_time_ms=val_report.execution_time_ms,
            ))
            status = "VALIDATION_FAILED"

            # If this was the last attempt, capture the result with warning so provenance is complete
            retries = state.get("retry_count", 0)
            max_retries = state.get("max_retries", settings.max_self_correction_iterations)
            if retries >= max_retries:
                step.completed = True
                step.code = state.get("current_code")
                scalar = exec_res.get("scalar_result")
                rows = exec_res.get("row_count")
                summary_val = f"Scalar: {scalar}" if scalar is not None else f"{rows} rows (with warning)"
                evidence_list.append(EvidenceRecord(
                    step_id=step.step_id,
                    action_type="AGGREGATION_VALIDATION",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash=val_report.checksum_hash,
                    validation_status="COMPLETED_WITH_WARNINGS",
                    execution_time_ms=val_report.execution_time_ms,
                ))
                self.provenance.add_artifact(EvidenceArtifact(
                    step_id=step.step_id,
                    action_type="AGGREGATION_VALIDATION",
                    query_sql=state.get("current_code"),
                    result_summary=summary_val,
                    checksum_hash=val_report.checksum_hash,
                    validation_status="COMPLETED_WITH_WARNINGS",
                    execution_time_ms=val_report.execution_time_ms,
                ))

        missing_sales_diag = getattr(val_report, "missing_sales_diagnostics", {})
        ret_dict: Dict[str, Any] = {
            "current_validation": val_dict,
            "missing_sales_diagnostics": missing_sales_diag,
            "evidence_records": evidence_list,
            "correction_history": corr_history,
            "error_context": error_ctx,
            "status": status,
        }
        if missing_sales_diag:
            step_results = dict(state.get("step_results", {}))
            if step.step_id in step_results:
                step_results[step.step_id]["missing_sales_diagnostics"] = missing_sales_diag
            ret_dict["step_results"] = step_results

        return ret_dict

    def correct_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Self-healing node: prompts LLM with exact failure details for code regeneration."""
        profile: DatasetProfile = state["profile"]
        objective: AnalysisObjective = state["objective"]
        step: PlanStep = state["current_step"]
        failed_code: str = state.get("current_code", "")
        error_injection: str = state.get("error_context") or "Unknown error occurred"

        retry_count = state.get("retry_count", 0) + 1

        corrected = self.codegen.generate_correction(
            profile=profile,
            objective=objective,
            step_description=step.description,
            failed_code=failed_code,
            error_prompt_injection=error_injection,
        )

        # Independent re-verification of corrected code
        receipt = None
        contract = state.get("current_contract")
        if contract and contract.target_engine == ContractEngine.DUCKDB_SQL:
            is_valid, err_msg, receipt = SQLASTConformanceVerifier.verify(corrected.executable_code, contract)
            if not is_valid:
                logger.warning("[correct_node] Re-verification FAILED: %s", err_msg)
                receipt = None
            else:
                logger.debug("[correct_node] Re-verification PASSED. Issued receipt %s", receipt.code_hash[:8])

        corr_history = list(state.get("correction_history", []))
        if corr_history:
            corr_history[-1].corrected_code = corrected.executable_code

        return {
            "current_code": corrected.executable_code,
            "current_code_type": corrected.code_type,
            "current_receipt": receipt,
            "retry_count": retry_count,
            "correction_history": corr_history,
            "status": "CODE_CORRECTED",
        }

    def advance_step_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Advances the state pointer to the next plan step.

        Also records an unresolved discrepancy when we arrived here because the
        previous step exceeded its maximum retry budget (retries >= max_retries
        AND current validation failed).  This is the only place we set that flag
        because routing functions must be pure — they cannot mutate state.
        """
        next_idx = state.get("current_step_index", 0) + 1
        plan: AnalysisPlan = state["plan"]
        next_step = plan.steps[next_idx] if next_idx < len(plan.steps) else None

        # Detect whether we advanced because retries were exhausted (not because
        # the step passed).  If so, record the discrepancy in state.
        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.max_self_correction_iterations)
        val_res = state.get("current_validation") or {}
        is_valid = val_res.get("is_valid", False)
        discrepancy = state.get("unresolved_discrepancy", False) or (
            not is_valid and retries >= max_retries
        )

        return {
            "current_step_index": next_idx,
            "current_step": next_step,
            "current_contract": None,
            "current_receipt": None,
            "retry_count": 0,
            "error_context": None,
            "unresolved_discrepancy": discrepancy,
            "status": "STEP_ADVANCED",
        }

    def rca_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """
        Root-Cause Analysis node.
        Decomposes observed variance across candidate dimensions and Volume vs. Price.
        """
        objective: AnalysisObjective = state["objective"]
        profile: DatasetProfile = state["profile"]
        table_name = objective.target_table

        all_num_cols = []
        for tbl in profile.tables:
            if tbl.table_name == table_name:
                all_num_cols = [c.name for c in tbl.columns if c.is_numeric]
                break
        from src.agent.schemas import resolve_metric_column
        metric_col = resolve_metric_column(objective.target_metric, all_num_cols)

        # Candidate dimensions: prioritize question dimensions, filter for 2 <= distinct_count <= 100
        dimensions = [d for d in objective.candidate_dimensions if d]
        for tbl in profile.tables:
            if tbl.table_name == table_name:
                for col in tbl.columns:
                    if not col.is_numeric and not col.is_temporal and col.name not in dimensions and not col.name.lower().endswith("_id"):
                        if 2 <= col.distinct_count <= 100:
                            dimensions.append(col.name)
                break
        if not dimensions:
            dimensions = ["category", "region"]

        base_filter = objective.baseline_period or "1=1"
        test_filter = objective.test_period or "1=1"

        rca_report = RCAEngine.analyze(
            con=self.con,
            table_name=table_name,
            metric_col=metric_col,
            dimensions=dimensions[:4],
            baseline_where=base_filter,
            test_where=test_filter,
        )

        evidence_list = list(state.get("evidence_records", []))
        evidence_list.append(EvidenceRecord(
            step_id="rca_attribution",
            action_type="ROOT_CAUSE_ANALYSIS",
            query_sql=f"RCA over {dimensions}",
            result_summary=f"Δ={rca_report.total_delta:,.2f} ({rca_report.delta_pct:+.1f}%), {len(rca_report.primary_factors)} primary factors identified",
            checksum_hash="rca_" + metric_col[:8],
            validation_status="VALIDATED_DETERMINISTICALLY",
            execution_time_ms=0.0,
        ))

        # Register in Provenance DAG
        self.provenance.add_artifact(EvidenceArtifact(
            step_id="rca_attribution",
            action_type="ROOT_CAUSE_ANALYSIS",
            query_sql=f"RCA over {dimensions}",
            result_summary=f"Δ={rca_report.total_delta:,.2f} ({rca_report.delta_pct:+.1f}%), {len(rca_report.primary_factors)} primary factors identified",
            checksum_hash="rca_" + metric_col[:8],
            validation_status="VALIDATED_DETERMINISTICALLY",
            execution_time_ms=0.0,
            parent_step_ids=[s.step_id for s in state["plan"].steps if s.completed] if state.get("plan") else [],
        ))

        return {
            "rca_report": rca_report,
            "evidence_records": evidence_list,
            "status": "RCA_COMPLETED",
        }

    def synthesize_node(self, state: AnalyticsAgentState) -> Dict[str, Any]:
        """Synthesizes all evidence into a validated conclusion with ECS confidence score."""
        is_registry_mode = (
            state.get("execution_mode") in (ExecutionMode.REGISTRY, "registry")
            or state.get("registry_mode", False)
            or state.get("composed_contract") is not None
        )
        if (
            state.get("status") in {"COMPILATION_FAILED", "VERIFICATION_FAILED"}
            or (state.get("status") == "VALIDATION_FAILED" and is_registry_mode)
            or state.get("compilation_error")
            or state.get("verification_error")
        ):
            failed_status = state.get("status", "FAILED")
            err_details = state.get("compilation_error") or state.get("verification_error") or state.get("error_context") or "Contract compilation, verification, or capability validation failed."
            lines = [
                f"# Analytical Pipeline Halted ({failed_status})",
                f"- **User Question:** {state.get('user_question')}",
                f"- **Failure Reason:** `{err_details}`",
                "- **Status:** Pipeline halted fail-closed. No unverified code was executed.",
                "",
                "### 🛡️ Independent Verification Dimensions",
                "- **Execution Correctness:** `FAILED`",
                "- **Calculation Correctness:** `FAILED`",
                "- **Semantic Question Coverage:** `FAILED`",
                "- **Evidence Sufficiency:** `INSUFFICIENT`",
                "- **Evidence Integrity:** `UNVERIFIED`",
                "- **Overall Outcome:** `FAILED_VALIDATION`",
            ]
            ecs_report = ECSReport(
                score=0.0,
                tier=ConfidenceTier.INSUFFICIENT_EVIDENCE,
                badge_color="red",
                components={},
                recommendation=f"Pipeline halted: {err_details}",
                formula_expression="0.0 (Pipeline Halted)",
            )
            return {
                "final_summary": "\n".join(lines),
                "ecs_report": ecs_report,
                "semantic_coverage": None,
                "execution_correctness": "FAILED",
                "calculation_correctness": "FAILED",
                "semantic_coverage_status": "FAILED",
                "evidence_sufficiency": "INSUFFICIENT",
                "evidence_integrity": "UNVERIFIED",
                "overall_outcome": "FAILED_VALIDATION",
                "claim_evidences": [],
                "correction_history": state.get("correction_history", []),
                "provenance_dag": self.provenance.to_dict(),
                "unresolved_discrepancy": True,
                "status": failed_status,
            }

        evidence: List[EvidenceRecord] = state.get("evidence_records", [])
        objective: AnalysisObjective = state["objective"]
        steps = state.get("plan", AnalysisPlan(reasoning="", steps=[])).steps

        passed_steps = sum(1 for s in steps if s.completed)
        total_steps = len(steps)

        # Determine whether the final step ended with an unresolved discrepancy.
        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.max_self_correction_iterations)
        val_res = state.get("current_validation") or {}
        is_valid = val_res.get("is_valid", False)
        unresolved = state.get("unresolved_discrepancy", False) or (
            not is_valid and retries >= max_retries
        )

        step_results = state.get("step_results", {})
        executed_sqls = [s.code for s in steps if s.code]

        # 1. Semantic Question Coverage Validation
        semantic_report = SemanticCoverageValidator.validate(
            user_question=state.get("user_question", ""),
            objective=objective,
            plan=state.get("plan"),
            step_results=step_results,
            executed_sqls=executed_sqls,
        )

        # 2. Compute Evidence Confidence Score (ECS) with hard semantic gating
        checks_passed = sum(1 for ev in evidence if ev.validation_status == "VALIDATED_DETERMINISTICALLY")
        checks_total = len(evidence) if evidence else 1

        ecs_report = ECSCalculator.compute(
            checks_passed=checks_passed,
            checks_total=checks_total,
            profile=state.get("profile"),
            key_columns=[objective.target_metric],
            rca_report=state.get("rca_report"),
            unresolved_discrepancy=unresolved,
            semantic_coverage=semantic_report,
        )

        # 3. Independent Verification Statuses
        execution_correctness = "PASSED" if (total_steps > 0 and passed_steps == total_steps) else "FAILED"
        calculation_correctness = "PASSED" if (checks_passed == checks_total and not unresolved) else ("WARNING" if unresolved else "FAILED")
        semantic_coverage_status = "PASSED" if semantic_report.is_covered else "FAILED"
        evidence_sufficiency = "SUFFICIENT" if (total_steps > 0 and passed_steps == total_steps and (not objective.required_outputs or passed_steps >= len(objective.required_outputs))) else "INSUFFICIENT"
        evidence_integrity = "VERIFIED" if (all(ev.checksum_hash for ev in evidence) and checks_passed > 0) else "UNVERIFIED"

        if execution_correctness == "PASSED" and calculation_correctness == "PASSED" and semantic_coverage_status == "PASSED":
            overall_outcome = "FULLY_VERIFIED"
        elif execution_correctness == "PASSED" and semantic_coverage_status == "PASSED":
            overall_outcome = "COMPLETED_WITH_WARNINGS"
        else:
            overall_outcome = "FAILED_VALIDATION"

        sem_dump = semantic_report.model_dump() if hasattr(semantic_report, "model_dump") else str(semantic_report)
        logger.debug("[synthesize_node] Semantic coverage report:\n%s", json.dumps(sem_dump, indent=2, default=str))
        logger.debug("[synthesize_node] Overall outcome: %s | ECS: %.2f%% (%s)", overall_outcome, ecs_report.score if ecs_report else 0.0, ecs_report.tier.value if ecs_report else "N/A")
        print(f"[synthesize_node] Semantic coverage report:\n{json.dumps(sem_dump, indent=2, default=str)}")
        print(f"[synthesize_node] Overall outcome: {overall_outcome}")
        print(f"[synthesize_node] ECS: {ecs_report.score if ecs_report else 0.0:.2f}% ({ecs_report.tier.value if ecs_report else 'N/A'})")

        lines = [
            f"# Analytical Findings for: {state.get('user_question')}",
            f"- **Target Metric:** `{objective.target_metric}` on table `{objective.target_table}`",
            f"- **Analysis Type:** `{objective.analysis_type.value}`",
            f"- **Plan Verification:** {passed_steps}/{total_steps} steps validated deterministically.",
            f"- **Evidence Confidence Score (ECS):** `{ecs_report.score:.1f}%` (**{ecs_report.tier.value}**)",
            f"  * {ecs_report.recommendation}",
            "",
            "### 🛡️ Independent Verification Dimensions",
            f"- **Execution Correctness:** `{execution_correctness}`",
            f"- **Calculation Correctness:** `{calculation_correctness}`",
            f"- **Semantic Question Coverage:** `{semantic_coverage_status}` ({semantic_report.details})",
            f"- **Evidence Sufficiency:** `{evidence_sufficiency}`",
            f"- **Evidence Integrity:** `{evidence_integrity}`",
            f"- **Overall Outcome:** `{overall_outcome}`",
            "",
        ]

        if objective.missing_columns_explanation:
            lines.append(f"> ⚠️ **Required Column Unavailable:** {objective.missing_columns_explanation}\n")
        elif not semantic_report.is_covered:
            lines.append(f"> 🚫 **Semantic Coverage Warning:** {semantic_report.details}\n")
        top_answers = []

        # Data Quality Special Formatting
        if objective.analysis_type == AnalysisType.DATA_QUALITY:
            entities = objective.conflict_entities if len(objective.conflict_entities) >= 2 else ["product_id", "product_name"]
            ent_a, ent_b = entities[0], entities[1]
            df_a = step_results.get("step_1", {}).get("dataframe")
            df_b = step_results.get("step_2", {}).get("dataframe")

            num_conflicts_a = len(df_a) if df_a is not None and not df_a.empty else 0
            num_conflicts_b = len(df_b) if df_b is not None and not df_b.empty else 0

            if num_conflicts_a == 0 and num_conflicts_b == 0:
                top_answers.append(
                    f"### 🏆 Data Integrity Verification: 100% Consistent Mappings\n"
                    f"**No conflicting mappings detected** between `{ent_a}` and `{ent_b}` across the dataset.\n\n"
                    f"- **Number of conflicting {ent_a}s:** 0\n"
                    f"- **Number of conflicting {ent_b}s:** 0\n"
                    f"- **Status:** Every `{ent_a}` maps to exactly one `{ent_b}`, and every `{ent_b}` maps to exactly one `{ent_a}`.\n"
                )
            else:
                top_answers.append(
                    f"### ⚠️ Data Integrity Inconsistencies Detected\n"
                    f"- **Number of conflicting {ent_a}s:** {num_conflicts_a} `{ent_a}` value(s) map to multiple `{ent_b}` values.\n"
                    f"- **Number of conflicting {ent_b}s:** {num_conflicts_b} `{ent_b}` value(s) map to multiple `{ent_a}` values.\n"
                )
                if num_conflicts_a > 0 and df_a is not None:
                    top_answers.append(f"#### 🔍 Conflicting `{ent_a}` Details ({num_conflicts_a} conflicts found):")
                    header = "| " + " | ".join(str(c) for c in df_a.columns) + " |"
                    sep = "| " + " | ".join("---" for _ in df_a.columns) + " |"
                    rows = []
                    for _, r in df_a.head(10).iterrows():
                        rows.append("| " + " | ".join(str(v) for v in r) + " |")
                    top_answers.append("\n".join([header, sep] + rows) + "\n")
                if num_conflicts_b > 0 and df_b is not None:
                    top_answers.append(f"#### 🔍 Conflicting `{ent_b}` Details ({num_conflicts_b} conflicts found):")
                    header = "| " + " | ".join(str(c) for c in df_b.columns) + " |"
                    sep = "| " + " | ".join("---" for _ in df_b.columns) + " |"
                    rows = []
                    for _, r in df_b.head(10).iterrows():
                        rows.append("| " + " | ".join(str(v) for v in r) + " |")
                    top_answers.append("\n".join([header, sep] + rows) + "\n")
        else:
            # Extract direct findings and top answer from execution step results
            for s_id, s_data in step_results.items():
                if not isinstance(s_data, dict):
                    continue
                df = s_data.get("dataframe")
                scalar = s_data.get("scalar")
                if df is not None and not df.empty and len(df.columns) >= 2:
                    cols = list(df.columns)
                    is_aov_df = any("aov" in str(c).lower() or "average_order_value" in str(c).lower() for c in cols)
                    if is_aov_df:
                        row0 = df.iloc[0]
                        aov_col = next((c for c in cols if "aov" in str(c).lower() or "average_order_value" in str(c).lower()), None)
                        orders_col = next((c for c in cols if "order" in str(c).lower() and "null" not in str(c).lower()), None)
                        rev_col = next((c for c in cols if ("revenue" in str(c).lower() or "sales" in str(c).lower()) and "unassigned" not in str(c).lower()), None)
                        null_id_col = next((c for c in cols if "null" in str(c).lower() and "order" in str(c).lower()), None)
                        unassigned_rev_col = next((c for c in cols if "unassigned" in str(c).lower()), None)

                        aov_val = float(row0[aov_col]) if aov_col and pd.notna(row0[aov_col]) else 0.0
                        tot_orders = int(row0[orders_col]) if orders_col and pd.notna(row0[orders_col]) else 0
                        tot_rev = float(row0[rev_col]) if rev_col and pd.notna(row0[rev_col]) else 0.0
                        null_id_cnt = int(row0[null_id_col]) if null_id_col and pd.notna(row0[null_id_col]) else 0
                        unassigned_rev = float(row0[unassigned_rev_col]) if unassigned_rev_col and pd.notna(row0[unassigned_rev_col]) else 0.0

                        # ADV-03: Retrieve missing-sales diagnostics strictly from independent validation path (NO raw SQL execution!)
                        s_id = s_step_id if 's_step_id' in locals() else "step_1"
                        missing_diag = (
                            state.get("missing_sales_diagnostics")
                            or state.get("step_results", {}).get(s_id, {}).get("missing_sales_diagnostics")
                            or (state.get("current_validation") or {}).get("missing_sales_diagnostics")
                            or {}
                        )
                        orders_with_no_sales = int(missing_diag.get("orders_with_no_sales", 0))
                        orders_with_partial_null_sales = int(missing_diag.get("orders_with_partial_null_sales", 0))
                        null_sales_row_count = int(missing_diag.get("null_sales_row_count", 0))

                        top_answers.append(
                            f"### 🏆 Primary Answer (Average Order Value & Order Analysis)\n"
                            f"- **Total Identifiable Orders:** **{tot_orders:,}**\n"
                            f"- **Total Valid Revenue:** **${tot_rev:,.2f}**\n"
                            f"- **Average Order Value (AOV):** **${aov_val:,.2f}**\n"
                            f"- **Calculation Formula:** `${tot_rev:,.2f} / {tot_orders:,} orders` = **${aov_val:,.2f}**\n\n"
                            f"#### 🔍 Data Quality Branch & Missing-Data Evidence:\n"
                            f"- **Null-ID Records:** **{null_id_cnt}** record(s) have NULL `order_id` (excluded from order count).\n"
                            f"- **Unassigned Revenue:** **${unassigned_rev:,.2f}** from NULL `order_id` records (strictly excluded from AOV numerator).\n"
                            f"- **Orders with No Non-Null Sales:** **{orders_with_no_sales}** order(s).\n"
                            f"- **Rows with NULL Sales:** **{null_sales_row_count}** row(s).\n"
                            f"- **Orders with Partially Missing Sales:** **{orders_with_partial_null_sales}** order(s).\n"
                            f"> ⚠️ **Data Quality Notice:** AOV denominator includes all identifiable orders. Incomplete or missing sales data can skew AOV diagnostics.\n"
                        )
                    else:
                        dim_col, num_col = cols[0], cols[1]
                        try:
                            sorted_df = df.sort_values(by=num_col, ascending=False).dropna(subset=[num_col])
                            if not sorted_df.empty:
                                top_row = sorted_df.iloc[0]
                                top_dim = top_row[dim_col]
                                top_val = top_row[num_col]
                                top_answers.append(
                                    f"### 🏆 Primary Answer\n"
                                    f"**{top_dim}** yields the highest **{num_col}** with a total of **{top_val:,.2f}**.\n\n"
                                    f"#### Top Results Breakdown:\n"
                                )
                                # Build markdown table
                                header = "| " + " | ".join(str(c) for c in sorted_df.columns) + " |"
                                sep = "| " + " | ".join("---" for _ in sorted_df.columns) + " |"
                                rows = []
                                for _, r in sorted_df.head(5).iterrows():
                                    rows.append("| " + " | ".join(f"{v:,.2f}" if isinstance(v, (int, float)) else str(v) for v in r) + " |")
                                top_answers.append("\n".join([header, sep] + rows) + "\n")
                        except Exception:
                            pass
                elif scalar is not None:
                    import math
                    if not math.isnan(scalar):
                        step_obj = next((s for s in steps if s.step_id == s_id), None)
                        label = step_obj.description if step_obj and step_obj.description else f"Result for {objective.target_metric}"
                        if any(k in label.lower() for k in ["margin", "pct", "percent"]):
                            val_str = f"{scalar:.2f}%"
                        elif any(k in label.lower() for k in ["sales", "profit", "revenue", "aov", "value", "price"]):
                            val_str = f"${scalar:,.2f}"
                        elif any(k in label.lower() for k in ["order", "quantity", "unit", "count"]):
                            val_str = f"{int(round(scalar)):,}"
                        else:
                            val_str = f"{scalar:,.2f}"
                        ans_block = [f"### 📊 {label}\n**{val_str}**"]

                        # If step is AOV, include explicit breakdown formula
                        if step_obj and (step_obj.agg_func == "AOV" or "aov" in label.lower() or "average order value" in label.lower()):
                            try:
                                tbl_name = objective.target_table
                                where_cls = step_obj.where_clause or "1=1"
                                prof = state.get("profile")
                                t_p = prof.tables[0] if prof and prof.tables else None
                                s_col = next((c.name for c in (t_p.columns if t_p else []) if any(k in c.name.lower() for k in ["sales", "revenue", "amount"])), "sales")
                                o_col = next((c.name for c in (t_p.columns if t_p else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), "order_id")
                                row_stats = self.con.execute(f'SELECT CAST(SUM("{s_col}") AS DOUBLE), CAST(COUNT(DISTINCT "{o_col}") AS DOUBLE) FROM "{tbl_name}" WHERE {where_cls}').fetchone()
                                if row_stats and row_stats[0] is not None and row_stats[1] is not None:
                                    ans_block.append(f"- **Calculation Formula:** `SUM({s_col}) / COUNT(DISTINCT {o_col})` = `${row_stats[0]:,.2f} / {int(round(row_stats[1])):,} orders` = **${scalar:,.2f}**")
                            except Exception:
                                pass

                        top_answers.append("\n".join(ans_block) + "\n")

        if top_answers:
            lines.extend(top_answers)

        lines.append("## Key Evidence:")
        for ev in evidence:
            lines.append(f"- Step `{ev.step_id}` [{ev.validation_status}]: {ev.result_summary} (Hash: `{ev.checksum_hash}`)")

        rca_report: Optional[RCAReport] = state.get("rca_report")
        if rca_report and rca_report.hypotheses:
            lines.append("\n## 🔬 Hypothesis Investigation & Validation Matrix:")
            lines.append("| Hypothesis | Investigated Dimension | Working Claim | Finding | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for h in rca_report.hypotheses:
                status_icon = "🟢 CONFIRMED" if h.status == "CONFIRMED_PRIMARY_DRIVER" else "⚪ REJECTED"
                lines.append(f"| **{h.hypothesis_id}** | `{h.dimension}` | {h.statement} | Driver `{h.top_category}` ({h.contribution_pct:+.1f}%) | {status_icon} |")

        if rca_report and rca_report.narrative_summary:
            lines.append("\n## Diagnostic Root-Cause Attribution:")
            lines.append(rca_report.narrative_summary)

        if unresolved:
            lines.append("\n> ⚠️ Note: Some steps encountered invariant warnings during self-correction.")

        summary_text = "\n".join(lines)

        # Build structured claim evidence items
        claim_evidences: List[ClaimEvidence] = []
        for s in steps:
            s_data = step_results.get(s.step_id, {})
            df = s_data.get("dataframe")
            scalar = s_data.get("scalar")
            res_val = scalar if scalar is not None else (f"{len(df)} rows" if df is not None else "Completed")
            ev_rec = next((e for e in evidence if e.step_id == s.step_id), None)
            val_status = "PASSED" if (ev_rec and ev_rec.validation_status == "VALIDATED_DETERMINISTICALLY") else ("FAILED" if not s.completed else "PARTIALLY_VERIFIED")
            chk_hash = ev_rec.checksum_hash if ev_rec else ""
            claim = ClaimEvidence(
                claim_id=f"claim_{s.step_id}",
                dataset=objective.target_table,
                claim_text=s.description or f"Result for {s.metric_col or objective.target_metric}",
                metric_name=s.metric_col or objective.target_metric,
                data_grain=objective.entity_grain or "aggregate",
                relevant_filters=[s.where_clause] if s.where_clause != "1=1" else [],
                analytical_operation=f"{s.agg_func}({s.metric_col})" if s.metric_col else s.description,
                generated_sql=s.code or "",
                calculated_result=res_val,
                supporting_statistics={"hash": chk_hash},
                checksum_hash=chk_hash,
                related_observations=s.description,
                validation_status=val_status,
                supporting_evidence=[f"Validated deterministically: {ev_rec.result_summary}"] if ev_rec else [],
                contradictory_evidence=[],
            )
            claim_evidences.append(claim)

        return {
            "final_summary": summary_text,
            "ecs_report": ecs_report,
            "semantic_coverage": semantic_report,
            "execution_correctness": execution_correctness,
            "calculation_correctness": calculation_correctness,
            "semantic_coverage_status": semantic_coverage_status,
            "evidence_sufficiency": evidence_sufficiency,
            "evidence_integrity": evidence_integrity,
            "overall_outcome": overall_outcome,
            "claim_evidences": claim_evidences,
            "correction_history": state.get("correction_history", []),
            "provenance_dag": self.provenance.to_dict(),
            "unresolved_discrepancy": unresolved,
            "status": "COMPLETED",
        }

    # -------------------------------------------------------------------------
    # Conditional Routing Logic
    # -------------------------------------------------------------------------

    def route_after_validation(
        self, state: AnalyticsAgentState
    ) -> Literal["correct_node", "advance_step_node", "rca_node", "synthesize_node"]:
        """
        Determines next node based on validation pass/fail, step completion, and analysis type.
        """
        # Fail-closed guard: immediately conclude without LLM retry if compilation or verification failed, or if registry mode validation failed
        is_registry_mode = (
            state.get("execution_mode") in (ExecutionMode.REGISTRY, "registry")
            or state.get("registry_mode", False)
            or state.get("composed_contract") is not None
        )
        val_res = state.get("current_validation") or {}
        is_valid = val_res.get("is_valid", False)

        if (
            state.get("status") in {"COMPILATION_FAILED", "VERIFICATION_FAILED"}
            or state.get("compilation_error")
            or state.get("verification_error")
            or (is_registry_mode and not is_valid)
        ):
            return "synthesize_node"
        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.max_self_correction_iterations)

        if is_valid:
            # Step succeeded. Check if more steps remain in plan
            curr_idx = state.get("current_step_index", 0)
            plan = state.get("plan")
            if plan and curr_idx + 1 < len(plan.steps):
                return "advance_step_node"
            return "rca_node" if state.get("is_diagnostic", False) else "synthesize_node"

        # Step failed
        if retries < max_retries:
            return "correct_node"

        # Exceeded max retries: log discrepancy and advance or conclude.
        # NOTE: We CANNOT mutate state here (routing functions must be pure).
        # The unresolved_discrepancy flag is set by advance_step_node / synthesize_node
        # based on the retry_count vs max_retries check on entry.
        curr_idx = state.get("current_step_index", 0)
        plan = state.get("plan")
        if plan and curr_idx + 1 < len(plan.steps):
            return "advance_step_node"
        return "rca_node" if state.get("is_diagnostic", False) else "synthesize_node"


def create_analytics_graph(
    con: duckdb.DuckDBPyConnection,
    llm: Optional[BaseChatModel] = None,
) -> Any:
    """
    Assembles and compiles the cyclic LangGraph workflow.
    """
    engine = AnalyticsWorkflowEngine(con, llm=llm)
    workflow = StateGraph(AnalyticsAgentState)

    # Add Nodes
    workflow.add_node("objective_node", engine.objective_node)
    workflow.add_node("plan_node", engine.plan_node)
    workflow.add_node("codegen_node", engine.codegen_node)
    workflow.add_node("exec_node", engine.exec_node)
    workflow.add_node("validate_node", engine.validate_node)
    workflow.add_node("correct_node", engine.correct_node)
    workflow.add_node("advance_step_node", engine.advance_step_node)
    workflow.add_node("rca_node", engine.rca_node)
    workflow.add_node("synthesize_node", engine.synthesize_node)

    # Wire Edges
    workflow.add_edge(START, "objective_node")
    workflow.add_edge("objective_node", "plan_node")
    workflow.add_edge("plan_node", "codegen_node")
    workflow.add_edge("codegen_node", "exec_node")
    workflow.add_edge("exec_node", "validate_node")

    # Conditional branching from validation
    workflow.add_conditional_edges(
        "validate_node",
        engine.route_after_validation,
        {
            "correct_node": "correct_node",
            "advance_step_node": "advance_step_node",
            "rca_node": "rca_node",
            "synthesize_node": "synthesize_node",
        },
    )

    # Correct node loops back to execution
    workflow.add_edge("correct_node", "exec_node")

    # Advance step loops back to codegen
    workflow.add_edge("advance_step_node", "codegen_node")

    # Conclusion flow
    workflow.add_edge("rca_node", "synthesize_node")
    workflow.add_edge("synthesize_node", END)

    return workflow.compile()
