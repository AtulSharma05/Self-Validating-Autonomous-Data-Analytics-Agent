"""
Planner module — decomposes an AnalysisObjective into an ordered AnalysisPlan.
Supports both LLM-driven multi-step formulation and deterministic rule-based plan templates.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.objective_extractor import _build_llm
from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    DatasetProfile,
    PlanStep,
)


_PLANNER_SYSTEM_PROMPT = """\
You are an expert analytical workflow planner. Given a dataset profile and an AnalysisObjective, \
decompose the objective into an ordered sequence of discrete, verifiable analytical steps.

## Authoritative Scope & Planning Guidelines
1. Authoritative Scope:
   - "required_outputs" and "target_metrics" define the STRICT, authoritative scope of the plan.
   - "candidate_dimensions" are optional schema context only. Do NOT perform exploratory breakdowns, slices, or groupings across candidate dimensions unless explicitly requested by the objective or user question.
   - For simple descriptive questions, generate ONLY the calculations necessary to satisfy requested outputs. Do NOT add unnecessary exploratory analysis steps.

2. Step Count Constraints:
   - Decompose into strictly 2 to 4 actionable analytical steps. Plans with more than 4 steps will be rejected.

3. Entity Deduplication: When asked to count entities (e.g. orders, customers, products), specify:
   - "agg_func": "COUNT_DISTINCT"
   - "metric_col": "<entity_id_col>" (e.g. "order_id")
   - "description": "Count unique <entity>s using COUNT(DISTINCT ...)"

4. Average Order Value (AOV):
   - AOV requires both unique order count and total sales volume:
     * Step A (Necessary intermediate / requested output): Calculate unique orders: agg_func="COUNT_DISTINCT", metric_col="order_id"
     * Step B (Necessary intermediate): Calculate total sales: agg_func="SUM", metric_col="sales"
     * Step C (Requested final output): Calculate Average Order Value (AOV): agg_func="AOV", metric_col="sales" (SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0))
   - NEVER use AVG(sales) for AOV, as that calculates the line-item average.

5. Requested Dimension Grouping vs Exploratory Analysis:
   - Only include "group_col" if the objective explicitly asks for grouping/breakdown/ranking by that dimension (e.g., "by region", "which product gives highest profit").
   - If no grouping was requested, keep "group_col": null.

6. Filter Conditions: Always inherit and carry forward relevant WHERE filter conditions from the objective (e.g. year filters).

For each step, specify:
1. "step_id": (e.g. "step_1", "step_2")
2. "description": What to compute (e.g. "Calculate total revenue for Q3 2023 by region")
3. "metric_col": Numeric metric or entity ID column
4. "agg_func": "SUM", "AVG", "COUNT", "COUNT_DISTINCT", or "AOV"
5. "group_col": Categorical dimension to group by, or null for scalar total
6. "where_clause": SQL WHERE clause (or "1=1" if none)

Respond ONLY with valid JSON:
{{
  "reasoning": "<Analytical justification for this sequence of steps>",
  "steps": [
    {{
      "step_id": "step_1",
      "description": "...",
      "metric_col": "...",
      "agg_func": "SUM",
      "group_col": null,
      "where_clause": "..."
    }}
  ]
}}
"""

_PLANNER_USER_TEMPLATE = """\
## Dataset Schema
{schema}

## Analysis Objective
- Analysis Type: {analysis_type}
- Target Metric: {target_metric}
- Target Metrics (all requested): {target_metrics}
- Required Outputs: {required_outputs}
- Entity Grain: {entity_grain}
- Table: {target_table}
- Time Column: {time_column}
- Test Period: {test_period}
- Baseline Period: {baseline_period}
- Filter Conditions: {filters}
- Candidate Dimensions (Context ONLY - do NOT break down unless requested): {dimensions}

## Instructions
Decompose this into strictly 2 to 4 actionable analytical steps.
Generate ONLY the steps necessary to satisfy the required outputs and their intermediate calculations.
Do not add unrequested dimension breakdowns or exploratory steps.
Respond with ONLY valid JSON:"""


class Planner:
    """
    Formulates a structured AnalysisPlan from an AnalysisObjective.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = _build_llm(json_mode=True)
        return self._llm

    def create_plan(
        self,
        profile: DatasetProfile,
        objective: AnalysisObjective,
    ) -> AnalysisPlan:
        """
        Generates a validated AnalysisPlan. Falls back to deterministic templates
        if LLM call fails or is not reachable.
        """
        used_planner = None
        plan = None
        try:
            plan = self._plan_with_llm(profile, objective)
            used_planner = "LLM planner"
            logger.debug("[create_plan] Planner used: %s", used_planner)
        except Exception as exc:
            used_planner = "deterministic fallback"
            logger.debug("[create_plan] LLM planner failed with exception: %s: %s", type(exc).__name__, exc, exc_info=True)
            logger.debug("[create_plan] Planner used: %s", used_planner)
            plan = self.create_template_plan(profile, objective)

        logger.debug("[create_plan] Resulting plan.model_dump():\n%s", json.dumps(plan.model_dump(), indent=2, default=str))
        return plan

    def _plan_with_llm(
        self,
        profile: DatasetProfile,
        objective: AnalysisObjective,
    ) -> AnalysisPlan:
        schema_context = profile.to_compact_llm_context()
        user_content = _PLANNER_USER_TEMPLATE.format(
            schema=schema_context,
            analysis_type=objective.analysis_type.value,
            target_metric=objective.target_metric,
            target_metrics=", ".join(objective.target_metrics) if objective.target_metrics else objective.target_metric,
            required_outputs=", ".join(objective.required_outputs) if objective.required_outputs else "none",
            entity_grain=objective.entity_grain or "none",
            target_table=objective.target_table,
            time_column=objective.time_column or "none",
            test_period=objective.test_period or "none",
            baseline_period=objective.baseline_period or "none",
            filters=", ".join(objective.filter_conditions) or "none",
            dimensions=", ".join(objective.candidate_dimensions) or "none",
        )

        logger.debug("[_plan_with_llm] Complete planner user prompt:\n%s", user_content)

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)

        logger.debug("[_plan_with_llm] Raw LLM response before parsing:\n%s", raw_text)

        return self._parse_json_plan(raw_text, profile, objective)

    def create_template_plan(
        self,
        profile: DatasetProfile,
        objective: AnalysisObjective,
    ) -> AnalysisPlan:
        """
        Rule-based deterministic planner ensuring 100% reliability offline.
        """
        where_clause = " AND ".join(objective.filter_conditions) if objective.filter_conditions else "1=1"
        target_raw = objective.target_metric.lower()
        tbl = profile.tables[0] if profile.tables else None
        all_cols = [c.name for c in tbl.columns] if tbl else []
        sales_col = next((c.name for c in (tbl.columns if tbl else []) if c.name.lower() in ["sales", "revenue", "amount"]), "sales")
        profit_col = next((c.name for c in (tbl.columns if tbl else []) if "profit" in c.name.lower()), "profit")
        order_col = next((c.name for c in (tbl.columns if tbl else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), "order_id")

        # 0. Check for missing columns explanation on objective
        if getattr(objective, "missing_columns_explanation", None):
            return AnalysisPlan(
                reasoning=f"Cannot execute requested analysis: {objective.missing_columns_explanation}",
                steps=[
                    PlanStep(
                        step_id="step_1",
                        description=f"Missing column check: {objective.missing_columns_explanation}",
                        metric_col="unavailable",
                        agg_func="INFO",
                        where_clause="1=1",
                    )
                ]
            )

        # Check for Data Quality / Bidirectional Mapping Consistency Checks
        if objective.analysis_type == AnalysisType.DATA_QUALITY or objective.quality_check_type == "bidirectional_mapping":
            entities = objective.conflict_entities if len(objective.conflict_entities) >= 2 else ["product_id", "product_name"]
            ent_a, ent_b = entities[0], entities[1]
            steps = [
                PlanStep(
                    step_id="step_1",
                    description=f'Check if any {ent_a} is mapped to multiple {ent_b} (HAVING COUNT(DISTINCT "{ent_b}") > 1)',
                    metric_col=ent_b,
                    agg_func="COUNT_DISTINCT",
                    group_col=ent_a,
                    where_clause=where_clause,
                ),
                PlanStep(
                    step_id="step_2",
                    description=f'Check if any {ent_b} is mapped to multiple {ent_a} (HAVING COUNT(DISTINCT "{ent_a}") > 1)',
                    metric_col=ent_a,
                    agg_func="COUNT_DISTINCT",
                    group_col=ent_b,
                    where_clause=where_clause,
                ),
            ]
            reasoning = f"Execute two independent integrity checks: verify {ent_a} -> {ent_b} consistency and {ent_b} -> {ent_a} consistency."
            return AnalysisPlan(reasoning=reasoning, steps=steps)

        # Check for discount capping counterfactual scenario
        if objective.is_scenario and objective.scenario_params.get("type") == "discount_cap":
            cap_val = objective.scenario_params.get("threshold", 0.20)
            cap_pct = int(cap_val * 100)
            steps = [
                PlanStep(
                    step_id="step_1",
                    description=f"Calculate revised total sales with maximum discount capped at {cap_pct}%",
                    metric_col=sales_col,
                    agg_func="SUM",
                    where_clause=where_clause,
                ),
                PlanStep(
                    step_id="step_2",
                    description=f"Calculate revised total profit with maximum discount capped at {cap_pct}%",
                    metric_col=profit_col,
                    agg_func="SUM",
                    where_clause=where_clause,
                ),
                PlanStep(
                    step_id="step_3",
                    description=f"Calculate revised profit margin percentage with maximum discount capped at {cap_pct}%",
                    metric_col=profit_col,
                    agg_func="AVG",
                    where_clause=where_clause,
                ),
            ]
            reasoning = f"Simulate counterfactual scenario capping maximum discounts at {cap_pct}% to compute revised total sales, total profit, and profit margin."
            return AnalysisPlan(reasoning=reasoning, steps=steps)

        # Check for entity counting (e.g. unique orders) and AOV
        is_distinct_order = "count(distinct" in target_raw and "order" in target_raw
        is_aov = (
            "aov" in target_raw
            or "average order value" in target_raw
            or any("aov" in m.lower() or "average order value" in m.lower() for m in (objective.target_metrics or []))
            or any("aov" in ro.lower() or "average order value" in ro.lower() for ro in (objective.required_outputs or []))
        )

        if is_aov:
            steps = [
                PlanStep(
                    step_id="step_1",
                    description=f'Calculate total unique orders using COUNT(DISTINCT "{order_col}")',
                    metric_col=order_col,
                    agg_func="COUNT_DISTINCT",
                    group_col=None,
                    where_clause=where_clause,
                ),
                PlanStep(
                    step_id="step_2",
                    description=f'Calculate total sales volume for "{sales_col}"',
                    metric_col=sales_col,
                    agg_func="SUM",
                    group_col=None,
                    where_clause=where_clause,
                ),
                PlanStep(
                    step_id="step_3",
                    description=f'Calculate Average Order Value (AOV) as total {sales_col} divided by unique {order_col}',
                    metric_col=sales_col,
                    agg_func="AOV",
                    group_col=None,
                    where_clause=where_clause,
                ),
            ]
            reasoning = "Determine unique entity count, total sales volume, and compute true Average Order Value (AOV) preventing line-item grain failure."
            return AnalysisPlan(reasoning=reasoning, steps=steps)

        if is_distinct_order:
            steps = [
                PlanStep(
                    step_id="step_1",
                    description=f'Calculate total unique orders using COUNT(DISTINCT "{order_col}")',
                    metric_col=order_col,
                    agg_func="COUNT_DISTINCT",
                    group_col=None,
                    where_clause=where_clause,
                ),
            ]
            reasoning = "Calculate deduplicated unique entity count."
            return AnalysisPlan(reasoning=reasoning, steps=steps)

        # Check for multiple requested target metrics
        if objective.target_metrics and len(objective.target_metrics) > 1:
            from src.agent.schemas import resolve_metric_column
            steps = []
            for idx, m in enumerate(objective.target_metrics, start=1):
                m_clean = m.lower().strip()
                if "profit_margin" in m_clean or "margin" in m_clean:
                    steps.append(PlanStep(
                        step_id=f"step_{idx}",
                        description="Calculate profit margin percentage",
                        metric_col=profit_col,
                        agg_func="AVG",
                        where_clause=where_clause,
                    ))
                elif "sales" in m_clean or "revenue" in m_clean:
                    steps.append(PlanStep(
                        step_id=f"step_{idx}",
                        description=f"Calculate total {sales_col}",
                        metric_col=sales_col,
                        agg_func="SUM",
                        where_clause=where_clause,
                    ))
                elif "profit" in m_clean:
                    steps.append(PlanStep(
                        step_id=f"step_{idx}",
                        description=f"Calculate total {profit_col}",
                        metric_col=profit_col,
                        agg_func="SUM",
                        where_clause=where_clause,
                    ))
                elif "order" in m_clean:
                    steps.append(PlanStep(
                        step_id=f"step_{idx}",
                        description=f'Calculate total unique orders using COUNT(DISTINCT "{order_col}")',
                        metric_col=order_col,
                        agg_func="COUNT_DISTINCT",
                        where_clause=where_clause,
                    ))
                elif "aov" in m_clean:
                    steps.append(PlanStep(
                        step_id=f"step_{idx}",
                        description=f'Calculate Average Order Value (AOV) as total {sales_col} divided by unique {order_col}',
                        metric_col=sales_col,
                        agg_func="AOV",
                        where_clause=where_clause,
                    ))
                else:
                    resolved = resolve_metric_column(m_clean, all_cols)
                    steps.append(PlanStep(
                        step_id=f"step_{idx}",
                        description=f"Calculate aggregate {resolved}",
                        metric_col=resolved,
                        agg_func="SUM",
                        where_clause=where_clause,
                    ))
            reasoning = f"Calculate multiple requested outputs: {', '.join(objective.target_metrics)}."
            return AnalysisPlan(reasoning=reasoning, steps=steps)

        # Check for Period-over-Period Comparative analysis
        if objective.analysis_type == AnalysisType.COMPARATIVE and objective.baseline_period and objective.test_period:
            steps = [
                PlanStep(
                    step_id="step_1",
                    description=f'Calculate baseline total {sales_col} for {objective.baseline_period}',
                    metric_col=sales_col,
                    agg_func="SUM",
                    where_clause=objective.baseline_period,
                ),
                PlanStep(
                    step_id="step_2",
                    description=f'Calculate test total {sales_col} for {objective.test_period}',
                    metric_col=sales_col,
                    agg_func="SUM",
                    where_clause=objective.test_period,
                ),
                PlanStep(
                    step_id="step_3",
                    description=f'Calculate percentage change in {sales_col} from {objective.baseline_period} to {objective.test_period}',
                    metric_col=sales_col,
                    agg_func="AVG",
                    where_clause=where_clause,
                ),
            ]
            reasoning = f"Compare baseline ({objective.baseline_period}) against test ({objective.test_period}) and compute percentage change."
            return AnalysisPlan(reasoning=reasoning, steps=steps)

        # Standard metric extraction
        from src.agent.schemas import resolve_metric_column
        clean_metric = resolve_metric_column(objective.target_metric, all_cols)

        agg = "SUM"
        if "avg" in objective.target_metric.lower():
            agg = "AVG"
        elif "count" in objective.target_metric.lower():
            agg = "COUNT"

        # Find requested dimension if any
        requested_dim = next((d for d in (objective.candidate_dimensions or []) if self._is_grouping_requested(objective, d)), None)
        primary_dim = requested_dim or (objective.candidate_dimensions[0] if objective.candidate_dimensions else None)

        steps: list[PlanStep] = []

        if objective.analysis_type == AnalysisType.DIAGNOSTIC:
            reasoning = "Perform baseline comparison followed by dimensional factor decomposition."
            steps.append(PlanStep(
                step_id="step_1",
                description=f"Calculate overall total {clean_metric}",
                metric_col=clean_metric,
                agg_func=agg,
                group_col=None,
                where_clause=where_clause
            ))
            if primary_dim:
                steps.append(PlanStep(
                    step_id="step_2",
                    description=f"Decompose {clean_metric} across {primary_dim}",
                    metric_col=clean_metric,
                    agg_func=agg,
                    group_col=primary_dim,
                    where_clause=where_clause
                ))
            if len(objective.candidate_dimensions) > 1:
                sec_dim = objective.candidate_dimensions[1]
                steps.append(PlanStep(
                    step_id="step_3",
                    description=f"Decompose {clean_metric} across secondary dimension {sec_dim}",
                    metric_col=clean_metric,
                    agg_func=agg,
                    group_col=sec_dim,
                    where_clause=where_clause
                ))
        else:
            reasoning = f"Execute descriptive/comparative aggregation of {clean_metric}."
            is_ranking = any(k in objective.target_metric.lower() for k in ["highest", "top", "rank", "best", "most", "lowest", "least", "which"]) or any(
                any(k in ro.lower() for k in ["highest", "top", "rank", "best", "most", "lowest", "least", "which"])
                for ro in (objective.required_outputs or [])
            ) or (
                primary_dim and objective.target_table and any(k in primary_dim.lower() for k in ["product", "name"]) and "product" in objective.target_table.lower()
            )
            if primary_dim and self._is_grouping_requested(objective, primary_dim):
                if is_ranking:
                    # When looking for top/highest entity, breakdown first to answer user's question directly
                    steps.append(PlanStep(
                        step_id="step_1",
                        description=f"Calculate {clean_metric} grouped by {primary_dim} to identify the highest {primary_dim}",
                        metric_col=clean_metric,
                        agg_func=agg,
                        group_col=primary_dim,
                        where_clause=where_clause
                    ))
                    steps.append(PlanStep(
                        step_id="step_2",
                        description=f"Compute overall aggregate {clean_metric} for verification",
                        metric_col=clean_metric,
                        agg_func=agg,
                        group_col=None,
                        where_clause=where_clause
                    ))
                else:
                    steps.append(PlanStep(
                        step_id="step_1",
                        description=f"Calculate {clean_metric} grouped by {primary_dim}",
                        metric_col=clean_metric,
                        agg_func=agg,
                        group_col=primary_dim,
                        where_clause=where_clause
                    ))
                    steps.append(PlanStep(
                        step_id="step_2",
                        description=f"Compute overall aggregate {clean_metric} for verification",
                        metric_col=clean_metric,
                        agg_func=agg,
                        group_col=None,
                        where_clause=where_clause
                    ))
            else:
                steps.append(PlanStep(
                    step_id="step_1",
                    description=f"Compute aggregate {clean_metric}",
                    metric_col=clean_metric,
                    agg_func=agg,
                    group_col=None,
                    where_clause=where_clause
                ))

        return AnalysisPlan(reasoning=reasoning, steps=steps)

    def _is_grouping_requested(
        self,
        objective: Optional[AnalysisObjective],
        dim: Optional[str],
    ) -> bool:
        """
        Returns True if the objective or query explicitly requested grouping by this dimension,
        or if authorized for legitimate diagnostic exploration.
        """
        if not objective or not dim:
            return False

        dim_lower = dim.lower()
        dim_base = dim_lower.replace("_name", "").replace("_id", "").replace("_key", "").replace("_number", "")

        tm = (objective.target_metric or "").lower()

        # 1. Direct explicit mention in target_metric
        if dim_lower in tm or (len(dim_base) >= 4 and dim_base in tm):
            return True
        if f"by {dim_lower}" in tm or f"per {dim_lower}" in tm or f"grouped by {dim_lower}" in tm:
            return True
        if len(dim_base) >= 4 and (f"by {dim_base}" in tm or f"per {dim_base}" in tm or f"grouped by {dim_base}" in tm):
            return True

        # 2. Direct explicit mention in required_outputs
        for ro in (objective.required_outputs or []):
            ro_lower = ro.lower()
            if dim_lower in ro_lower or (len(dim_base) >= 4 and dim_base in ro_lower):
                return True
            if f"by {dim_lower}" in ro_lower or f"per {dim_lower}" in ro_lower:
                return True
            if len(dim_base) >= 4 and (f"by {dim_base}" in ro_lower or f"per {dim_base}" in ro_lower):
                return True

        # 3. Direct explicit mention in target_metrics
        for m in (objective.target_metrics or []):
            m_lower = m.lower()
            if dim_lower in m_lower or (len(dim_base) >= 4 and dim_base in m_lower):
                return True
            if f"by {dim_lower}" in m_lower or f"per {dim_lower}" in m_lower:
                return True

        # 4. Explicit entity grain match
        if objective.entity_grain:
            eg_lower = objective.entity_grain.lower()
            if eg_lower == dim_lower or eg_lower == dim_base or (len(dim_base) >= 4 and dim_base in eg_lower):
                return True

        # 5. Ranking questions: Only authorize if this specific entity is the requested ranking target
        is_ranking = any(k in tm for k in ["highest", "top", "rank", "best", "most", "lowest", "least", "which"]) or any(
            any(k in ro.lower() for k in ["highest", "top", "rank", "best", "most", "lowest", "least", "which"])
            for ro in (objective.required_outputs or [])
        ) or (
            objective.target_table and len(dim_base) >= 4 and dim_base in objective.target_table.lower()
        )
        if is_ranking:
            ranking_targets = [tm] + [ro.lower() for ro in (objective.required_outputs or [])]
            if objective.entity_grain:
                ranking_targets.append(objective.entity_grain.lower())
            if objective.target_table:
                ranking_targets.append(objective.target_table.lower())
            for target_text in ranking_targets:
                if dim_lower in target_text or (len(dim_base) >= 4 and dim_base in target_text):
                    return True
            return False

        # 6. Diagnostic exploration:
        # Permit legitimate RCA exploration for up to primary/secondary diagnostic dimensions or filter conditions,
        # but do not automatically authorize every candidate dimension in the schema.
        if objective.analysis_type == AnalysisType.DIAGNOSTIC:
            for filt in (objective.filter_conditions or []):
                if dim_lower in filt.lower():
                    return True
            top_diagnostic_dims = [d.lower() for d in (objective.candidate_dimensions or [])[:2]]
            if dim_lower in top_diagnostic_dims:
                return True
            return False

        return False

    def _validate_plan_coverage(
        self,
        plan: AnalysisPlan,
        objective: AnalysisObjective,
    ) -> bool:
        """
        Validates that every requested output and target metric is covered by the plan steps.
        Returns False if any required metric is missing or improperly computed.
        """
        if not plan.steps:
            return False

        step_texts = [
            f"{s.step_id} {s.description} {s.metric_col} {s.agg_func} {s.group_col or ''}".lower()
            for s in plan.steps
        ]

        # Check required_outputs
        if objective.required_outputs:
            for ro in objective.required_outputs:
                ro_lower = ro.lower()
                if "order" in ro_lower and any(k in ro_lower for k in ["total", "count", "unique", "how many"]):
                    if not any("order" in t and ("distinct" in t or "count_distinct" in t or "unique" in t) for t in step_texts):
                        return False
                elif "aov" in ro_lower or "average order value" in ro_lower:
                    if not any("aov" in t or "average order value" in t for t in step_texts):
                        return False
                elif "sales" in ro_lower or "revenue" in ro_lower:
                    if not any(("sales" in t or "revenue" in t) and ("sum" in t or "total" in t) for t in step_texts):
                        return False
                elif "profit" in ro_lower and "margin" not in ro_lower:
                    if not any("profit" in t and ("sum" in t or "total" in t) for t in step_texts):
                        return False
                elif "margin" in ro_lower:
                    if not any("margin" in t for t in step_texts):
                        return False

        # Check target_metrics
        if objective.target_metrics:
            for m in objective.target_metrics:
                m_lower = m.lower()
                if "order" in m_lower:
                    if not any("order" in t and ("distinct" in t or "count_distinct" in t) for t in step_texts):
                        return False
                elif "aov" in m_lower or "average order value" in m_lower:
                    if not any("aov" in t or "average order value" in t for t in step_texts):
                        return False
                elif "profit" in m_lower and "margin" not in m_lower:
                    if not any("profit" in t for t in step_texts):
                        return False
                elif "sales" in m_lower or "revenue" in m_lower:
                    if not any("sales" in t or "revenue" in t for t in step_texts):
                        return False
                elif "margin" in m_lower:
                    if not any("margin" in t for t in step_texts):
                        return False

        # If AOV was requested, ensure we do NOT calculate AOV via AVG(sales) alone without unique orders
        is_aov_requested = (
            "aov" in objective.target_metric.lower()
            or "average order value" in objective.target_metric.lower()
            or any("aov" in ro.lower() or "average order value" in ro.lower() for ro in (objective.required_outputs or []))
            or any("aov" in m.lower() or "average order value" in m.lower() for m in (objective.target_metrics or []))
        )
        if is_aov_requested:
            has_order_distinct = any(s.agg_func == "COUNT_DISTINCT" or "distinct" in s.description.lower() for s in plan.steps)
            has_aov_or_sales_div = any(s.agg_func == "AOV" or "aov" in s.description.lower() for s in plan.steps)
            if not (has_order_distinct and has_aov_or_sales_div):
                return False

        return True

    def _normalize_and_filter_steps(
        self,
        steps: list[PlanStep],
        profile: Optional[DatasetProfile],
        objective: Optional[AnalysisObjective],
    ) -> list[PlanStep]:
        """
        Deterministically filters unnecessary exploratory steps, protects necessary
        intermediate calculations, enforces step constraints, and renumbers steps.
        """
        if not objective:
            return steps

        tbl = profile.tables[0] if profile and profile.tables else None
        all_cols = [c.name for c in tbl.columns] if tbl else []
        order_col = next((c.name for c in (tbl.columns if tbl else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), "order_id")
        sales_col = next((c.name for c in (tbl.columns if tbl else []) if c.name.lower() in ["sales", "revenue", "amount"]), "sales")
        profit_col = next((c.name for c in (tbl.columns if tbl else []) if "profit" in c.name.lower()), "profit")

        from src.agent.schemas import resolve_metric_column

        is_aov = (
            "aov" in objective.target_metric.lower()
            or "average order value" in objective.target_metric.lower()
            or any("aov" in ro.lower() or "average order value" in ro.lower() for ro in (objective.required_outputs or []))
            or any("aov" in m.lower() or "average order value" in m.lower() for m in (objective.target_metrics or []))
        )

        kept_steps: list[PlanStep] = []

        for step in steps:
            if all_cols and step.metric_col:
                step.metric_col = resolve_metric_column(step.metric_col, all_cols)

            desc_lower = step.description.lower()

            # Entity Deduplication Grain Guard
            if order_col and "order" in desc_lower and any(k in desc_lower for k in ["how many", "count", "number of", "total order"]):
                step.agg_func = "COUNT_DISTINCT"
                step.metric_col = order_col
                if "distinct" not in desc_lower:
                    step.description = f'Calculate total unique orders using COUNT(DISTINCT "{order_col}")'

            # AOV Guard: Never allow AVG(sales) for AOV
            if "aov" in desc_lower or "average order value" in desc_lower or step.agg_func.upper() == "AOV":
                step.agg_func = "AOV"
                step.metric_col = sales_col
                step.description = f'Calculate Average Order Value (AOV) as total {sales_col} divided by unique {order_col}'

            # Check if this step has an unrequested dimension grouping
            if step.group_col:
                if not self._is_grouping_requested(objective, step.group_col):
                    logger.debug("[_normalize_and_filter_steps] Pruning unrequested dimensional step: %s (group_col=%s)", step.step_id, step.group_col)
                    continue

            # Classify step: Requested Final Output, Necessary Intermediate, or Unnecessary
            is_requested = False
            for ro in (objective.required_outputs or []):
                ro_lower = ro.lower()
                if any(w in desc_lower for w in ro_lower.split() if len(w) > 3):
                    is_requested = True
                    break
            for tm in (objective.target_metrics or []):
                tm_lower = tm.lower()
                if step.metric_col.lower() in tm_lower or tm_lower in desc_lower:
                    is_requested = True
                    break
            if objective.target_metric.lower() in desc_lower or step.metric_col.lower() in objective.target_metric.lower():
                is_requested = True

            # Necessary intermediate calculations
            is_necessary_intermediate = False
            if is_aov:
                if step.agg_func == "SUM" and step.metric_col == sales_col:
                    is_necessary_intermediate = True
                if step.agg_func == "COUNT_DISTINCT" and step.metric_col == order_col:
                    is_necessary_intermediate = True

            if "margin" in objective.target_metric.lower() or any("margin" in m.lower() for m in (objective.target_metrics or [])):
                if step.agg_func == "SUM" and (step.metric_col == profit_col or step.metric_col == sales_col):
                    is_necessary_intermediate = True

            # For DIAGNOSTIC mode:
            is_diagnostic_legitimate = False
            if objective.analysis_type == AnalysisType.DIAGNOSTIC:
                if step.group_col is None or self._is_grouping_requested(objective, step.group_col):
                    is_diagnostic_legitimate = True

            if is_requested or is_necessary_intermediate or is_diagnostic_legitimate:
                kept_steps.append(step)
            else:
                logger.debug("[_normalize_and_filter_steps] Pruning unnecessary exploratory step: %s (%s)", step.step_id, step.description)

        # For AOV: Ensure necessary intermediate sales volume step is present
        if is_aov:
            has_sales = any(s.agg_func == "SUM" and s.metric_col == sales_col and not s.group_col for s in kept_steps)
            if not has_sales and len(kept_steps) < 4:
                where_cl = kept_steps[0].where_clause if kept_steps else "1=1"
                sales_step = PlanStep(
                    step_id="temp_sales",
                    description=f'Calculate total sales volume for "{sales_col}"',
                    metric_col=sales_col,
                    agg_func="SUM",
                    group_col=None,
                    where_clause=where_cl,
                )
                aov_idx = next((i for i, s in enumerate(kept_steps) if s.agg_func == "AOV"), len(kept_steps))
                kept_steps.insert(aov_idx, sales_step)

        # Enforce max 4 steps: prune extras while protecting essential requested & intermediate steps
        if len(kept_steps) > 4:
            essential = []
            extras = []
            for s in kept_steps:
                desc = s.description.lower()
                is_step_essential = False
                if any(ro.lower() in desc for ro in (objective.required_outputs or [])):
                    is_step_essential = True
                elif any(tm.lower() in desc or s.metric_col.lower() in tm.lower() for tm in (objective.target_metrics or [])):
                    is_step_essential = True
                elif is_aov and (s.agg_func in ["AOV", "COUNT_DISTINCT"] or (s.agg_func == "SUM" and s.metric_col == sales_col)):
                    is_step_essential = True
                elif "margin" in objective.target_metric.lower() and s.agg_func == "SUM" and s.metric_col in [profit_col, sales_col]:
                    is_step_essential = True
                elif objective.analysis_type == AnalysisType.DIAGNOSTIC and (s.group_col is None or (objective.candidate_dimensions and s.group_col == objective.candidate_dimensions[0])):
                    is_step_essential = True

                if is_step_essential:
                    essential.append(s)
                else:
                    extras.append(s)

            if len(essential) > 4:
                logger.debug("[_normalize_and_filter_steps] Essential steps count (%d) exceeds 4 limit; falling back to template plan", len(essential))
                return self.create_template_plan(profile, objective).steps

            allowed_extras = 4 - len(essential)
            kept_steps = essential + extras[:allowed_extras]

        # Re-assign sequential step IDs
        for i, s in enumerate(kept_steps, start=1):
            s.step_id = f"step_{i}"

        return kept_steps

    def _parse_json_plan(
        self,
        raw_text: str,
        profile: Optional[DatasetProfile] = None,
        objective: Optional[AnalysisObjective] = None,
    ) -> AnalysisPlan:
        try:
            cleaned = raw_text.strip()
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                m2 = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if m2:
                    cleaned = m2.group(0)

            data = json.loads(cleaned)
            plan = AnalysisPlan(**data)
        except Exception as exc:
            logger.debug("[_parse_json_plan] Failed to parse JSON plan: %s, falling back to deterministic template", exc)
            return self.create_template_plan(profile, objective)

        if not objective:
            return plan

        # Filter and normalize steps
        plan.steps = self._normalize_and_filter_steps(plan.steps, profile, objective)

        is_aov = (
            "aov" in objective.target_metric.lower()
            or "average order value" in objective.target_metric.lower()
            or any("aov" in ro.lower() or "average order value" in ro.lower() for ro in (objective.required_outputs or []))
            or any("aov" in m.lower() or "average order value" in m.lower() for m in (objective.target_metrics or []))
        )
        is_simple_scalar = (
            len(objective.required_outputs) <= 1
            and len(objective.target_metrics) <= 1
            and not is_aov
            and objective.analysis_type != AnalysisType.DIAGNOSTIC
        )

        # Enforce 2-4 step constraint
        if len(plan.steps) > 4 or len(plan.steps) < 2:
            if not (is_simple_scalar and len(plan.steps) == 1):
                logger.debug("[_parse_json_plan] Plan step count %d outside 2-4 constraint, falling back to deterministic template", len(plan.steps))
                return self.create_template_plan(profile, objective)

        # Validate semantic coverage
        if not self._validate_plan_coverage(plan, objective):
            logger.debug("[_parse_json_plan] Plan failed semantic coverage validation, falling back to deterministic template")
            return self.create_template_plan(profile, objective)

        return plan

