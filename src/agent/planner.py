"""
Planner module — decomposes an AnalysisObjective into an ordered AnalysisPlan.
Supports both LLM-driven multi-step formulation and deterministic rule-based plan templates.
"""

from __future__ import annotations

import json
import re
from typing import Optional

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

## Planning Guidelines & Rules
1. Entity Deduplication: When asked to count entities (e.g. orders, customers, products), specify:
   - "agg_func": "COUNT_DISTINCT"
   - "metric_col": "<entity_id_col>" (e.g. "order_id")
   - "description": "Count unique <entity>s using COUNT(DISTINCT ...)"
2. Average Order Value (AOV):
   - Step A: Calculate unique orders: agg_func="COUNT_DISTINCT", metric_col="order_id"
   - Step B: Calculate total sales: agg_func="SUM", metric_col="sales"
   - Step C: Calculate Average Order Value (AOV): SUM(sales) / NULLIF(COUNT(DISTINCT order_id), 0)
3. Ranking / Highest Entity (e.g. "which product gives the highest profit"):
   - Set "group_col" to the entity column (e.g. "product_name") and "metric_col" to the target metric (e.g. "profit")
4. Filter Conditions: Always inherit and carry forward relevant WHERE filter conditions from the objective (e.g. year filters).

For each step, specify:
1. "step_id": (e.g. "step_1", "step_2")
2. "description": What to compute (e.g. "Calculate total revenue for Q3 2023 by region")
3. "metric_col": Numeric metric or entity ID column
4. "agg_func": "SUM", "AVG", "COUNT", or "COUNT_DISTINCT"
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
- Table: {target_table}
- Time Column: {time_column}
- Test Period: {test_period}
- Baseline Period: {baseline_period}
- Filter Conditions: {filters}
- Candidate Dimensions: {dimensions}

## Instructions
Decompose this into 2 to 4 actionable analytical steps.
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
        try:
            return self._plan_with_llm(profile, objective)
        except Exception:
            # Deterministic heuristic fallback
            return self.create_template_plan(profile, objective)

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
            target_table=objective.target_table,
            time_column=objective.time_column or "none",
            test_period=objective.test_period or "none",
            baseline_period=objective.baseline_period or "none",
            filters=", ".join(objective.filter_conditions) or "none",
            dimensions=", ".join(objective.candidate_dimensions) or "none",
        )

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)

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
        is_aov = "aov" in target_raw or "average order value" in target_raw

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

        primary_dim = (
            objective.candidate_dimensions[0]
            if objective.candidate_dimensions
            else None
        )

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
            if primary_dim and ("highest" in objective.target_metric.lower() or "product" in primary_dim.lower() or "name" in primary_dim.lower()):
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
                    description=f"Compute aggregate {clean_metric}",
                    metric_col=clean_metric,
                    agg_func=agg,
                    group_col=None,
                    where_clause=where_clause
                ))
                if primary_dim:
                    steps.append(PlanStep(
                        step_id="step_2",
                        description=f"Breakdown {clean_metric} grouped by {primary_dim}",
                        metric_col=clean_metric,
                        agg_func=agg,
                        group_col=primary_dim,
                        where_clause=where_clause
                    ))

        return AnalysisPlan(reasoning=reasoning, steps=steps)

    def _parse_json_plan(
        self,
        raw_text: str,
        profile: Optional[DatasetProfile] = None,
        objective: Optional[AnalysisObjective] = None,
    ) -> AnalysisPlan:
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

        # Ensure all requested metrics from objective are covered by the plan
        if objective and (objective.required_outputs or (objective.target_metrics and len(objective.target_metrics) > 1)):
            planned_descs = " ".join(s.description.lower() for s in plan.steps)
            is_distinct_order = "count(distinct" in objective.target_metric.lower() and "order" in objective.target_metric.lower()
            is_aov = "aov" in objective.target_metric.lower()
            if is_distinct_order and ("distinct" not in planned_descs and "unique" not in planned_descs):
                return self.create_template_plan(profile, objective)
            if is_aov and "aov" not in planned_descs and "average order value" not in planned_descs:
                return self.create_template_plan(profile, objective)

        # Post-process and sanitize steps
        tbl = profile.tables[0] if profile and profile.tables else None
        all_cols = [c.name for c in tbl.columns] if tbl else []
        order_col = next((c.name for c in (tbl.columns if tbl else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), None)
        sales_col = next((c.name for c in (tbl.columns if tbl else []) if c.name.lower() in ["sales", "revenue", "amount"]), "sales")

        from src.agent.schemas import resolve_metric_column
        for step in plan.steps:
            if all_cols:
                step.metric_col = resolve_metric_column(step.metric_col or (objective.target_metric if objective else "sales"), all_cols)

            desc_lower = step.description.lower()
            # Entity Deduplication Grain Guard
            if order_col and "order" in desc_lower and any(k in desc_lower for k in ["how many", "count", "number of", "total order"]):
                step.agg_func = "COUNT_DISTINCT"
                step.metric_col = order_col
                if "distinct" not in desc_lower:
                    step.description = f'Calculate total unique orders using COUNT(DISTINCT "{order_col}")'

            # AOV Guard
            if "aov" in desc_lower or "average order value" in desc_lower:
                step.agg_func = "AOV"
                step.metric_col = sales_col
                step.description = f'Calculate Average Order Value (AOV) as total {sales_col} divided by unique {order_col}'

        return plan

