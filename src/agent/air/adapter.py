"""Bidirectional Adapter: Lossless Legacy Snapshot & Semantic AIR v2.1 Mapping."""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from src.agent.schemas import (
    AnalysisObjective,
    AnalysisPlan,
    AnalysisType,
    ColumnProfile,
    DatasetProfile,
    PlanStep,
    TableProfile,
    resolve_metric_column,
)
from src.agent.air.schemas import (
    AIRLifecycleState,
    ColumnRef,
    ComparisonOperator,
    ConversionDiagnostic,
    DataType,
    DimensionAuthorizationMatrix,
    FeasibilityStatus,
    GrainAssessment,
    GrainConfidence,
    LegacyContext,
    Predicate,
)
from src.agent.air.operations import (
    AggregateMeasureSpec,
    AggregateOperation,
    AggregationFunction,
    AnalyticalOperation,
    DerivedExpressionSpec,
    FilterOperation,
    RankOperation,
    SortCriterion,
    SortDirection,
)
from src.agent.air.dag import MasterAIR
from src.agent.air.filter_parser import StrictFilterParser
from src.agent.air.temporal import TemporalBoundaryResolver


class AIRAdapterResult(BaseModel):
    air: Optional[MasterAIR] = None
    diagnostics: List[ConversionDiagnostic] = Field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return self.air is not None and not any(d.is_fatal for d in self.diagnostics)


# D1: Explicit, deterministic ranking entity map
# Longer phrases precede shorter ones so 'product category' matches before 'product'
RANKING_ENTITY_MAP: List[Tuple[str, List[str]]] = [
    ("product category", ["product_category", "category"]),
    ("product categories", ["product_category", "category"]),
    ("category", ["product_category", "category"]),
    ("categories", ["product_category", "category"]),
    ("sub category", ["sub_category", "subcategory"]),
    ("sub-category", ["sub_category", "subcategory"]),
    ("sub categories", ["sub_category", "subcategory"]),
    ("sub-categories", ["sub_category", "subcategory"]),
    ("products", ["product_name", "product_id"]),
    ("product", ["product_name", "product_id"]),
    ("customers", ["customer_name", "customer_id"]),
    ("customer", ["customer_name", "customer_id"]),
    ("orders", ["order_id", "order_number"]),
    ("order", ["order_id", "order_number"]),
    ("regions", ["region"]),
    ("region", ["region"]),
    ("states", ["state"]),
    ("state", ["state"]),
    ("cities", ["city"]),
    ("city", ["city"]),
    ("segments", ["segment"]),
    ("segment", ["segment"]),
    ("ship mode", ["ship_mode", "shipping_mode"]),
    ("shipping mode", ["ship_mode", "shipping_mode"]),
]

DIM_SYNONYMS: Dict[str, str] = {
    "region": "region",
    "ship mode": "ship_mode",
    "shipping mode": "ship_mode",
    "ship_mode": "ship_mode",
    "segment": "segment",
    "category": "category",
    "product category": "product_category",
    "sub category": "sub_category",
    "sub_category": "sub_category",
    "state": "state",
    "city": "city",
    "customer": "customer_name",
    "product": "product_name",
}


class AIRAdapter:
    """Lossless Legacy Translation with deep context snapshot."""

    @classmethod
    def from_legacy(
        cls,
        objective: AnalysisObjective,
        plan: Optional[AnalysisPlan] = None,
        profile: Optional[DatasetProfile] = None,
        user_question: str = "",
    ) -> AIRAdapterResult:
        diagnostics: List[ConversionDiagnostic] = []
        question = user_question or objective.target_metric

        # 1. Deep lossless legacy snapshot
        legacy_context = LegacyContext(
            objective_snapshot=objective.model_copy(deep=True),
            plan_snapshot=plan.model_copy(deep=True) if plan else None,
        )

        # 2. Dataset schema lookup
        table_name = objective.target_table
        table_prof: Optional[TableProfile] = None
        if profile and profile.tables:
            table_prof = next((t for t in profile.tables if t.table_name == table_name), None)

        table_columns: List[str] = [c.name for c in table_prof.columns] if table_prof else []
        column_type_map: Dict[str, DataType] = {}
        if table_prof:
            for col in table_prof.columns:
                dt = col.data_type.lower()
                if any(k in dt for k in ["int", "bigint", "smallint"]):
                    column_type_map[col.name] = DataType.INTEGER
                elif any(k in dt for k in ["float", "double", "decimal", "numeric"]):
                    column_type_map[col.name] = DataType.FLOAT
                elif any(k in dt for k in ["date", "timestamp"]):
                    column_type_map[col.name] = DataType.DATETIME
                elif "bool" in dt:
                    column_type_map[col.name] = DataType.BOOLEAN
                else:
                    column_type_map[col.name] = DataType.STRING

        # 3. Evidence-Based Grain Assessment
        target_entity = objective.entity_grain or "unknown"
        confidence = GrainConfidence.UNKNOWN
        cardinality_ratio = 0.0
        candidate_keys: List[str] = []
        requires_rollup = False
        evidence_notes: List[str] = []

        if table_prof:
            candidate_col = next(
                (c for c in table_prof.columns if target_entity.lower() in c.name.lower() and c.name.lower().endswith(("_id", "_key", "id"))),
                None,
            )
            if not candidate_col and table_prof.primary_key_candidates:
                candidate_col = next((c for c in table_prof.columns if c.name in table_prof.primary_key_candidates), None)

            if candidate_col and table_prof.row_count > 0:
                candidate_keys = [candidate_col.name]
                cardinality_ratio = candidate_col.distinct_count / float(table_prof.row_count)
                if candidate_col.distinct_count == table_prof.row_count and candidate_col.null_count == 0:
                    confidence = GrainConfidence.PROVEN_UNIQUE
                    evidence_notes.append(f"Key '{candidate_col.name}' is proven unique ({table_prof.row_count} rows, 0 nulls).")
                elif candidate_col.distinct_count < table_prof.row_count:
                    confidence = GrainConfidence.INFERRED_MULTI_GRAIN
                    requires_rollup = True
                    evidence_notes.append(
                        f"Entity '{target_entity}' key '{candidate_col.name}' has {candidate_col.distinct_count:,} distinct "
                        f"values across {table_prof.row_count:,} rows. Multi-grain rollup required."
                    )
                else:
                    confidence = GrainConfidence.CANDIDATE
                    evidence_notes.append(f"Key '{candidate_col.name}' is a candidate key; uniqueness not proven.")
            elif table_prof.primary_key_candidates:
                candidate_keys = list(table_prof.primary_key_candidates)
                confidence = GrainConfidence.CANDIDATE
                evidence_notes.append(f"TableProfile declared candidate keys: {candidate_keys}, but empirical evidence is unverified.")
            else:
                evidence_notes.append("No primary key candidate found in metadata.")
        else:
            evidence_notes.append("No TableProfile provided; grain confidence remains UNKNOWN.")

        grain_assessment = GrainAssessment(
            target_entity=target_entity,
            candidate_keys=candidate_keys,
            confidence=confidence,
            cardinality_ratio=cardinality_ratio,
            supporting_evidence=" ".join(evidence_notes),
            requires_entity_rollup=requires_rollup,
        )

        # 4. Fail-Closed Dimension Authorization Matrix (D1 & D2)
        dimensions, dim_diag = cls._authorize_dimensions(objective, question)
        if dim_diag:
            diagnostics.append(dim_diag)
            if dim_diag.is_fatal:
                return AIRAdapterResult(air=None, diagnostics=diagnostics)

        # 5. Semantic Operation Compilation (Phase 1: Filter, Aggregate, Rank)
        operations: List[AnalyticalOperation] = []
        terminal_ids: List[str] = []

        if plan and plan.steps:
            current_dataset = table_name
            current_schema = dict(column_type_map)
            prev_op_id: Optional[str] = None

            for idx, step in enumerate(plan.steps, start=1):
                is_terminal = (idx == len(plan.steps))
                step_ops, diag = cls._map_plan_step(
                    step=step,
                    step_index=idx,
                    current_dataset=current_dataset,
                    current_schema=current_schema,
                    prev_op_id=prev_op_id,
                    is_terminal=is_terminal,
                    objective=objective,
                    table_columns=table_columns,
                    table_name=table_name,
                    question=question,
                )

                if diag:
                    diagnostics.append(diag)
                    if diag.is_fatal:
                        return AIRAdapterResult(air=None, diagnostics=diagnostics)

                for op in step_ops:
                    operations.append(op)
                    current_dataset = op.output_alias
                    current_schema = dict(op.output_schema)
                    prev_op_id = op.operation_id
                    if op.is_terminal:
                        terminal_ids.append(op.operation_id)

        lifecycle = AIRLifecycleState.PLANNED if (plan and plan.steps) else AIRLifecycleState.EXTRACTED

        air = MasterAIR(
            lifecycle_state=lifecycle,
            raw_user_question=question,
            metadata_summary=f"AIR v2.1 translated from legacy Objective ({objective.analysis_type.value})",
            feasibility_status=FeasibilityStatus.FEASIBLE,
            grain_assessment=grain_assessment,
            dimensions=dimensions,
            operations=operations,
            terminal_operation_ids=terminal_ids,
            requested_outputs=list(objective.required_outputs),
            legacy_context=legacy_context,
        )

        return AIRAdapterResult(air=air, diagnostics=diagnostics)

    @classmethod
    def _authorize_dimensions(
        cls, objective: AnalysisObjective, question: str
    ) -> Tuple[List[DimensionAuthorizationMatrix], Optional[ConversionDiagnostic]]:
        """
        D1 & D2: Deterministic, fail-closed dimension authorization.
        Supports complete compound grouping sets and explicit ranking entity resolution.
        """
        q_lower = question.lower()
        candidate_dims = objective.candidate_dimensions

        # ---------------------------------------------------------------------
        # D2: Deterministic Compound Grouping Extraction
        # ---------------------------------------------------------------------
        explicit_grouping_dims: Set[str] = set()
        m_group = re.search(
            r'\b(?:by|broken down by|grouped by)\s+(.*?)(?:\s+\b(?:in|for|where|with|during|from|at)\b|$|[.;])',
            q_lower,
        )
        if m_group:
            clause = m_group.group(1).strip()
            # Split strictly by comma, 'and', '&'
            tokens = [t.strip() for t in re.split(r'[,&]|\band\b', clause) if t.strip()]
            for tok in tokens:
                clean_tok = tok.replace("_", " ")
                matched = False
                if clean_tok in DIM_SYNONYMS and DIM_SYNONYMS[clean_tok] in candidate_dims:
                    explicit_grouping_dims.add(DIM_SYNONYMS[clean_tok])
                    matched = True
                elif tok in candidate_dims:
                    explicit_grouping_dims.add(tok)
                    matched = True
                else:
                    for c in candidate_dims:
                        if c.replace("_", " ") == clean_tok:
                            explicit_grouping_dims.add(c)
                            matched = True
                            break

        # ---------------------------------------------------------------------
        # D1: Deterministic Ranking Entity Resolution
        # ---------------------------------------------------------------------
        ranking_entity_dim: Optional[str] = None
        is_ranking_question = any(k in q_lower for k in ["highest", "top", "lowest", "bottom", "rank"])
        if is_ranking_question:
            resolved_dim, err = cls._resolve_ranking_entity(q_lower, candidate_dims)
            if err:
                return (
                    [],
                    ConversionDiagnostic(
                        is_fatal=True,
                        diagnostic_reason=err,
                        remediation_hint="Specify the ranking entity explicitly without ambiguity.",
                    ),
                )
            ranking_entity_dim = resolved_dim

        # ---------------------------------------------------------------------
        # Construct DimensionAuthorizationMatrix (Fail-Closed)
        # ---------------------------------------------------------------------
        dims: List[DimensionAuthorizationMatrix] = []
        for dim in candidate_dims:
            is_grouping = dim in explicit_grouping_dims
            is_ranking = (dim == ranking_entity_dim)
            is_requested = is_grouping or is_ranking

            allow_grouping = is_grouping or is_ranking
            allow_order_by = is_ranking
            allow_filter = True  # filtering is permitted on known candidate dimensions

            dims.append(
                DimensionAuthorizationMatrix(
                    column_name=dim,
                    is_requested_in_final_output=is_requested,
                    allow_filter_predicate=allow_filter,
                    allow_grouping=allow_grouping,
                    allow_order_by=allow_order_by,
                    allow_exploratory_drilldown=(objective.analysis_type == AnalysisType.DIAGNOSTIC),
                )
            )

        return dims, None

    @classmethod
    def _resolve_ranking_entity(
        cls, q_lower: str, candidate_dimensions: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        D1: Maps explicit ranking entity phrases to a unique candidate dimension.
        Returns (matched_dimension, error_message).
        """
        for entity_phrase, target_cols in RANKING_ENTITY_MAP:
            if re.search(r'\b' + re.escape(entity_phrase) + r'\b', q_lower):
                matching_dims = [c for c in target_cols if c in candidate_dimensions]
                if len(matching_dims) == 1:
                    return matching_dims[0], None
                elif len(matching_dims) > 1:
                    return None, (
                        f"Ambiguous entity mapping: '{entity_phrase}' matches multiple candidate dimensions: "
                        f"{matching_dims}. Disambiguate by naming the exact column."
                    )
                else:
                    return None, (
                        f"Entity '{entity_phrase}' requested for ranking but no matching column found in candidate dimensions: "
                        f"{candidate_dimensions}."
                    )
        return None, None

    @classmethod
    def _map_plan_step(
        cls,
        step: PlanStep,
        step_index: int,
        current_dataset: str,
        current_schema: Dict[str, DataType],
        prev_op_id: Optional[str],
        is_terminal: bool,
        objective: AnalysisObjective,
        table_columns: List[str],
        table_name: str,
        question: str,
    ) -> Tuple[List[AnalyticalOperation], Optional[ConversionDiagnostic]]:
        desc_lower = step.description.lower()
        q_lower = question.lower()
        agg_raw = (step.agg_func or "").strip().upper()
        op_id = f"op_{step.step_id}"
        out_alias = f"view_{step.step_id}"
        depends_on = [prev_op_id] if prev_op_id else []

        # ---------------------------------------------------------------------
        # 1. Reject Unsupported Operations in Phase 1 (Comparison, Join, Transform)
        # ---------------------------------------------------------------------
        if objective.analysis_type == AnalysisType.COMPARATIVE or any(k in desc_lower for k in ["compare", "growth", "delta", "vs"]):
            return (
                [],
                ConversionDiagnostic(
                    is_fatal=True,
                    step_id=step.step_id,
                    diagnostic_reason="ComparisonOperation is deferred to Phase 3/4. Not supported in Phase 1.",
                    remediation_hint="Use descriptive or aggregation planning.",
                ),
            )

        # ---------------------------------------------------------------------
        # 2. Ranking Path: D3 Semantic Grain Distinctions
        # ---------------------------------------------------------------------
        if any(k in desc_lower for k in ["top", "highest", "lowest", "bottom", "rank"]):
            metric_raw = step.metric_col or "sales"
            metric = resolve_metric_column(metric_raw, table_columns)
            direction = SortDirection.ASC if any(k in desc_lower for k in ["lowest", "bottom"]) else SortDirection.DESC
            m_limit = re.search(r'\b(?:top|bottom)\s+(\d+)\b', desc_lower)
            limit_val = int(m_limit.group(1)) if m_limit else 5

            generated_ops: List[AnalyticalOperation] = []

            # D3: Classify ranking intent
            text_context = (desc_lower + " " + q_lower)
            is_raw_record_ranking = any(
                k in text_context for k in [
                    "individual transaction", "single transaction", "single order line",
                    "individual record", "single sale", "largest transaction", "highest transaction",
                ]
            )

            # Check if ranking target is an entity
            ranking_entity, rank_err = cls._resolve_ranking_entity(q_lower, objective.candidate_dimensions)
            if rank_err and not is_raw_record_ranking:
                return (
                    [],
                    ConversionDiagnostic(
                        is_fatal=True,
                        step_id=step.step_id,
                        diagnostic_reason=rank_err,
                        remediation_hint="Ensure requested ranking entity maps unambiguously to candidate dimensions.",
                    ),
                )

            # Check entity grain aggregation (e.g. order grain from line-item table)
            is_order_grain_ranking = (
                objective.entity_grain == "order" or any(k in text_context for k in ["highest order", "top order", "largest order"])
            ) and ("order_id" in table_columns)

            if is_raw_record_ranking:
                # Case 1: Rank raw transaction records without grouping
                rank_op = RankOperation(
                    operation_id=op_id,
                    description=step.description,
                    depends_on=depends_on,
                    input_datasets=[current_dataset],
                    output_alias=out_alias,
                    output_schema=dict(current_schema),
                    order_by=[
                        SortCriterion(
                            column=ColumnRef(name=metric, data_type=current_schema.get(metric, DataType.FLOAT)),
                            direction=direction,
                            is_metric=True,
                        )
                    ],
                    limit=limit_val,
                    group_by_partitions=[],
                    is_terminal=is_terminal,
                )
                generated_ops.append(rank_op)
                return (generated_ops, None)

            elif is_order_grain_ranking:
                # Case 3: Rank entities after aggregation to requested entity grain (order_id)
                agg_op_id = f"{op_id}_agg"
                agg_alias = f"{out_alias}_agg"
                agg_out_schema = {
                    "order_id": current_schema.get("order_id", DataType.STRING),
                    "total_" + metric: DataType.FLOAT,
                }
                agg_op = AggregateOperation(
                    operation_id=agg_op_id,
                    description=f"Aggregate {metric} by order_id for order ranking",
                    depends_on=depends_on,
                    input_datasets=[current_dataset],
                    output_alias=agg_alias,
                    output_schema=agg_out_schema,
                    group_by_columns=[ColumnRef(name="order_id", data_type=agg_out_schema["order_id"])],
                    measures=[
                        AggregateMeasureSpec(
                            output_alias="total_" + metric,
                            function=AggregationFunction.SUM,
                            target_column=ColumnRef(name=metric, data_type=DataType.FLOAT),
                        )
                    ],
                    is_terminal=False,
                )
                generated_ops.append(agg_op)

                rank_op = RankOperation(
                    operation_id=op_id,
                    description=step.description,
                    depends_on=[agg_op_id],
                    input_datasets=[agg_alias],
                    output_alias=out_alias,
                    output_schema=dict(agg_out_schema),
                    order_by=[
                        SortCriterion(
                            column=ColumnRef(name="total_" + metric, data_type=DataType.FLOAT),
                            direction=direction,
                            is_metric=True,
                        )
                    ],
                    limit=limit_val,
                    group_by_partitions=[],
                    is_terminal=is_terminal,
                )
                generated_ops.append(rank_op)
                return (generated_ops, None)

            elif ranking_entity:
                # Case 2: Rank entity by aggregated metric (e.g. top 5 products by revenue)
                agg_op_id = f"{op_id}_agg"
                agg_alias = f"{out_alias}_agg"
                agg_out_schema = {
                    ranking_entity: current_schema.get(ranking_entity, DataType.STRING),
                    "total_" + metric: DataType.FLOAT,
                }
                agg_op = AggregateOperation(
                    operation_id=agg_op_id,
                    description=f"Aggregate {metric} by {ranking_entity} for ranking",
                    depends_on=depends_on,
                    input_datasets=[current_dataset],
                    output_alias=agg_alias,
                    output_schema=agg_out_schema,
                    group_by_columns=[ColumnRef(name=ranking_entity, data_type=agg_out_schema[ranking_entity])],
                    measures=[
                        AggregateMeasureSpec(
                            output_alias="total_" + metric,
                            function=AggregationFunction.SUM,
                            target_column=ColumnRef(name=metric, data_type=DataType.FLOAT),
                        )
                    ],
                    is_terminal=False,
                )
                generated_ops.append(agg_op)

                rank_op = RankOperation(
                    operation_id=op_id,
                    description=step.description,
                    depends_on=[agg_op_id],
                    input_datasets=[agg_alias],
                    output_alias=out_alias,
                    output_schema=dict(agg_out_schema),
                    order_by=[
                        SortCriterion(
                            column=ColumnRef(name="total_" + metric, data_type=DataType.FLOAT),
                            direction=direction,
                            is_metric=True,
                        )
                    ],
                    limit=limit_val,
                    group_by_partitions=[],
                    is_terminal=is_terminal,
                )
                generated_ops.append(rank_op)
                return (generated_ops, None)
            else:
                return (
                    [],
                    ConversionDiagnostic(
                        is_fatal=True,
                        step_id=step.step_id,
                        diagnostic_reason=f"Cannot reliably determine ranking entity or grain for step: '{step.description}'.",
                        remediation_hint="Specify whether ranking individual rows or an explicit entity.",
                    ),
                )

        # ---------------------------------------------------------------------
        # 3. Filter Operation
        # ---------------------------------------------------------------------
        if agg_raw in {"", "FILTER", "NONE"} and (step.where_clause != "1=1" or "filter" in desc_lower):
            predicates, err = StrictFilterParser.parse_conjunction(step.where_clause, table_columns)
            if err:
                return (
                    [],
                    ConversionDiagnostic(
                        is_fatal=True,
                        step_id=step.step_id,
                        diagnostic_reason=err,
                        remediation_hint="Use standard conjunctive AND with supported operators.",
                    ),
                )
            filter_op = FilterOperation(
                operation_id=op_id,
                description=step.description,
                depends_on=depends_on,
                input_datasets=[current_dataset],
                output_alias=out_alias,
                output_schema=dict(current_schema),
                predicates=predicates,
                is_terminal=is_terminal,
            )
            return ([filter_op], None)

        # ---------------------------------------------------------------------
        # 4. Aggregate Operation (including AOV as SUM / COUNT_DISTINCT / ratio)
        # ---------------------------------------------------------------------
        if agg_raw in {"SUM", "AVG", "COUNT", "COUNT_DISTINCT", "COUNT(DISTINCT)", "DISTINCT_COUNT", "MIN", "MAX", "AOV"}:
            group_cols: List[ColumnRef] = []
            out_schema: Dict[str, DataType] = {}

            if step.group_col:
                col_type = current_schema.get(step.group_col, DataType.STRING)
                group_cols.append(ColumnRef(name=step.group_col, data_type=col_type))
                out_schema[step.group_col] = col_type

            measures: List[AggregateMeasureSpec] = []
            derived: List[DerivedExpressionSpec] = []

            if agg_raw == "AOV":
                sales_col = resolve_metric_column("sales", table_columns)
                order_col = next((c for c in table_columns if "order" in c.lower() and "id" in c.lower()), "order_id")

                measures.append(
                    AggregateMeasureSpec(
                        output_alias="total_sales",
                        function=AggregationFunction.SUM,
                        target_column=ColumnRef(name=sales_col, data_type=DataType.FLOAT),
                    )
                )
                measures.append(
                    AggregateMeasureSpec(
                        output_alias="unique_orders",
                        function=AggregationFunction.COUNT_DISTINCT,
                        target_column=ColumnRef(name=order_col, data_type=DataType.STRING),
                    )
                )
                derived.append(
                    DerivedExpressionSpec(
                        output_alias="aov",
                        expression="total_sales / NULLIF(unique_orders, 0)",
                        referenced_columns=["total_sales", "unique_orders"],
                        null_handling="null_if_zero",
                    )
                )
                out_schema["total_sales"] = DataType.FLOAT
                out_schema["unique_orders"] = DataType.INTEGER
                out_schema["aov"] = DataType.FLOAT
            else:
                metric = resolve_metric_column(step.metric_col, table_columns)
                fn_map = {
                    "SUM": AggregationFunction.SUM,
                    "AVG": AggregationFunction.AVG,
                    "COUNT": AggregationFunction.COUNT,
                    "COUNT_DISTINCT": AggregationFunction.COUNT_DISTINCT,
                    "COUNT(DISTINCT)": AggregationFunction.COUNT_DISTINCT,
                    "DISTINCT_COUNT": AggregationFunction.COUNT_DISTINCT,
                    "MIN": AggregationFunction.MIN,
                    "MAX": AggregationFunction.MAX,
                }
                agg_fn = fn_map[agg_raw]
                out_type = DataType.INTEGER if "COUNT" in agg_raw else DataType.FLOAT
                out_alias_name = metric if not step.group_col else f"{agg_raw.lower()}_{metric}"

                measures.append(
                    AggregateMeasureSpec(
                        output_alias=out_alias_name,
                        function=agg_fn,
                        target_column=ColumnRef(name=metric, data_type=out_type),
                    )
                )
                out_schema[out_alias_name] = out_type

            # Parse step WHERE filter if present
            step_preds: List[Predicate] = []
            if step.where_clause and step.where_clause.strip() not in {"1=1", "1 = 1", "(1=1)", ""}:
                step_preds, err = StrictFilterParser.parse_conjunction(step.where_clause, table_columns)
                if err:
                    return (
                        [],
                        ConversionDiagnostic(
                            is_fatal=True,
                            step_id=step.step_id,
                            diagnostic_reason=f"Step filter parsing error: {err}",
                            remediation_hint="Use standard conjunctive AND expressions.",
                        ),
                    )

            agg_op = AggregateOperation(
                operation_id=op_id,
                description=step.description,
                depends_on=depends_on,
                input_datasets=[current_dataset],
                output_alias=out_alias,
                output_schema=out_schema,
                group_by_columns=group_cols,
                measures=measures,
                derived_measures=derived,
                step_filter_predicates=step_preds,
                is_terminal=is_terminal,
            )
            return ([agg_op], None)

        return (
            [],
            ConversionDiagnostic(
                is_fatal=True,
                step_id=step.step_id,
                diagnostic_reason=f"Unsupported agg_func '{step.agg_func}' in Phase 1.",
                remediation_hint="Use SUM, AVG, COUNT, COUNT_DISTINCT, MIN, MAX, AOV, or RANK.",
            ),
        )

    # -------------------------------------------------------------------------
    # Lossless Exact Legacy Restoration
    # -------------------------------------------------------------------------

    @classmethod
    def to_legacy_objective(cls, air: MasterAIR) -> AnalysisObjective:
        """Restores exact legacy AnalysisObjective from snapshot if available."""
        if air.legacy_context and air.legacy_context.objective_snapshot:
            return air.legacy_context.objective_snapshot.model_copy(deep=True)

        requested_dims = [d.column_name for d in air.dimensions if d.is_requested_in_final_output]
        return AnalysisObjective(
            analysis_type=AnalysisType.DESCRIPTIVE,
            target_table=air.operations[0].input_datasets[0] if air.operations else "sample_superstore",
            target_metric=air.raw_user_question,
            target_metrics=[],
            entity_grain=air.grain_assessment.target_entity,
            required_outputs=[f"Output for {d}" for d in requested_dims] or ["Result"],
            candidate_dimensions=[d.column_name for d in air.dimensions],
            filter_conditions=[],
        )

    @classmethod
    def to_legacy_plan(cls, air: MasterAIR) -> AnalysisPlan:
        """Restores exact legacy AnalysisPlan from snapshot if available."""
        if air.legacy_context and air.legacy_context.plan_snapshot:
            return air.legacy_context.plan_snapshot.model_copy(deep=True)

        steps: List[PlanStep] = []
        for op in air.operations:
            step_id = op.operation_id.replace("op_", "")
            if isinstance(op, AggregateOperation):
                metric = op.measures[0].target_column.name if op.measures else "sales"
                agg_fn = op.measures[0].function.value.upper() if op.measures else "SUM"
                group_col = op.group_by_columns[0].name if op.group_by_columns else None
                steps.append(PlanStep(step_id=step_id, description=op.description, metric_col=metric, agg_func=agg_fn, group_col=group_col, where_clause="1=1"))
            elif isinstance(op, RankOperation):
                metric = op.order_by[0].column.name if op.order_by else "sales"
                steps.append(PlanStep(step_id=step_id, description=op.description, metric_col=metric, agg_func="RANK", group_col=None, where_clause="1=1"))
            elif isinstance(op, FilterOperation):
                steps.append(PlanStep(step_id=step_id, description=op.description, metric_col="", agg_func="FILTER", group_col=None, where_clause="1=1"))

        return AnalysisPlan(
            reasoning=f"Projected from MasterAIR v{air.air_version} ({len(steps)} operations).",
            steps=steps,
        )
