"""Pydantic data schemas for data profiling, agent objectives, validation, and provenance."""

from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    DESCRIPTIVE = "DESCRIPTIVE"
    DIAGNOSTIC = "DIAGNOSTIC"
    COMPARATIVE = "COMPARATIVE"
    EXPLORATORY = "EXPLORATORY"
    DATA_QUALITY = "DATA_QUALITY"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"


class ColumnProfile(BaseModel):
    name: str = Field(description="Column name")
    data_type: str = Field(description="DuckDB/SQL data type")
    is_numeric: bool = Field(default=False)
    is_temporal: bool = Field(default=False)
    is_categorical: bool = Field(default=False)
    total_count: int = Field(default=0)
    null_count: int = Field(default=0)
    null_percentage: float = Field(default=0.0)
    distinct_count: int = Field(default=0)
    sample_values: List[Any] = Field(default_factory=list)
    # Numerical distributions
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    stddev_value: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    # Temporal distributions
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class TableProfile(BaseModel):
    table_name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    primary_key_candidates: List[str] = Field(default_factory=list)
    temporal_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    session_id: str
    tables: List[TableProfile]
    total_tables: int = Field(default=0)

    def to_compact_llm_context(self) -> str:
        """Formats metadata into a compact, privacy-preserving markdown summary for the LLM prompt."""
        sections = []
        for tbl in self.tables:
            sec = [f"### Table: `{tbl.table_name}` ({tbl.row_count:,} rows, {tbl.column_count} columns)"]
            sec.append("| Column | Type | Null% | Distinct | Sample / Range |")
            sec.append("| :--- | :--- | :--- | :--- | :--- |")
            grain_warnings = []
            for col in tbl.columns:
                if col.is_numeric and col.min_value is not None:
                    summary = f"Range: [{col.min_value:,.2f} to {col.max_value:,.2f}], Mean: {col.mean_value:,.2f}"
                elif col.is_temporal and col.min_date:
                    summary = f"Span: {col.min_date} to {col.max_date}"
                elif col.is_categorical and col.sample_values:
                    sample_str = ", ".join(f"'{s}'" for s in col.sample_values[:5])
                    summary = f"Categories: [{sample_str}]"
                else:
                    summary = f"Sample: {col.sample_values[:3]}"
                
                sec.append(f"| `{col.name}` | {col.data_type} | {col.null_percentage:.1f}% | {col.distinct_count} | {summary} |")

                # Detect line-item grain for entity IDs
                col_lower = col.name.lower()
                if any(k in col_lower for k in ["order", "customer", "product", "transaction", "invoice", "user", "item"]) and col_lower.endswith(("_id", "_no", "_number", "_key", "id")):
                    if col.distinct_count < tbl.row_count:
                        grain_warnings.append(
                            f"- Entity `{col.name}` has {col.distinct_count:,} distinct values across {tbl.row_count:,} rows "
                            f"(table grain is line-item level). To count entities, use `COUNT(DISTINCT \"{col.name}\")`."
                        )

            if grain_warnings:
                sec.append("\n**⚠️ Table Grain & Entity Deduplication Notice:**")
                sec.extend(grain_warnings)
                sec.append(
                    "- Note on Average Order Value (AOV): Compute `SUM(\"sales\") / NULLIF(COUNT(DISTINCT \"order_id\"), 0)`, "
                    "never `AVG(\"sales\")` which computes line-item average."
                )

            sections.append("\n".join(sec))
        return "\n\n".join(sections)


class AnalysisObjective(BaseModel):
    analysis_type: AnalysisType = Field(description="Identified analysis type")
    target_metric: str = Field(description="Primary metric to calculate, e.g. 'SUM(revenue)'")
    target_table: str = Field(description="Primary table containing target metric")
    time_column: Optional[str] = Field(None, description="Time dimension column if time-based")
    test_period: Optional[str] = Field(None, description="Current / test period filter, e.g. '2023-Q3'")
    baseline_period: Optional[str] = Field(None, description="Baseline comparison period, e.g. '2023-Q2'")
    filter_conditions: List[str] = Field(default_factory=list, description="SQL WHERE filters")
    candidate_dimensions: List[str] = Field(default_factory=list, description="Categorical dimensions for RCA drill-down")
    target_metrics: List[str] = Field(default_factory=list, description="All requested metrics when multiple metrics are asked")
    entity_grain: Optional[str] = Field(None, description="Entity grain, e.g. 'order', 'customer', 'product'")
    required_outputs: List[str] = Field(default_factory=list, description="Explicit human-readable list of required outputs")
    missing_columns_explanation: Optional[str] = Field(None, description="Explanation if requested metric/filter cannot be performed due to missing columns")
    is_scenario: bool = Field(default=False, description="Whether this is a counterfactual / scenario simulation question")
    scenario_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for counterfactual scenario")
    quality_check_type: Optional[str] = Field(None, description="Type of data quality check: 'bidirectional_mapping', 'null_check', etc.")
    conflict_entities: List[str] = Field(default_factory=list, description="Entities involved in data quality/conflict checks, e.g. ['product_id', 'product_name']")


class CorrectionAttempt(BaseModel):
    attempt_number: int = Field(description="Sequential attempt counter (1, 2, 3)")
    step_id: str = Field(description="Step being corrected")
    failed_code: str = Field(description="The code that failed execution or validation")
    error_type: str = Field(description="Classified error type")
    error_message: str = Field(description="Descriptive error explanation")
    corrected_code: Optional[str] = Field(default=None, description="Regenerated corrected code")
    validation_passed: bool = Field(default=False, description="Whether corrected code passed validation")
    execution_time_ms: float = Field(default=0.0)


class CalculationVerification(BaseModel):
    metric_name: str
    formula: str
    expected_value: float
    actual_value: float
    discrepancy: float
    is_correct: bool
    feedback: Optional[Dict[str, Any]] = None


class SemanticCoverageReport(BaseModel):
    is_covered: bool = Field(description="True if all requested metrics, filters, and entity grains are verified")
    requested_metrics: List[str] = Field(default_factory=list)
    executed_metrics: List[str] = Field(default_factory=list)
    missing_metrics: List[str] = Field(default_factory=list)
    unauthorized_substitutions: List[Dict[str, str]] = Field(default_factory=list)
    requested_filters: List[str] = Field(default_factory=list)
    applied_filters: List[str] = Field(default_factory=list)
    missing_filters: List[str] = Field(default_factory=list)
    details: str = Field(default="")


class ClaimEvidence(BaseModel):
    claim_id: str
    dataset: str
    claim_text: str = Field(default="", description="Human-readable claim or finding statement")
    metric_name: str = Field(default="", description="Target metric or entity column")
    data_grain: str = Field(default="aggregate", description="Entity grain, e.g. order, line_item, product")
    relevant_filters: List[str] = Field(default_factory=list)
    analytical_operation: str = Field(default="")
    generated_sql: str = Field(default="")
    calculated_result: Any = Field(default=None)
    supporting_statistics: Dict[str, Any] = Field(default_factory=dict)
    checksum_hash: str = Field(default="", description="SHA-256 deterministic checksum hash")
    validation_status: str = Field(default="PASSED", description="PASSED, FAILED, PARTIALLY_VERIFIED, NOT_APPLICABLE")
    supporting_evidence: List[str] = Field(default_factory=list)
    contradictory_evidence: List[str] = Field(default_factory=list)
    related_observations: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Structured serialization ensuring safe rendering on frontend."""
        return {
            "claim_id": self.claim_id,
            "dataset": self.dataset,
            "claim_text": self.claim_text or self.related_observations or self.analytical_operation,
            "metric_name": self.metric_name or self.analytical_operation,
            "data_grain": self.data_grain or "aggregate",
            "relevant_filters": self.relevant_filters,
            "analytical_operation": self.analytical_operation,
            "generated_sql": self.generated_sql,
            "calculated_result": self.calculated_result,
            "supporting_statistics": self.supporting_statistics,
            "checksum_hash": self.checksum_hash or self.supporting_statistics.get("hash", ""),
            "validation_status": self.validation_status,
            "supporting_evidence": self.supporting_evidence,
            "contradictory_evidence": self.contradictory_evidence,
            "related_observations": self.related_observations,
        }


class ValidationResult(BaseModel):
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    audit_details: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    step_id: str = Field(description="Unique identifier, e.g. 'step_1'")
    description: str = Field(description="Clear explanation of what this step computes")
    metric_col: str = Field(default="", description="Target metric column to aggregate/validate")
    agg_func: str = Field(default="SUM", description="Aggregation function (SUM, AVG, COUNT)")
    group_col: Optional[str] = Field(default=None, description="Optional grouping dimension")
    where_clause: str = Field(default="1=1", description="SQL filter condition for this step")
    completed: bool = Field(default=False)
    code: Optional[str] = Field(default=None)
    result_summary: Optional[str] = Field(default=None)


class AnalysisPlan(BaseModel):
    reasoning: str = Field(description="Analytical rationale for this plan")
    steps: List[PlanStep] = Field(default_factory=list)


class FactorAttribution(BaseModel):
    dimension: str = Field(description="Categorical dimension, e.g. 'region'")
    category: str = Field(description="Category value, e.g. 'West'")
    baseline_value: float = Field(default=0.0)
    test_value: float = Field(default=0.0)
    delta: float = Field(default=0.0)
    contribution_pct: float = Field(description="Percentage share of total variance explained")
    is_primary: bool = Field(default=False, description="True if |contribution_pct| >= 10%")


class VolumePriceDecomposition(BaseModel):
    baseline_volume: float
    test_volume: float
    baseline_avg_price: float
    test_avg_price: float
    volume_effect: float
    price_effect: float
    total_delta: float
    volume_contribution_pct: float
    price_contribution_pct: float


class HypothesisResult(BaseModel):
    hypothesis_id: str = Field(description="Identifier e.g. H_1, H_2")
    dimension: str = Field(description="Dimension or factor investigated")
    statement: str = Field(description="Analytical hypothesis statement")
    top_category: str = Field(default="", description="Category leading the effect")
    delta: float = Field(default=0.0, description="Observed change or magnitude")
    contribution_pct: float = Field(default=0.0, description="Percentage of variance explained")
    status: str = Field(description="CONFIRMED_PRIMARY_DRIVER or REJECTED_MINOR_FACTOR")
    evidence_hash: str = Field(default="", description="Audit hash")


class RCAReport(BaseModel):
    target_metric: str
    baseline_total: float
    test_total: float
    total_delta: float
    delta_pct: float
    primary_factors: List[FactorAttribution] = Field(default_factory=list)
    dimensional_factors: Dict[str, List[FactorAttribution]] = Field(default_factory=dict)
    hypotheses: List[HypothesisResult] = Field(default_factory=list)
    volume_price: Optional[VolumePriceDecomposition] = None
    narrative_summary: str = Field(default="")


class EvidenceRecord(BaseModel):
    step_id: str
    action_type: str
    query_sql: Optional[str] = None
    result_summary: str
    checksum_hash: str
    validation_status: str
    execution_time_ms: float


def resolve_metric_column(raw_metric: str, table_columns: List[str]) -> str:
    """
    Safely resolves a metric column name from a raw string or complex expression
    by matching against known table columns. Prevents fragments like 'CASE WHEN SUM'
    and avoids non-numeric ID columns.
    """
    if not table_columns:
        return raw_metric or "sales"

    if not raw_metric:
        non_id = [c for c in table_columns if not c.lower().endswith("_id") and c.lower() != "id"]
        return non_id[0] if non_id else table_columns[0]

    col_map = {c.lower(): c for c in table_columns}
    clean = raw_metric.strip().replace('"', '').replace("'", "").strip()
    if clean.lower() in col_map:
        return col_map[clean.lower()]

    # Synonym mapping: sales <-> revenue <-> amount, profit, quantity, discount
    synonyms = {
        "sales": ["revenue", "sales", "amount", "total_sales", "total_revenue"],
        "revenue": ["sales", "revenue", "amount", "total_sales", "total_revenue"],
        "profit": ["profit", "net_profit", "earnings", "income"],
        "quantity": ["quantity", "units_sold", "units", "qty", "volume"],
        "discount": ["discount", "discount_pct", "discount_percent", "discount_rate"],
    }
    clean_lower = clean.lower()
    if clean_lower in synonyms:
        for syn in synonyms[clean_lower]:
            if syn in col_map:
                return col_map[syn]
            for c_lower, c_orig in col_map.items():
                if syn in c_lower:
                    return c_orig

    # 1. Simple wrapper e.g. SUM(sales) or COUNT(DISTINCT "order_id")
    m = re.search(r'\b(?:SUM|AVG|COUNT|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?["\']?([a-zA-Z0-9_]+)["\']?\s*\)', raw_metric, re.IGNORECASE)
    if m:
        extracted = m.group(1).lower()
        if extracted in col_map:
            return col_map[extracted]
        if extracted in synonyms:
            for syn in synonyms[extracted]:
                if syn in col_map:
                    return col_map[syn]

    # 2. Check THEN clause in CASE statements e.g. THEN sales ELSE 0
    m_then = re.search(r'\bTHEN\s+["\']?([a-zA-Z0-9_]+)["\']?', raw_metric, re.IGNORECASE)
    if m_then:
        extracted = m_then.group(1).lower()
        if extracted in col_map:
            return col_map[extracted]
        if extracted in synonyms:
            for syn in synonyms[extracted]:
                if syn in col_map:
                    return col_map[syn]

    # 3. Search for any known table column mentioned in the expression
    # Prioritize longer column names first (e.g. sub_category over category)
    for c_lower, c_orig in sorted(col_map.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(c_lower) + r'\b', raw_metric.lower()):
            return c_orig

    # 4. Fallback: prioritize common numeric metrics if present
    common_metrics = ["revenue", "sales", "amount", "profit", "units_sold", "quantity", "price", "discount"]
    for cm in common_metrics:
        for c_lower, c_orig in col_map.items():
            if cm == c_lower or cm in c_lower:
                return c_orig

    # Filter out IDs, dates, and timestamps from candidates
    date_or_id = ["_id", "id", "date", "time", "timestamp", "year", "month", "day"]
    cand = [
        c for c in table_columns
        if not any(c.lower().endswith(k) or c.lower() == k for k in date_or_id)
    ]
    if cand:
        return cand[0]

    return table_columns[0]
