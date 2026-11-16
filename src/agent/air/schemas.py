"""AIR v2.1 Primitives, Enums, and Semantic Data Contracts."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from src.agent.schemas import AnalysisObjective, AnalysisPlan


class DataType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"


class ColumnRef(BaseModel):
    name: str = Field(..., description="Column identifier")
    data_type: Optional[DataType] = None
    table_name: Optional[str] = None


class ComparisonOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class Predicate(BaseModel):
    column: ColumnRef
    operator: ComparisonOperator
    literal_value: Union[str, int, float, bool, List[Union[str, int, float]], None] = None


class DimensionAuthorizationMatrix(BaseModel):
    """Enforces boundaries per dimension across operational contexts (Fail-Closed)."""
    column_name: str
    is_requested_in_final_output: bool = False
    allow_filter_predicate: bool = False
    allow_grouping: bool = False
    allow_order_by: bool = False
    allow_exploratory_drilldown: bool = False


class GrainConfidence(str, Enum):
    PROVEN_UNIQUE = "proven_unique"      # distinct == row_count, nulls == 0
    CANDIDATE = "candidate"              # listed in profile candidates, not proven
    INFERRED_MULTI_GRAIN = "multi_grain" # distinct < row_count (line-item grain)
    UNKNOWN = "unknown"                  # no metadata or candidate


class GrainAssessment(BaseModel):
    target_entity: str
    candidate_keys: List[str] = Field(default_factory=list)
    confidence: GrainConfidence = GrainConfidence.UNKNOWN
    cardinality_ratio: float = Field(default=0.0)
    supporting_evidence: str
    requires_entity_rollup: bool = False


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    NEEDS_CLARIFICATION = "needs_clarification"
    MISSING_COLUMNS = "missing_columns"
    UNKNOWN_GRAIN = "unknown_grain"


class AIRLifecycleState(str, Enum):
    EXTRACTED = "extracted"
    PLANNED = "planned"
    VALIDATED = "validated"
    EXECUTABLE = "executable"


class LegacyContext(BaseModel):
    """Deep lossless snapshot of legacy AnalysisObjective and AnalysisPlan."""
    objective_snapshot: AnalysisObjective
    plan_snapshot: Optional[AnalysisPlan] = None


class ConversionDiagnostic(BaseModel):
    is_fatal: bool
    step_id: Optional[str] = None
    diagnostic_reason: str
    remediation_hint: Optional[str] = None

