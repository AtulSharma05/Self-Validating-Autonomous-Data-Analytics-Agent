"""
Data models and typed boundary definitions for the AIR Capability Registry.

Phase 4.0 Core Schemas.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from src.agent.air.schemas import ColumnRef, DataType
from src.agent.air.contract import CardinalityExpectation, ContractEngine


class ExecutionMode(str, Enum):
    """
    Authoritative workflow execution mode established by trusted application code.
    Never dictated by LLM-generated output or external requests.
    """
    LEGACY = "legacy"      # Phase 1–3 backward compatibility path
    REGISTRY = "registry"  # Strict capability validation; missing/unregistered IDs fail closed


class CapabilityKind(str, Enum):
    ATOMIC = "atomic"        # Direct physical mapping or primitive analytical transformation
    COMPOSITE = "composite"  # Reusable analytical pattern defined as a logical sub-DAG


class SemanticRole(str, Enum):
    TEMPORAL_BOUNDARY = "temporal_boundary"
    CATEGORICAL_DIMENSION = "categorical_dimension"
    NUMERIC_METRIC = "numeric_metric"
    IDENTIFIABLE_ENTITY_ID = "identifiable_entity_id"
    DERIVED_NUMERATOR = "derived_numerator"
    DERIVED_DENOMINATOR = "derived_denominator"
    RAW_RECORD = "raw_record"


class DataGrain(str, Enum):
    TRANSACTION_LINE_ITEM = "transaction_line_item"
    IDENTIFIABLE_ENTITY = "identifiable_entity"
    GROUPED_DIMENSIONAL = "grouped_dimensional"
    SCALAR_TOTAL = "scalar_total"
    ANY = "any"


class PortCardinality(str, Enum):
    SCALAR = "scalar"                  # Exactly 1 scalar value
    COLUMN_VECTOR = "column_vector"    # 1D column vector
    RELATION_TABLE = "relation_table"  # 2D structured dataset


class PortPayloadType(BaseModel):
    """Explicitly distinguishes relational datasets from scalar/vector primitives."""
    model_config = ConfigDict(frozen=True)

    is_relation: bool = False
    primitive_type: Optional[DataType] = None  # None if is_relation is True
    allowed_primitive_types: Tuple[DataType, ...] = Field(default_factory=tuple)  # Authorized primitive types
    relation_schema: Optional[Tuple[Tuple[str, DataType], ...]] = None  # Frozen schema for relations

    def accepts_data_type(self, data_type: DataType) -> bool:
        """Verifies if the given DataType is authorized by this port payload specification."""
        if self.is_relation:
            return False
        if self.allowed_primitive_types:
            return data_type in self.allowed_primitive_types
        if self.primitive_type is not None:
            return data_type == self.primitive_type
        return False


class PortDefinition(BaseModel):
    """Immutable input or output port specification."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique port name within capability")
    payload_type: PortPayloadType = Field(..., description="Relational or primitive type specification")
    semantic_role: SemanticRole = Field(..., description="Semantic purpose of the port")
    grain: DataGrain = Field(default=DataGrain.ANY, description="Expected or produced data grain")
    cardinality: PortCardinality = Field(..., description="Shape of the port payload")
    is_required: bool = Field(default=True, description="Whether port must be bound")
    allow_nulls: bool = Field(default=False, description="Whether null values are permissible")
    description: str = Field(..., description="Human and validator readable description")

    def accepts_data_type(self, data_type: DataType) -> bool:
        """Verifies if the given DataType is authorized by this port definition."""
        return self.payload_type.accepts_data_type(data_type)


class ResourceConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_timeout_seconds: int = 30
    max_memory_mb: int = 512
    max_output_rows: Optional[int] = 10000


class CapabilityAuthorizationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    requires_grouping_authorization: bool = False
    requires_filtering_authorization: bool = False
    requires_ordering_authorization: bool = False
    requires_zero_divisor_guard: bool = False


class CapabilityManifestItem(BaseModel):
    """Sanitized, read-only manifest item exposed to LLM prompts for capability discovery."""
    model_config = ConfigDict(frozen=True)

    capability_id: str
    version: str
    kind: CapabilityKind
    description: str
    required_input_ports: Tuple[Dict[str, str], ...]
    output_ports: Tuple[Dict[str, str], ...]

