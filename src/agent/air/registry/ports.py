"""
Typed Port Binding Definitions for the AIR Capability Registry.

Phase 4.0 Core Schemas.
"""

from __future__ import annotations
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from src.agent.air.schemas import DataType
from src.agent.air.registry.models import PortCardinality, PortDefinition


class BaseTableRelationBinding(BaseModel):
    """Binds an entire validated base table to a relational input port."""
    model_config = ConfigDict(frozen=True)
    binding_type: Literal["base_table_relation"] = "base_table_relation"
    consumer_port_name: str
    table_name: str


class BaseTableColumnBinding(BaseModel):
    """Binds a specific column of a base table to a column port."""
    model_config = ConfigDict(frozen=True)
    binding_type: Literal["base_table_column"] = "base_table_column"
    consumer_port_name: str
    table_name: str
    column_name: str
    expected_data_type: DataType


class OperationOutputBinding(BaseModel):
    """Binds a typed output port of an upstream operation."""
    model_config = ConfigDict(frozen=True)
    binding_type: Literal["operation_output"] = "operation_output"
    consumer_port_name: str
    producer_operation_id: str
    producer_port_name: str
    column_name: Optional[str] = None


class LiteralParameterBinding(BaseModel):
    """Binds an immutable scalar parameter (e.g. date string, integer limit)."""
    model_config = ConfigDict(frozen=True)
    binding_type: Literal["literal_parameter"] = "literal_parameter"
    consumer_port_name: str
    literal_value: Union[str, int, float, bool]
    literal_data_type: DataType


PortBinding = Union[
    BaseTableRelationBinding,
    BaseTableColumnBinding,
    OperationOutputBinding,
    LiteralParameterBinding,
]


def validate_port_binding(port: PortDefinition, binding: PortBinding) -> bool:
    """Validates that a port binding adheres to the port definition contract."""
    if isinstance(binding, BaseTableColumnBinding):
        if port.cardinality != PortCardinality.COLUMN_VECTOR:
            return False
        return port.accepts_data_type(binding.expected_data_type)
    if isinstance(binding, BaseTableRelationBinding):
        return port.cardinality == PortCardinality.RELATION_TABLE and port.payload_type.is_relation
    return True

