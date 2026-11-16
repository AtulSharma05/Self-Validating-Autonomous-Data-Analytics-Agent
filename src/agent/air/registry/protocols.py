"""
Typed Protocol Definitions for AIR Capability Registry Handlers.

Phase 4.0 Core Protocols.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp

from src.agent.air.schemas import DataType, DimensionAuthorizationMatrix
from src.agent.air.operations import AnalyticalOperation
from src.agent.air.contract import (
    CardinalityExpectation,
    RequiredDerivedMeasureSpec,
    RequiredMeasureSpec,
    TypedPredicate,
)
from src.validation.validator import ValidationResult


class ContractFragment(BaseModel):
    """
    Typed, immutable fragment emitted by a capability compiler handler.
    Represents an isolated operation's contribution to an ExecutionContract.
    """
    model_config = ConfigDict(frozen=True)

    operation_id: str
    capability_id: str
    capability_version: str
    source_table: str
    authorized_source_columns: Tuple[str, ...] = Field(default_factory=tuple)
    authorized_group_by: Tuple[str, ...] = Field(default_factory=tuple)
    required_filters: Tuple[TypedPredicate, ...] = Field(default_factory=tuple)
    permitted_filters: Tuple[TypedPredicate, ...] = Field(default_factory=tuple)
    required_measures: Tuple[RequiredMeasureSpec, ...] = Field(default_factory=tuple)
    required_derived_measures: Tuple[RequiredDerivedMeasureSpec, ...] = Field(default_factory=tuple)
    authorized_having: Tuple[str, ...] = Field(default_factory=tuple)
    expected_output_columns: Tuple[Tuple[str, DataType], ...] = Field(default_factory=tuple)
    cardinality: CardinalityExpectation = CardinalityExpectation.DIMENSIONAL
    logical_dependencies: Tuple[str, ...] = Field(default_factory=tuple)


@runtime_checkable
class CapabilityCompilerHandler(Protocol):
    """Compiles a validated operation into a ContractFragment."""
    def compile_fragment(
        self,
        operation: AnalyticalOperation,
        dimensions: Dict[str, DimensionAuthorizationMatrix],
        base_schemas: Dict[str, Dict[str, DataType]],
    ) -> ContractFragment: ...


@runtime_checkable
class CapabilityVerifierHandler(Protocol):
    """Operation-specific SQL AST invariant checker (leaf-level checks)."""
    def verify_ast_fragment(
        self,
        scope_expression: exp.Expression,
        fragment: ContractFragment,
    ) -> Tuple[bool, Optional[str]]: ...


@runtime_checkable
class IndependentValidatorHandler(Protocol):
    """Computes expected metrics in independent Pandas execution on raw df_slice."""
    def validate_metric(
        self,
        operation: AnalyticalOperation,
        df_slice: pd.DataFrame,
        sql_actual_value: Any,
    ) -> ValidationResult: ...


@runtime_checkable
class PrerequisiteEvaluator(Protocol):
    """Evaluates whether upstream DAG state satisfies capability prerequisites."""
    def evaluate_prerequisite(
        self,
        operation: AnalyticalOperation,
        upstream_operations: List[AnalyticalOperation],
        base_schemas: Dict[str, Dict[str, DataType]],
    ) -> Tuple[bool, Optional[str]]: ...

