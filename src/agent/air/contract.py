"""
Engine-Neutral Execution Contract and Immutable Execution Receipt Schemas.

AIR v2.1 Execution Contracts decouple analytical authorization from LLM code generation
and target database execution engines.
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from src.agent.air.schemas import ColumnRef, DataType
from src.agent.air.operations import AggregationFunction


class ContractEngine(str, Enum):
    DUCKDB_SQL = "duckdb_sql"
    PANDAS_PYTHON = "pandas_python"
    HYBRID = "hybrid"


class CardinalityExpectation(str, Enum):
    SCALAR = "scalar"             # Exactly 1 row (scalar aggregate)
    EXACT_N = "exact_n"           # Exactly N rows
    BOUNDED = "bounded"           # At most N rows
    DIMENSIONAL = "dimensional"   # Grouped rows matching distinct combinations


class PredicateOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN_HALF_OPEN = "between_half_open"  # [start, end)


class TypedPredicate(BaseModel):
    """Atomic filter condition on an input column."""
    column: ColumnRef
    operator: PredicateOperator
    literal_value: Union[str, int, float, bool, List[Union[str, int, float]], None] = None
    secondary_literal: Optional[Union[str, int, float, None]] = None
    is_temporal: bool = False

    def describe(self) -> str:
        if self.operator == PredicateOperator.BETWEEN_HALF_OPEN:
            return f"{self.column.name} >= '{self.literal_value}' AND {self.column.name} < '{self.secondary_literal}'"
        op_sym = {
            PredicateOperator.EQ: "=",
            PredicateOperator.NEQ: "!=",
            PredicateOperator.GT: ">",
            PredicateOperator.GTE: ">=",
            PredicateOperator.LT: "<",
            PredicateOperator.LTE: "<=",
            PredicateOperator.IN: "IN",
            PredicateOperator.NOT_IN: "NOT IN",
            PredicateOperator.IS_NULL: "IS NULL",
            PredicateOperator.IS_NOT_NULL: "IS NOT NULL",
        }.get(self.operator, str(self.operator))
        if self.literal_value is not None:
            return f"{self.column.name} {op_sym} '{self.literal_value}'"
        return f"{self.column.name} {op_sym}"


class RequiredMeasureSpec(BaseModel):
    output_alias: str
    function: AggregationFunction
    target_column: ColumnRef
    condition_predicate: Optional[TypedPredicate] = None


class RequiredDerivedMeasureSpec(BaseModel):
    """
    Authoritative specification for derived arithmetic projections.
    Decoupled from hardcoded column names.
    """
    output_alias: str
    numerator_source_alias: str
    denominator_source_alias: str
    expected_operator: Literal["/"] = "/"
    enforce_nullif_zero: bool = True
    target_data_type: DataType = DataType.FLOAT


class ExecutionContract(BaseModel):
    """
    Engine-neutral execution contract compiled from validated AIR.
    Authoritative boundary for code generation and sandbox execution.
    """
    contract_id: str = Field(..., description="Unique contract ID, e.g. 'contract_step_1'")
    execution_unit_id: str = Field(..., description="Execution unit ID, e.g. 'unit_step_1'")
    air_operation_ids: List[str] = Field(default_factory=list, description="AIR operations implemented")
    target_table: str = Field(..., description="Target database table/view name")
    
    # Column authorization boundaries
    authorized_source_columns: List[str] = Field(default_factory=list)
    authorized_group_by: List[str] = Field(default_factory=list, description="Strict whitelist of allowed GROUP BY dimensions")
    forbidden_group_by: List[str] = Field(default_factory=list, description="Explicit blacklist of candidate dimensions forbidden from grouping")
    
    # Filter specifications
    required_filters: List[TypedPredicate] = Field(default_factory=list, description="Filters that MUST be logically enforced")
    permitted_filters: List[TypedPredicate] = Field(default_factory=list, description="Explicitly authorized optional filters")
    forbidden_filters: List[TypedPredicate] = Field(default_factory=list, description="Filters that MUST NOT be present")
    
    # Measures and output schema
    required_measures: List[RequiredMeasureSpec] = Field(default_factory=list)
    required_derived_measures: List[RequiredDerivedMeasureSpec] = Field(default_factory=list)
    logical_operation_mapping: Dict[str, str] = Field(default_factory=dict)
    authorized_having: List[str] = Field(default_factory=list, description="Explicitly authorized HAVING aggregate filters")
    expected_output_columns: Dict[str, DataType] = Field(default_factory=dict)
    expected_cardinality: CardinalityExpectation = CardinalityExpectation.DIMENSIONAL
    
    # Engine & constraints
    target_engine: ContractEngine = ContractEngine.DUCKDB_SQL
    timeout_seconds: int = 30


class ExecutionReceipt(BaseModel):
    """
    Immutable receipt issued ONLY after independent conformance validation passes.
    Presented to the execution sandbox as an execution gatekeeper token.
    """
    model_config = ConfigDict(frozen=True)

    contract_id: str
    execution_unit_id: str
    code_hash: str              # SHA-256 hex digest of stripped code payload
    target_engine: ContractEngine
    conformance_passed: bool
    timestamp: float = Field(default_factory=time.time)
    validation_details: Optional[str] = None

    @classmethod
    def create(
        cls,
        contract: ExecutionContract,
        code: str,
        conformance_passed: bool = True,
        validation_details: Optional[str] = None,
    ) -> ExecutionReceipt:
        code_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
        return cls(
            contract_id=contract.contract_id,
            execution_unit_id=contract.execution_unit_id,
            code_hash=code_hash,
            target_engine=contract.target_engine,
            conformance_passed=conformance_passed,
            timestamp=time.time(),
            validation_details=validation_details,
        )

    def matches(self, code: str, contract: ExecutionContract) -> tuple[bool, Optional[str]]:
        """Verifies if this receipt is valid for the given code and contract."""
        if not self.conformance_passed:
            return False, "ExecutionReceipt indicates conformance verification FAILED."
        if self.contract_id != contract.contract_id:
            return False, f"Receipt contract_id mismatch: expected '{contract.contract_id}', got '{self.contract_id}'."
        if self.execution_unit_id != contract.execution_unit_id:
            return False, f"Receipt execution_unit_id mismatch: expected '{contract.execution_unit_id}', got '{self.execution_unit_id}'."
        if self.target_engine != contract.target_engine:
            return False, f"Receipt target_engine mismatch: expected '{contract.target_engine}', got '{self.target_engine}'."
        
        current_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
        if self.code_hash != current_hash:
            return False, f"Receipt code_hash mismatch: code payload was altered after verification."
        
        return True, None

