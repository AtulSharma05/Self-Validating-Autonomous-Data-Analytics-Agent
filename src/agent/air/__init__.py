"""Analytical Intermediate Representation (AIR v2.1) Package."""

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
    BaseAnalyticalOperation,
    DerivedExpressionSpec,
    FilterOperation,
    RankOperation,
    SortCriterion,
    SortDirection,
)
from src.agent.air.dag import DAGIntegrityValidator, DAGValidationError, MasterAIR
from src.agent.air.temporal import TemporalBoundaryResolver
from src.agent.air.filter_parser import StrictFilterParser
from src.agent.air.adapter import AIRAdapter, AIRAdapterResult
from src.agent.air.contract import (
    CardinalityExpectation,
    ContractEngine,
    ExecutionContract,
    ExecutionReceipt,
    PredicateOperator,
    RequiredMeasureSpec,
    TypedPredicate,
)
from src.agent.air.compiler import AIRContractCompiler, CompilerError
from src.agent.air.sql_verifier import SQLASTConformanceVerifier

__all__ = [
    "AIRLifecycleState",
    "ColumnRef",
    "ComparisonOperator",
    "ConversionDiagnostic",
    "DataType",
    "DimensionAuthorizationMatrix",
    "FeasibilityStatus",
    "GrainAssessment",
    "GrainConfidence",
    "LegacyContext",
    "Predicate",
    "AggregateMeasureSpec",
    "AggregateOperation",
    "AggregationFunction",
    "AnalyticalOperation",
    "BaseAnalyticalOperation",
    "DerivedExpressionSpec",
    "FilterOperation",
    "RankOperation",
    "SortCriterion",
    "SortDirection",
    "DAGIntegrityValidator",
    "DAGValidationError",
    "MasterAIR",
    "TemporalBoundaryResolver",
    "StrictFilterParser",
    "AIRAdapter",
    "AIRAdapterResult",
    "CardinalityExpectation",
    "ContractEngine",
    "ExecutionContract",
    "ExecutionReceipt",
    "PredicateOperator",
    "RequiredMeasureSpec",
    "TypedPredicate",
    "AIRContractCompiler",
    "CompilerError",
    "SQLASTConformanceVerifier",
]
