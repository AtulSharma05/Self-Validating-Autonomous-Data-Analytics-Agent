"""AIR v2.1 Concrete Analytical Operations and Expression Validators."""

from __future__ import annotations
import ast
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Union
from pydantic import BaseModel, Field, model_validator

from src.agent.air.schemas import ColumnRef, DataType, Predicate


class AggregationFunction(str, Enum):
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class AggregateMeasureSpec(BaseModel):
    output_alias: str = Field(..., description="Alias for aggregated column")
    function: AggregationFunction
    target_column: ColumnRef


def _is_statically_zero(node: ast.AST) -> bool:
    """Recursively evaluates if an AST node statically resolves to zero."""
    if isinstance(node, ast.Constant):
        return node.value in (0, 0.0, -0.0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_statically_zero(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _is_statically_zero(node.left) and _is_statically_zero(node.right)
        if isinstance(node.op, ast.Mult):
            return _is_statically_zero(node.left) or _is_statically_zero(node.right)
    return False


class DerivedExpressionSpec(BaseModel):
    output_alias: str
    expression: str
    referenced_columns: List[str]
    null_handling: Literal["null_if_zero", "coalesce_zero", "propagate"] = "null_if_zero"

    @model_validator(mode="after")
    def validate_ast(self) -> DerivedExpressionSpec:
        """Validates the derived arithmetic expression using a restricted AST."""
        expr_str = self.expression.strip()
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid derived expression syntax: '{expr_str}': {e}")

        # Allowed AST node types for restricted arithmetic expression evaluation
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Name,
            ast.Constant,
            ast.Call,
            ast.Load,
        )

        referenced_names: Set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(
                    f"Forbidden operation '{type(node).__name__}' in derived expression '{expr_str}'."
                )
            # D5: Statically provable zero divisor check
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if _is_statically_zero(node.right):
                    raise ValueError(f"Statically provable zero divisor in derived expression: '{expr_str}'.")

            if isinstance(node, ast.Call):
                # Only allow NULLIF / nullif calls
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id.upper()
                if func_name not in {"NULLIF"}:
                    raise ValueError(f"Forbidden function call '{func_name}' in derived expression '{expr_str}'.")
            elif isinstance(node, ast.Name):
                referenced_names.add(node.id)

        # Check that referenced names match declared referenced_columns
        declared_set = set(self.referenced_columns)
        # Exclude NULLIF function identifier
        referenced_names.discard("NULLIF")
        referenced_names.discard("nullif")

        missing = referenced_names - declared_set
        if missing:
            raise ValueError(
                f"Derived expression '{expr_str}' references undeclared columns: {missing}. "
                f"Declared referenced_columns: {self.referenced_columns}."
            )

        return self


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class SortCriterion(BaseModel):
    column: ColumnRef
    direction: SortDirection = SortDirection.ASC
    is_metric: bool = Field(
        default=False,
        description="True if column is a computed aggregate/metric, False if it is a dimension entity."
    )


class BaseAnalyticalOperation(BaseModel):
    operation_id: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    input_datasets: List[str] = Field(..., description="Names of tables/views read.")
    output_alias: str = Field(..., description="Name of view produced.")
    output_schema: Dict[str, DataType] = Field(..., description="Exact columns and types produced.")
    is_terminal: bool = False


class FilterOperation(BaseAnalyticalOperation):
    predicates: List[Predicate] = Field(default_factory=list)


class AggregateOperation(BaseAnalyticalOperation):
    group_by_columns: List[ColumnRef] = Field(default_factory=list)
    measures: List[AggregateMeasureSpec] = Field(default_factory=list)
    derived_measures: List[DerivedExpressionSpec] = Field(default_factory=list)
    step_filter_predicates: List[Predicate] = Field(default_factory=list)


class RankOperation(BaseAnalyticalOperation):
    order_by: List[SortCriterion]
    limit: int = Field(default=5, ge=1)
    offset: int = Field(default=0, ge=0)
    group_by_partitions: List[ColumnRef] = Field(default_factory=list)


class DerivedOperation(BaseAnalyticalOperation):
    expressions: List[DerivedExpressionSpec] = Field(default_factory=list)


# Supported operational union
AnalyticalOperation = Union[
    FilterOperation,
    AggregateOperation,
    RankOperation,
    DerivedOperation,
]

