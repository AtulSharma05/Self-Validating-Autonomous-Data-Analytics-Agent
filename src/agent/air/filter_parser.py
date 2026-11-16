"""Strict SQLGlot-based filter parser for legacy WHERE clauses."""

from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union
import sqlglot
from sqlglot import exp

from src.agent.air.schemas import ColumnRef, ComparisonOperator, Predicate


class StrictFilterParser:
    """
    Parses legacy SQL WHERE clauses into typed Predicates using SQLGlot AST.
    Enforces a strict grammar:
    - Pure conjunctive AND composition
    - Proper handling of string literals containing 'AND' or 'OR' without splitting
    - Atomic parsing of BETWEEN ... AND ...
    - Fail-closed: unsupported boolean operators (OR), subqueries, or functions produce fatal diagnostics.
    """

    @classmethod
    def parse_conjunction(
        cls,
        where_clause: str,
        table_columns: Optional[List[str]] = None,
    ) -> Tuple[List[Predicate], Optional[str]]:
        """
        Parses a SQL WHERE clause fragment into a list of atomic Predicates.

        Returns:
            (predicates, error_message): If error_message is not None, parsing failed fatally.
        """
        text = where_clause.strip()
        if not text or text in {"1=1", "1 = 1", "(1=1)", "( 1=1 )"}:
            return ([], None)

        try:
            parsed = sqlglot.parse_one(f"SELECT * FROM _dummy_table_ WHERE {text}", read="duckdb")
        except Exception as e:
            return ([], f"SQL syntax error in where_clause '{where_clause}': {e}")

        where_node = parsed.args.get("where")
        if not where_node or not where_node.this:
            return ([], None)

        root_expr = where_node.this

        # Invariant: Reject any OR expressions in Phase 1
        if root_expr.find(exp.Or):
            return (
                [],
                f"Unsupported boolean operator 'OR' in where_clause: '{where_clause}'. "
                f"Only conjunctive AND filters are supported in Phase 1.",
            )

        # Invariant: Reject subqueries
        if root_expr.find(exp.Select):
            return ([], f"Unsupported subquery in where_clause: '{where_clause}'.")

        # Decompose conjunctions into atomic leaves
        leaves = cls._decompose_and(root_expr)
        predicates: List[Predicate] = []

        for leaf in leaves:
            pred, err = cls._parse_leaf(leaf)
            if err:
                return ([], f"Unsupported predicate '{leaf.sql()}' in where_clause '{where_clause}': {err}")
            if pred:
                predicates.append(pred)

        return (predicates, None)

    @classmethod
    def _decompose_and(cls, node: exp.Expression) -> List[exp.Expression]:
        """Flattens an AST tree of AND nodes into atomic leaf expressions."""
        if isinstance(node, exp.And):
            return cls._decompose_and(node.left) + cls._decompose_and(node.right)
        if isinstance(node, exp.Paren):
            return cls._decompose_and(node.this)
        return [node]

    @classmethod
    def _parse_leaf(cls, node: exp.Expression) -> Tuple[Optional[Predicate], Optional[str]]:
        # Unwrap outer parentheses if any
        while isinstance(node, exp.Paren):
            node = node.this

        # 1. BETWEEN: col BETWEEN low AND high
        if isinstance(node, exp.Between):
            col_name = cls._extract_col_name(node.this)
            if not col_name:
                return (None, "Invalid column reference in BETWEEN clause.")
            low = cls._extract_literal(node.args.get("low"))
            high = cls._extract_literal(node.args.get("high"))
            return (
                Predicate(
                    column=ColumnRef(name=col_name),
                    operator=ComparisonOperator.BETWEEN,
                    literal_value=[low, high],
                ),
                None,
            )

        # 2. Binary Comparisons: =, !=, <>, >, >=, <, <=
        cmp_map = {
            exp.EQ: ComparisonOperator.EQ,
            exp.NEQ: ComparisonOperator.NEQ,
            exp.GT: ComparisonOperator.GT,
            exp.GTE: ComparisonOperator.GTE,
            exp.LT: ComparisonOperator.LT,
            exp.LTE: ComparisonOperator.LTE,
        }
        for ast_type, op in cmp_map.items():
            if isinstance(node, ast_type):
                col_name = cls._extract_col_name(node.this)
                if not col_name:
                    # Check if right is column and left is literal
                    col_name = cls._extract_col_name(node.expression)
                    if not col_name:
                        return (None, f"No recognized column identifier in comparison '{node.sql()}'.")
                    val = cls._extract_literal(node.this)
                else:
                    val = cls._extract_literal(node.expression)

                return (Predicate(column=ColumnRef(name=col_name), operator=op, literal_value=val), None)

        # 3. NOT wrapping: NOT (IS NULL) -> IS_NOT_NULL; NOT (IN (...)) -> NOT_IN
        if isinstance(node, exp.Not):
            inner = node.this
            while isinstance(inner, exp.Paren):
                inner = inner.this

            if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
                col_name = cls._extract_col_name(inner.this)
                if not col_name:
                    return (None, "Invalid column in IS NOT NULL.")
                return (Predicate(column=ColumnRef(name=col_name), operator=ComparisonOperator.IS_NOT_NULL), None)

            if isinstance(inner, exp.In):
                col_name = cls._extract_col_name(inner.this)
                if not col_name:
                    return (None, "Invalid column in NOT IN.")
                vals = [cls._extract_literal(e) for e in inner.expressions]
                return (Predicate(column=ColumnRef(name=col_name), operator=ComparisonOperator.NOT_IN, literal_value=vals), None)

            return (None, f"Unsupported NOT expression: '{node.sql()}'.")

        # 4. IS NULL
        if isinstance(node, exp.Is):
            col_name = cls._extract_col_name(node.this)
            if not col_name:
                return (None, "Invalid column in IS clause.")
            if isinstance(node.expression, exp.Null):
                return (Predicate(column=ColumnRef(name=col_name), operator=ComparisonOperator.IS_NULL), None)
            return (None, f"Unsupported IS operand: '{node.sql()}'.")

        # 5. IN List
        if isinstance(node, exp.In):
            col_name = cls._extract_col_name(node.this)
            if not col_name:
                return (None, "Invalid column in IN clause.")
            vals = [cls._extract_literal(e) for e in node.expressions]
            op = ComparisonOperator.NOT_IN if node.args.get("is_not") else ComparisonOperator.IN
            return (Predicate(column=ColumnRef(name=col_name), operator=op, literal_value=vals), None)

        return (None, f"Unsupported expression structure: '{node.sql()}'.")

    @classmethod
    def _extract_col_name(cls, node: Optional[exp.Expression]) -> Optional[str]:
        if not node:
            return None
        if isinstance(node, exp.Column):
            return node.name.strip('"')
        if isinstance(node, exp.Literal):
            return None
        if hasattr(node, "name") and node.name:
            return node.name.strip('"')
        return None

    @classmethod
    def _extract_literal(cls, node: Optional[exp.Expression]) -> Any:
        if not node:
            return None
        if isinstance(node, exp.Literal):
            val = node.this
            if node.is_int:
                return int(val)
            if node.is_number:
                return float(val)
            return val
        if isinstance(node, exp.Null):
            return None
        if isinstance(node, exp.Boolean):
            return node.this
        # Fallback to sql string representation
        return node.sql().strip("'\"")

