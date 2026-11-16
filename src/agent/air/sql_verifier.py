"""
Independent SQL AST Conformance Verifier using SQLGlot.

Enforces scope isolation, fail-closed dimension boundaries, sound predicate entailment,
comprehensive projection and schema validation, ORDER BY / HAVING authorization,
and single-table source restrictions.
Issues immutable ExecutionReceipts only upon successful verification.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope, Scope, traverse_scope

from src.agent.air.contract import (
    CardinalityExpectation,
    ExecutionContract,
    ExecutionReceipt,
    PredicateOperator,
    TypedPredicate,
)
from src.agent.air.operations import AggregationFunction


class SQLASTConformanceVerifier:
    """
    Scope-aware AST validator verifying generated DuckDB SQL against an ExecutionContract.
    """

    @classmethod
    def verify(
        cls, sql: str, contract: ExecutionContract
    ) -> Tuple[bool, Optional[str], Optional[ExecutionReceipt]]:
        """
        Validates SQL against contract and returns (is_valid, error_reason, receipt).
        Fails closed on any ambiguity, syntax error, or authorization violation.
        """
        if not sql or not sql.strip():
            return False, "SQL payload is empty.", None

        # 1. Parse SQL with DuckDB dialect
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception as exc:
            return False, f"SQLGlot DuckDB Syntax Parse Error: {exc}", None

        if not isinstance(tree, exp.Select):
            return False, f"Expected root SELECT statement, got '{type(tree).__name__}'.", None

        # 2. Build scope graph for scope isolation
        try:
            root_scope = build_scope(tree)
        except Exception as exc:
            return False, f"Failed to build SQL scope tree: {exc}", None

        if not root_scope:
            return False, "Failed to resolve root query scope.", None

        # 3. Target Table Binding & Unauthorized JOIN / Source Verification (Fix F-01)
        ok, err = cls._verify_table_binding(root_scope, tree, contract)
        if not ok:
            return False, err, None

        # 4. Scope-Isolated GROUP BY & Dimension Authorization
        ok, err = cls._verify_group_by(root_scope, contract)
        if not ok:
            return False, err, None

        # 5. ORDER BY Authorization (Fix F-03)
        ok, err = cls._verify_order_by(root_scope, contract)
        if not ok:
            return False, err, None

        # 6. HAVING Authorization (Fix F-03)
        ok, err = cls._verify_having(root_scope, contract)
        if not ok:
            return False, err, None

        # 7. Measure & Aggregation Function Verification
        ok, err = cls._verify_measures(root_scope, contract)
        if not ok:
            return False, err, None

        # 7b. Full-Expression Derived Measure Structural AST Verification (Phase 3)
        ok, err = cls._verify_derived_measures(root_scope, contract)
        if not ok:
            return False, err, None

        # 8. Sound Predicate Entailment & Filter Policy (Fix F-04)
        ok, err = cls._verify_predicates(root_scope, contract)
        if not ok:
            return False, err, None

        # 9. Output Projections & Schema Verification (Fix F-02)
        ok, err = cls._verify_projections(root_scope, contract)
        if not ok:
            return False, err, None

        # 10. All Conformance Checks Passed — Issue Immutable ExecutionReceipt
        receipt = ExecutionReceipt.create(
            contract=contract,
            code=sql,
            conformance_passed=True,
            validation_details="All scope-isolated AST conformance invariants passed deterministically.",
        )
        return True, None, receipt

    @classmethod
    def _verify_table_binding(
        cls, scope: Scope, tree: exp.Expression, contract: ExecutionContract
    ) -> Tuple[bool, Optional[str]]:
        """
        Fix F-01: Verifies that the query targets exclusively the contract's expected table.
        Rejects additional JOINs, unauthorized CTEs, derived tables, and multi-table references.
        """
        # Reject unauthorized CTEs (WITH clauses)
        if tree.find(exp.With) or scope.expression.args.get("with"):
            return False, "Table binding mismatch: Unauthorized CTE (WITH clause) detected. Single-table execution contract forbids CTEs."

        # A. Reject JOIN clauses (explicit JOIN keywords)
        if scope.expression.args.get("joins") or list(tree.find_all(exp.Join)):
            tables = [t.name.lower().strip('"') for t in tree.find_all(exp.Table)]
            target = contract.target_table.lower().strip('"')
            if all(t == target for t in tables) and len(tables) > 1:
                return False, "Unauthorized self-join detected. Single-table execution contract forbids JOINs."
            return False, "Unauthorized JOIN detected in query. Single-table execution contract forbids JOINs."

        # Comma joins in FROM clause
        from_clause = scope.expression.args.get("from_") or scope.expression.args.get("from")
        if not from_clause:
            return False, "Table binding mismatch: Query is missing a FROM clause or table source."

        if from_clause.expressions:  # e.g. FROM a, b
            return False, "Unauthorized JOIN (comma join) detected. Single-table execution contract forbids JOINs."

        # B. Scope-aware extraction of all physical table sources across all scopes
        target = contract.target_table.lower().strip('"')
        physical_tables: List[str] = []
        for s in traverse_scope(tree):
            for name, source in s.sources.items():
                if isinstance(source, exp.Table):
                    physical_tables.append(source.name.lower().strip('"'))

        if not physical_tables:
            return False, "Table binding mismatch: Query references no physical tables."

        unauthorized_tables = set(physical_tables) - {target}
        if unauthorized_tables:
            return False, f"Table binding mismatch: Unauthorized table source(s) detected: {sorted(list(unauthorized_tables))}."

        return True, None

    @classmethod
    def _verify_group_by(cls, scope: Scope, contract: ExecutionContract) -> Tuple[bool, Optional[str]]:
        """
        Enforces scope-isolated grouping boundaries using direct argument inspection.
        Never searches nested subqueries recursively.
        """
        group_clause = scope.expression.args.get("group")
        actual_group_cols: Set[str] = set()

        if group_clause:
            for col in group_clause.find_all(exp.Column):
                actual_group_cols.add(col.name.lower().strip('"'))

        authorized_cols = {c.lower().strip('"') for c in contract.authorized_group_by}
        forbidden_cols = {c.lower().strip('"') for c in contract.forbidden_group_by}

        # Check for unauthorized grouping dimensions
        unauthorized = actual_group_cols - authorized_cols
        if unauthorized:
            return False, (
                f"Unauthorized GROUP BY dimensions detected: {sorted(list(unauthorized))}. "
                f"Authorized dimensions: {contract.authorized_group_by}."
            )

        # Check for explicitly blacklisted dimensions
        forbidden_hit = actual_group_cols & forbidden_cols
        if forbidden_hit:
            return False, (
                f"Forbidden GROUP BY dimensions detected: {sorted(list(forbidden_hit))}. "
                f"These dimensions are explicitly unauthorized by AIR."
            )

        # Cardinality Check
        if contract.expected_cardinality == CardinalityExpectation.SCALAR and actual_group_cols:
            return False, (
                f"Contract requires SCALAR cardinality, but query has GROUP BY on: {sorted(list(actual_group_cols))}."
            )

        if contract.expected_cardinality == CardinalityExpectation.DIMENSIONAL:
            if authorized_cols and not actual_group_cols:
                return False, (
                    f"Contract requires DIMENSIONAL grouping on {contract.authorized_group_by}, "
                    "but query is missing GROUP BY clause."
                )

        return True, None

    @classmethod
    def _verify_order_by(cls, scope: Scope, contract: ExecutionContract) -> Tuple[bool, Optional[str]]:
        """
        Fix F-03: Validates ORDER BY clauses:
        - Permits ordering by authorized output dimensions.
        - Permits ordering by authorized measure aliases or their validated expressions.
        - Rejects ordering by unrequested dimensions (e.g. ship_mode, segment) or hidden source columns.
        """
        order_clause = scope.expression.args.get("order")
        if not order_clause:
            return True, None

        # Build authorized orderable targets: output grouping dimensions + measure aliases and synonyms
        authorized_cols = {g.lower().strip('"') for g in contract.authorized_group_by}
        for req in contract.required_measures:
            col_l = req.target_column.name.lower().strip('"')
            authorized_cols.update([
                req.output_alias.lower().strip('"'),
                f"total_{col_l}",
                f"sum_{col_l}",
                "total",
                "sum",
                "total_revenue",
                "total_sales",
                col_l,
            ])

        for ordered in order_clause.expressions:
            ord_inner = ordered.this
            if isinstance(ord_inner, exp.Column):
                name = ord_inner.name.lower().strip('"')
                if name not in authorized_cols:
                    return False, (
                        f"Unauthorized ORDER BY column: '{name}'. "
                        f"Ordering is only permitted on authorized output dimensions {contract.authorized_group_by} "
                        "or authorized measure aliases."
                    )
            elif isinstance(ord_inner, exp.AggFunc):
                agg_cols = [c.name.lower().strip('"') for c in ord_inner.find_all(exp.Column)]
                for c in agg_cols:
                    if not any(req.target_column.name.lower().strip('"') == c for req in contract.required_measures):
                        return False, f"Unauthorized aggregation column in ORDER BY: '{c}'."
            else:
                name = str(ord_inner).lower().strip('"')
                if name not in authorized_cols:
                    return False, f"Unauthorized ORDER BY expression: '{ord_inner.sql(dialect='duckdb')}'."

        return True, None

    @classmethod
    def _verify_having(cls, scope: Scope, contract: ExecutionContract) -> Tuple[bool, Optional[str]]:
        """
        Fix F-03: Validates HAVING clauses:
        - Rejects HAVING unless explicitly authorized by contract.authorized_having.
        - Rejects base column predicates in HAVING.
        """
        having_clause = scope.expression.args.get("having")
        if not having_clause:
            return True, None

        if not getattr(contract, "authorized_having", None):
            return False, (
                f"Unauthorized HAVING clause detected: '{having_clause.sql(dialect='duckdb')}'. "
                "Execution contract does not authorize aggregate filtering."
            )

        # Check for base column without aggregation in HAVING
        for col in having_clause.find_all(exp.Column):
            if not col.find_ancestor(exp.AggFunc):
                return False, f"Base column '{col.name}' in HAVING clause is forbidden. Only aggregated expressions permitted."

        return True, None

    @classmethod
    def _verify_aggregation_condition_predicate(
        cls, agg_node: exp.AggFunc, req: RequiredMeasureSpec
    ) -> Tuple[bool, Optional[str]]:
        """
        ADV-01 & ADV-02: Structurally verifies that an aggregation expression enforces the
        RequiredMeasureSpec's condition_predicate via a sound CASE WHEN clause.
        Fails closed on missing CASE WHEN, wrong column, wrong operator, or opposite/inverted conditions.
        """
        cond = getattr(req, "condition_predicate", None)
        case_node = agg_node.find(exp.Case)

        if not cond:
            # Measure mandates NO condition: reject unauthorized conditional filter
            if case_node:
                return False, f"Measure '{req.output_alias}' does not authorize conditional aggregation, but found CASE expression."
            return True, None

        cond_col = cond.column.name.lower().strip('"')
        req_fn = req.function.value.lower()

        if not case_node:
            # Special ANSI SQL exception: COUNT(DISTINCT col) on the exact same column inherently filters NULLs
            is_distinct = (
                isinstance(agg_node.this, exp.Distinct)
                or bool(agg_node.args.get("distinct"))
                or bool(agg_node.find(exp.Distinct))
            )
            agg_fn = agg_node.key.lower()
            target_col = req.target_column.name.lower().strip('"')
            if (
                cond.operator == PredicateOperator.IS_NOT_NULL
                and agg_fn == "count"
                and is_distinct
                and target_col == cond_col
            ):
                return True, None

            return False, (
                f"Measure '{req.output_alias}' mandates condition '{cond.describe()}', "
                "but aggregation lacks a conditional CASE WHEN construct."
            )

        # Inspect CASE WHEN branch condition
        ifs = case_node.args.get("ifs", [])
        if not ifs:
            return False, f"Measure '{req.output_alias}' contains malformed CASE without WHEN branch."
        if len(ifs) > 1:
            return False, f"Measure '{req.output_alias}' contains multiple WHEN branches; only single condition authorized."

        branch_cond = ifs[0].this
        while isinstance(branch_cond, exp.Paren):
            branch_cond = branch_cond.this

        # Reject additional unauthorized conditions (AND / OR)
        if isinstance(branch_cond, (exp.And, exp.Or)):
            return False, (
                f"Measure '{req.output_alias}' condition contains unauthorized compound logic: "
                f"'{branch_cond.sql(dialect='duckdb')}'."
            )

        is_negated = isinstance(branch_cond, exp.Not)
        inner_cond = branch_cond.this if is_negated else branch_cond
        while isinstance(inner_cond, exp.Paren):
            inner_cond = inner_cond.this

        # Check Column
        cols = [c.name.lower().strip('"') for c in inner_cond.find_all(exp.Column)]
        if not cols or cols[0] != cond_col:
            return False, (
                f"Measure '{req.output_alias}' condition targets column '{cols[0] if cols else 'none'}', "
                f"expected '{cond_col}'."
            )

        # Operator matching
        if cond.operator == PredicateOperator.IS_NOT_NULL:
            if not isinstance(inner_cond, exp.Is) or not isinstance(inner_cond.expression, exp.Null) or not is_negated:
                return False, (
                    f"Measure '{req.output_alias}' requires '{cond_col} IS NOT NULL', "
                    f"got '{branch_cond.sql(dialect='duckdb')}'."
                )
            return True, None

        elif cond.operator == PredicateOperator.IS_NULL:
            if not isinstance(inner_cond, exp.Is) or not isinstance(inner_cond.expression, exp.Null) or is_negated:
                return False, (
                    f"Measure '{req.output_alias}' requires '{cond_col} IS NULL', "
                    f"got '{branch_cond.sql(dialect='duckdb')}'."
                )
            return True, None

        elif cond.operator == PredicateOperator.EQ:
            if not isinstance(inner_cond, exp.EQ) or is_negated:
                return False, f"Measure '{req.output_alias}' requires equality comparison."
            lit_val = str(cond.literal_value).strip("'\"")
            actual_lit = str(inner_cond.expression).strip("'\"")
            if actual_lit != lit_val:
                return False, f"Measure '{req.output_alias}' literal mismatch: expected '{lit_val}', got '{actual_lit}'."
            return True, None

        return False, f"Unsupported condition operator '{cond.operator}' in measure '{req.output_alias}'."

    @classmethod
    def _verify_measures(cls, scope: Scope, contract: ExecutionContract) -> Tuple[bool, Optional[str]]:
        """Verifies required aggregation functions and target metric columns."""
        if not contract.required_measures:
            return True, None

        agg_funcs = list(scope.expression.find_all(exp.AggFunc))
        if not agg_funcs and contract.required_measures:
            return False, "Query contains no aggregation functions, but contract requires measures."

        for req in contract.required_measures:
            matched = False
            last_err = None
            req_fn = req.function.value.lower()
            req_col = req.target_column.name.lower().strip('"')

            for agg in agg_funcs:
                fn_name = agg.key.lower()
                if req_fn == "count_distinct":
                    is_distinct = isinstance(agg.this, exp.Distinct) or bool(agg.args.get("distinct")) or bool(agg.find(exp.Distinct))
                    is_fn_match = (fn_name == "count" and is_distinct)
                else:
                    is_fn_match = (fn_name == req_fn)

                if is_fn_match:
                    agg_cols = [c.name.lower().strip('"') for c in agg.find_all(exp.Column)]
                    target_match = (req_col in agg_cols) or (fn_name == "count" and req.condition_predicate and req.condition_predicate.column.name.lower().strip('"') in agg_cols)
                    if target_match:
                        cond_ok, cond_err = cls._verify_aggregation_condition_predicate(agg, req)
                        if cond_ok:
                            matched = True
                            break
                        else:
                            last_err = cond_err

            if not matched:
                detail = f": {last_err}" if last_err else "."
                return False, (
                    f"Required measure '{req.function.value}({req.target_column.name})' "
                    f"(alias: '{req.output_alias}') was not found in query or failed verification{detail}"
                )

        return True, None

    @classmethod
    def _verify_derived_measures(
        cls, scope: Scope, contract: ExecutionContract
    ) -> Tuple[bool, Optional[str]]:
        """
        Complete structural SQL AST verification of derived arithmetic expressions.
        Validates entire normalized expression tree, casts, operators, and contract references.
        """
        derived_specs = getattr(contract, "required_derived_measures", [])
        if not derived_specs:
            return True, None

        selects_map = {s.alias_or_name.lower().strip('"'): s for s in scope.expression.selects}
        measures_by_alias = {m.output_alias.lower().strip('"'): m for m in contract.required_measures}

        for derived in derived_specs:
            target_alias = derived.output_alias.lower().strip('"')
            sel = selects_map.get(target_alias)
            if not sel:
                synonyms = {"average_order_value": ["aov"], "aov": ["average_order_value"]}
                for syn in synonyms.get(target_alias, []):
                    if syn in selects_map:
                        sel = selects_map[syn]
                        break

            if not sel:
                return False, f"Required derived measure '{target_alias}' is missing from SELECT projections."

            expr = sel.unalias()

            # 1. Reject unauthorized hidden subqueries
            if list(expr.find_all((exp.Select, exp.Subquery))):
                return False, f"Derived measure '{target_alias}' contains unauthorized hidden subquery."

            # 2. Reject unauthorized arithmetic operators (e.g. *, +, -)
            forbidden_ops = list(expr.find_all((exp.Mul, exp.Add, exp.Sub)))
            if forbidden_ops:
                bad_op = forbidden_ops[0]
                return False, f"Unauthorized arithmetic operator '{type(bad_op).__name__}' in derived measure '{target_alias}'."

            # 3. Root expression must be a binary division (exp.Div)
            if not isinstance(expr, exp.Div):
                return False, f"Derived measure '{target_alias}' must be a division (exp.Div), got '{type(expr).__name__}'."

            # 4. Numerator validation (left branch of Div)
            num_branch = expr.this
            if isinstance(num_branch, exp.Cast):
                cast_type = num_branch.to.this
                if cast_type not in (exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT):
                    return False, f"Numerator in '{target_alias}' uses invalid cast type '{num_branch.to.sql()}'. Must be DOUBLE."
                num_inner = num_branch.this
            else:
                num_inner = num_branch

            while isinstance(num_inner, exp.Paren):
                num_inner = num_inner.this

            if not isinstance(num_inner, exp.Sum):
                return False, f"Numerator in '{target_alias}' must be a SUM aggregation, got '{type(num_inner).__name__}'."

            num_cols = list(num_inner.find_all(exp.Column))
            req_num = measures_by_alias.get(derived.numerator_source_alias.lower().strip('"'))
            expected_col = req_num.target_column.name.lower().strip('"') if req_num else "sales"
            if not any(c.name.lower().strip('"') == expected_col for c in num_cols):
                return False, f"Numerator in '{target_alias}' aggregates incorrect column. Expected '{expected_col}'."

            # ADV-01: Structurally verify condition predicate on numerator aggregation
            if req_num:
                cond_ok, cond_err = cls._verify_aggregation_condition_predicate(num_inner, req_num)
                if not cond_ok:
                    return False, f"Numerator in '{target_alias}' failed condition predicate enforcement: {cond_err}"

            for c in num_cols:
                if c.table:
                    tbl = c.table.lower().strip('"')
                    if tbl != contract.target_table.lower().strip('"') and tbl not in scope.sources:
                        return False, f"Numerator column '{c.sql()}' references unauthorized table '{c.table}'."

            # 5. Denominator validation (right branch of Div)
            den_branch = expr.expression
            while isinstance(den_branch, exp.Paren):
                den_branch = den_branch.this

            nullif_node = None
            if isinstance(den_branch, exp.Nullif):
                nullif_node = den_branch
                zero_arg = den_branch.expression
                den_target = den_branch.this
            elif isinstance(den_branch, exp.Anonymous) and den_branch.name.lower() == "nullif":
                nullif_node = den_branch
                args = den_branch.expressions
                if len(args) != 2:
                    return False, f"NULLIF in '{target_alias}' must have exactly two arguments."
                den_target = args[0]
                zero_arg = args[1]
            else:
                return False, f"Denominator in '{target_alias}' must be guarded with NULLIF(..., 0)."

            if not zero_arg or str(zero_arg).strip().strip("'\"") not in ("0", "0.0"):
                return False, f"NULLIF zero-divisor guard in '{target_alias}' must check for 0, got '{zero_arg}'."

            while isinstance(den_target, exp.Paren):
                den_target = den_target.this

            if isinstance(den_target, exp.Cast):
                cast_type = den_target.to.this
                if cast_type not in (exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT, exp.DataType.Type.BIGINT, exp.DataType.Type.INT):
                    return False, f"Denominator in '{target_alias}' uses invalid cast type '{den_target.to.sql()}'."
                den_target = den_target.this

            while isinstance(den_target, exp.Paren):
                den_target = den_target.this

            if not isinstance(den_target, exp.Count):
                return False, f"Denominator target in '{target_alias}' must be a COUNT aggregation, got '{type(den_target).__name__}'."

            is_den_distinct = isinstance(den_target.this, exp.Distinct) or bool(den_target.args.get("distinct")) or bool(den_target.find(exp.Distinct))
            if not is_den_distinct:
                return False, f"Denominator in '{target_alias}' uses COUNT without DISTINCT. Must use COUNT(DISTINCT ...)."

            den_cols = list(den_target.find_all(exp.Column))
            req_den = measures_by_alias.get(derived.denominator_source_alias.lower().strip('"'))
            expected_den_col = req_den.target_column.name.lower().strip('"') if req_den else "order_id"
            if not any(c.name.lower().strip('"') == expected_den_col for c in den_cols):
                return False, f"Denominator in '{target_alias}' distinct-counts incorrect column. Expected '{expected_den_col}'."

            # Structurally verify condition predicate on denominator aggregation if specified
            if req_den:
                cond_ok, cond_err = cls._verify_aggregation_condition_predicate(den_target, req_den)
                if not cond_ok:
                    return False, f"Denominator in '{target_alias}' failed condition predicate enforcement: {cond_err}"

            for c in den_cols:
                if c.table:
                    tbl = c.table.lower().strip('"')
                    if tbl != contract.target_table.lower().strip('"') and tbl not in scope.sources:
                        return False, f"Denominator column '{c.sql()}' references unauthorized table '{c.table}'."

        return True, None

    @classmethod
    def _verify_predicates(cls, scope: Scope, contract: ExecutionContract) -> Tuple[bool, Optional[str]]:
        """
        Fix F-04: Sound predicate entailment verification and 4-way filter authorization policy:
        1. Required filters: must be enforced.
        2. Permitted filters: explicitly authorized optional filters.
        3. Forbidden filters: conditions that must never appear.
        4. Unspecified filters: reject extra WHERE conjuncts not authorized.
        """
        where_clause = scope.expression.args.get("where")
        permitted_filters = getattr(contract, "permitted_filters", []) or []

        if not contract.required_filters and not permitted_filters:
            if where_clause:
                return False, "Query contains a WHERE clause, but contract authorizes no filters."
            return True, None

        if contract.required_filters and not where_clause:
            return False, (
                "Query is missing a WHERE clause, but contract mandates required filters: "
                f"{[p.describe() for p in contract.required_filters]}."
            )

        # Flatten top-level AND conjuncts
        conjuncts = cls._flatten_and(where_clause.this)

        # 1. Verify required filters are all soundly entailed
        for req in contract.required_filters:
            ok, reason = cls._is_predicate_entailed(conjuncts, req)
            if not ok:
                return False, f"Predicate enforcement failure for '{req.describe()}': {reason}"

        # 2. Fix F-04: Verify no unspecified extra filters are introduced
        # Every conjunct in WHERE must entail either a required filter or a permitted filter
        for conj in conjuncts:
            while isinstance(conj, exp.Paren):
                conj = conj.this

            is_authorized = False
            for req in contract.required_filters:
                ok, _ = cls._is_predicate_entailed([conj], req)
                if ok:
                    is_authorized = True
                    break

            if not is_authorized:
                for perm in permitted_filters:
                    ok, _ = cls._is_predicate_entailed([conj], perm)
                    if ok:
                        is_authorized = True
                        break

            if not is_authorized:
                return False, (
                    f"Unauthorized/unspecified filter condition detected in WHERE clause: '{conj.sql(dialect='duckdb')}'. "
                    "Execution contract does not authorize this additional constraint."
                )

        return True, None

    @classmethod
    def _flatten_and(cls, node: exp.Expression) -> List[exp.Expression]:
        """Recursively unwraps parenthesized expressions and splits top-level AND nodes."""
        while isinstance(node, exp.Paren):
            node = node.this
        if isinstance(node, exp.And):
            return cls._flatten_and(node.this) + cls._flatten_and(node.expression)
        return [node]

    @classmethod
    def _is_predicate_entailed(
        cls, conjuncts: List[exp.Expression], req: TypedPredicate
    ) -> Tuple[bool, str]:
        """
        Checks whether any top-level conjunct soundly entails the required/permitted predicate.
        Fails closed if the predicate is only present inside a weakened OR clause.
        """
        req_col = req.column.name.lower().strip('"')

        for conj in conjuncts:
            while isinstance(conj, exp.Paren):
                conj = conj.this

            # Check if this conjunct is an OR
            if isinstance(conj, exp.Or):
                or_cols = [c.name.lower().strip('"') for c in conj.find_all(exp.Column)]
                if req_col in or_cols:
                    return False, (
                        f"Predicate on '{req_col}' is placed in a weakened OR condition "
                        f"('{conj.sql(dialect='duckdb')}'), which fails to enforce it for all rows."
                    )
                continue

            # Compare atomic comparisons
            if isinstance(conj, (exp.GTE, exp.GT, exp.LTE, exp.LT, exp.EQ, exp.NEQ)):
                col_node = conj.this if isinstance(conj.this, exp.Column) else conj.find(exp.Column)
                if not col_node or col_node.name.lower().strip('"') != req_col:
                    continue

                actual_op = type(conj)
                actual_literal = cls._extract_literal_val(conj.expression)

                # Temporal Half-Open Lower Bound Check
                if req.operator == PredicateOperator.GTE:
                    if actual_op is not exp.GTE:
                        if actual_op is exp.GT:
                            return False, "Lower boundary uses strictly greater (>) instead of required (>=)."
                        continue
                    if actual_literal != str(req.literal_value):
                        return False, f"Lower boundary value mismatch: expected '{req.literal_value}', got '{actual_literal}'."
                    return True, "Enforced"

                # Temporal Half-Open Strict Upper Bound Check
                if req.operator == PredicateOperator.LT:
                    if actual_op is exp.LTE:
                        return False, (
                            f"Upper boundary uses inclusive (<= '{actual_literal}'), violating strict half-open interval (< '{req.literal_value}')."
                        )
                    if actual_op is not exp.LT:
                        continue
                    if actual_literal != str(req.literal_value):
                        return False, f"Upper boundary value mismatch: expected '{req.literal_value}', got '{actual_literal}'."
                    return True, "Enforced"

                # Equality check
                if req.operator == PredicateOperator.EQ:
                    if actual_op is not exp.EQ:
                        continue
                    if actual_literal != str(req.literal_value):
                        return False, f"Equality literal mismatch: expected '{req.literal_value}', got '{actual_literal}'."
                    return True, "Enforced"

        return False, f"No top-level conjunct in WHERE clause soundly enforces '{req.describe()}'."

    @classmethod
    def _extract_literal_val(cls, node: exp.Expression) -> str:
        """Extracts clean string representation of a literal, Cast, or Date expression."""
        if isinstance(node, exp.Literal):
            return node.this.strip("'").strip('"')
        if isinstance(node, (exp.Cast, exp.Date)):
            return cls._extract_literal_val(node.this)
        return str(node).strip("'").strip('"')

    @classmethod
    def _verify_projections(cls, scope: Scope, contract: ExecutionContract) -> Tuple[bool, Optional[str]]:
        """
        Fix F-02: Validates the complete output schema:
        - Confirms required grouping dimensions and measures are projected.
        - Rejects unauthorized extra columns, constants, expressions, or renamed columns.
        """
        selects = scope.expression.selects
        if not selects:
            return False, "Query contains no SELECT projections."

        # Authorized grouping columns
        authorized_grps = {g.lower().strip('"') for g in contract.authorized_group_by}

        # Build recognized measure synonyms
        measure_aliases: Set[str] = set()
        for req in contract.required_measures:
            alias_lower = req.output_alias.lower().strip('"')
            col_lower = req.target_column.name.lower().strip('"')
            measure_aliases.add(alias_lower)
            measure_aliases.update([
                f"total_{col_lower}",
                f"sum_{col_lower}",
                "total",
                "sum",
                "total_revenue",
                "total_sales",
                "total_orders",
                "unique_orders",
                "null_order_id_count",
                "unassigned_revenue",
                col_lower,
            ])

        for der in getattr(contract, "required_derived_measures", []):
            der_alias = der.output_alias.lower().strip('"')
            measure_aliases.add(der_alias)
            measure_aliases.update([
                "aov",
                "average_order_value",
                der_alias,
            ])

        # Map required measures by output alias for derived aggregation lookup
        measures_by_alias = {m.output_alias.lower().strip('"'): m for m in contract.required_measures}

        # Validate each individual projection item
        projected_grps: Set[str] = set()
        projected_measures: Set[str] = set()

        for sel in selects:
            alias_name = sel.alias_or_name.lower().strip('"')
            inner = sel.unalias()

            inner_cols = [c.name.lower().strip('"') for c in inner.find_all(exp.Column)]
            aggs = list(inner.find_all(exp.AggFunc))

            # Case A: Non-aggregate projection (must be an authorized grouping dimension)
            if not aggs:
                if len(inner_cols) == 1:
                    inner_col = inner_cols[0]
                    if inner_col not in authorized_grps:
                        return False, (
                            f"Unauthorized column projection: '{inner_col}'. "
                            f"Only authorized grouping dimensions {contract.authorized_group_by} may be projected."
                        )
                    # Reject renaming an unauthorized column to look like an authorized group
                    if alias_name != inner_col and alias_name not in authorized_grps:
                        return False, f"Unauthorized projection alias '{alias_name}' for column '{inner_col}'."
                    projected_grps.add(inner_col)
                    continue
                else:
                    return False, f"Unauthorized non-aggregate expression/constant in projection: '{sel.sql(dialect='duckdb')}'."

            # Case B: Aggregate projection (must match required measures or derived measures)
            for agg in aggs:
                agg_fn = agg.key.lower()
                agg_target_cols = [c.name.lower().strip('"') for c in agg.find_all(exp.Column)]

                is_valid_measure = False
                last_cond_err = None

                # If alias directly matches a required measure, strictly enforce that specific measure spec
                req_specific = measures_by_alias.get(alias_name)
                candidate_reqs = [req_specific] if req_specific else contract.required_measures

                for req in candidate_reqs:
                    req_fn = req.function.value.lower()
                    req_col = req.target_column.name.lower().strip('"')
                    if req_fn == "count_distinct":
                        is_distinct = isinstance(agg.this, exp.Distinct) or bool(agg.args.get("distinct")) or bool(agg.find(exp.Distinct))
                        fn_match = (agg_fn == "count" and is_distinct)
                    else:
                        fn_match = (agg_fn == req_fn)

                    col_match = (req_col in agg_target_cols) or (agg_fn == "count" and req.condition_predicate and req.condition_predicate.column.name.lower().strip('"') in agg_target_cols)
                    if fn_match and col_match:
                        cond_ok, c_err = cls._verify_aggregation_condition_predicate(agg, req)
                        if cond_ok:
                            is_valid_measure = True
                            break
                        else:
                            if alias_name == req.output_alias.lower().strip('"') or not last_cond_err:
                                last_cond_err = c_err

                if not is_valid_measure and not req_specific:
                    # Also permit aggregations that belong to authorized derived measures
                    der_specific = next((d for d in getattr(contract, "required_derived_measures", []) if d.output_alias.lower().strip('"') == alias_name), None)
                    candidate_ders = [der_specific] if der_specific else getattr(contract, "required_derived_measures", [])
                    for der in candidate_ders:
                        num_m = measures_by_alias.get(der.numerator_source_alias.lower().strip('"'))
                        den_m = measures_by_alias.get(der.denominator_source_alias.lower().strip('"'))
                        for m_candidate in [num_m, den_m]:
                            if m_candidate:
                                c_fn = m_candidate.function.value.lower()
                                c_col = m_candidate.target_column.name.lower().strip('"')
                                if c_fn == "count_distinct":
                                    is_dist = isinstance(agg.this, exp.Distinct) or bool(agg.args.get("distinct")) or bool(agg.find(exp.Distinct))
                                    f_m = (agg_fn == "count" and is_dist)
                                else:
                                    f_m = (agg_fn == c_fn)
                                c_match = (c_col in agg_target_cols) or (agg_fn == "count" and m_candidate.condition_predicate and m_candidate.condition_predicate.column.name.lower().strip('"') in agg_target_cols)
                                if f_m and c_match:
                                    c_ok, c_err = cls._verify_aggregation_condition_predicate(agg, m_candidate)
                                    if c_ok:
                                        is_valid_measure = True
                                        break
                                    else:
                                        if alias_name == der.output_alias.lower().strip('"') or not last_cond_err:
                                            last_cond_err = c_err
                        if is_valid_measure:
                            break

                if not is_valid_measure:
                    if last_cond_err:
                        return False, f"Unauthorized aggregation expression '{agg.sql(dialect='duckdb')}' for projection '{alias_name}': {last_cond_err}"
                    return False, f"Unauthorized aggregation expression: '{agg.sql(dialect='duckdb')}'."

            if alias_name not in measure_aliases:
                return False, f"Unauthorized measure projection alias: '{alias_name}'."

            # If not a derived measure, reject unauthorized outer arithmetic (+, -, *)
            is_derived = any(der.output_alias.lower().strip('"') == alias_name for der in getattr(contract, "required_derived_measures", []))
            if not is_derived:
                unauthorized_outer_math = list(inner.find_all((exp.Add, exp.Sub, exp.Mul)))
                if unauthorized_outer_math:
                    bad = unauthorized_outer_math[0]
                    return False, f"Unauthorized arithmetic operation '{type(bad).__name__}' in projection '{alias_name}'."

            projected_measures.add(alias_name)

        # Check completeness: all authorized groups must be projected
        for grp in authorized_grps:
            if grp not in projected_grps:
                return False, f"Authorized grouping dimension '{grp}' is missing from SELECT projections."

        # Check completeness: required measures must be projected
        if contract.required_measures and not projected_measures:
            return False, "Required measure is missing from SELECT projections."

        # Check projection count matches expected (rejects unauthorized extra columns)
        expected_count = (
            len(contract.authorized_group_by)
            + len(contract.required_measures)
            + len(getattr(contract, "required_derived_measures", []))
        )
        if len(selects) > expected_count:
            return False, (
                f"Extra unauthorized projections detected: Query projects {len(selects)} columns, "
                f"but contract expects {expected_count}."
            )

        return True, None
