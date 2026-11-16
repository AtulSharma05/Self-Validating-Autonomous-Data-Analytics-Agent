"""
CodeGenerator — translates a structured plan step into executable DuckDB SQL
or Python/Pandas code, with SQLGlot pre-validation before the code ever
touches the execution engine.

The LLM receives:
  - The DatasetProfile compact schema (column names, types, ranges)
  - The AnalysisObjective (what we're trying to compute)
  - A specific plan step description (e.g. "Calculate total revenue for Q2 by region")
  - DuckDB SQL dialect rules

The LLM must NOT invent column names. It uses only names from the schema.
Generated SQL is validated by SQLGlot before being returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.objective_extractor import _build_llm
from src.agent.schemas import AnalysisObjective, DatasetProfile
from src.agent.air.contract import ExecutionContract, PredicateOperator
from src.config import settings


# ---------------------------------------------------------------------------
# System prompt for code generation
# ---------------------------------------------------------------------------

_CODEGEN_SYSTEM_PROMPT = """\
You are an expert DuckDB SQL and Pandas code generator for a data analytics pipeline.

## DuckDB SQL Rules (MANDATORY — never violate these)
1. Always double-quote column identifiers: SELECT "revenue", "region" FROM sales
2. Use STRFTIME('%Y-%m', "order_date") for year-month grouping
3. For year filtering in WHERE clauses:
   - For DATE/TIMESTAMP columns: use EXTRACT(YEAR FROM "order_date") = 2017 (or YEAR("order_date") = 2017)
   - Or use standard date range: "order_date" >= '2017-01-01' AND "order_date" <= '2017-12-31'
   - NEVER use "order_date" LIKE '2017%' on DATE columns (triggers DuckDB type error) or on M/D/YYYY strings (starts with month, not year)
   - If date column is VARCHAR: use ("order_date" LIKE '%2017%' OR TRY_CAST("order_date" AS DATE) BETWEEN '2017-01-01' AND '2017-12-31')
4. Entity Deduplication & Table Grain Rules (CRITICAL — NEVER COUNT ROWS INSTEAD OF ENTITIES):
   - In transaction/retail/e-commerce tables, each row represents a single line-item, NOT a whole order.
   - When asked to count orders, unique transactions, customers, or products:
     ALWAYS use `COUNT(DISTINCT "order_id")` or `COUNT(DISTINCT "<entity_col>")`.
     NEVER use `COUNT("order_id")` or `COUNT(*)` to count orders!
5. Average Order Value (AOV) Rule:
   - Average Order Value is defined as: `Total Revenue / Unique Orders`.
   - Correct SQL for AOV:
     `SELECT CAST(SUM("sales") AS DOUBLE) / NULLIF(CAST(COUNT(DISTINCT "order_id") AS DOUBLE), 0) AS aov FROM table WHERE ...`
     Or:
     `SELECT AVG(order_total) FROM (SELECT "order_id", SUM(CAST("sales" AS DOUBLE)) AS order_total FROM table WHERE ... GROUP BY "order_id")`
   - NEVER use `SELECT AVG("sales")` for AOV! That computes average line-item value ($221.38 instead of $434.63).
6. Quarter extraction: CAST(CEIL(CAST(STRFTIME('%m', "order_date") AS INT) / 3.0) AS INT)
7. Protect against division-by-zero: NULLIF(denominator, 0)
8. Never use column aliases in WHERE clauses — repeat the expression
9. Use CAST(... AS DOUBLE) for all numeric aggregations to avoid Decimal type issues
10. Always assign the final result to a variable named: result_df (DataFrame) or result_scalar (number)
11. Use only table and column names from the provided schema — never invent new names
12. For Python code: import pandas as pd and numpy as np are pre-imported; con = DuckDB connection
13. Counterfactual Discount Capping & Scenario Modeling Rules:
   - If a step asks for revised total sales with discount capped at X% (e.g. 20%):
     Use: SUM(CASE WHEN "discount" > 0.20 THEN ("sales" / NULLIF(1.0 - "discount", 0)) * (1.0 - 0.20) ELSE "sales" END)
     NEVER filter using WHERE "discount" <= 0.20 (filtering drops rows instead of capping discounts).
   - If a step asks for revised total profit:
     Recovered revenue flows directly to profit:
     SUM(CASE WHEN "discount" > 0.20 THEN "profit" + (("sales" / NULLIF(1.0 - "discount", 0)) * (1.0 - 0.20) - "sales") ELSE "profit" END)
   - If a step asks for profit margin:
     Calculate: (SUM("profit") / NULLIF(SUM("sales"), 0)) * 100.0

## Output Format
Return EXACTLY one of these two formats — nothing else:

FORMAT A (SQL only):
```sql
SELECT ...
FROM table_name
WHERE ...
GROUP BY ...
ORDER BY ...
```

FORMAT B (Python + SQL hybrid):
```python
# Step description
result_df = con.execute(\"\"\"
    SELECT ...
\"\"\").df()
result_df['new_col'] = result_df['col_a'] / result_df['col_b'].replace(0, float('nan'))
```

Do NOT include any explanation outside the code block.
"""

_CODEGEN_USER_TEMPLATE = """\
## Dataset Schema
{schema}

## Analysis Objective
- Type: {analysis_type}
- Target Metric: {target_metric}
- Table: {target_table}
- Time Column: {time_column}
- Test Period: {test_period}
- Baseline Period: {baseline_period}
- Filters: {filters}
- Dimensions: {dimensions}

{previous_context}

## Plan Step to Implement
{step_description}

Generate the DuckDB SQL or Python code for this step:"""

_CODEGEN_CONTRACT_USER_TEMPLATE = """\
## Dataset Schema
{schema}

## Execution Contract & Analytical Boundaries (MANDATORY ENFORCEMENT)
- Target Table: {target_table}
- Authorized Grouping Dimensions (ONLY group by these): {authorized_group_by}
- Forbidden Dimensions (NEVER group by or introduce these): {forbidden_group_by}
- Required Filters (MUST be strictly applied in WHERE clause): {required_filters}
- Required Measures (MUST be aggregated): {required_measures}
- Expected Output Columns: {expected_columns}
- Cardinality Expectation: {cardinality}

{previous_context}

## Step Description
{step_description}

Generate the DuckDB SQL code strictly adhering to the Execution Contract boundaries above:"""


# ---------------------------------------------------------------------------
# GeneratedCode result
# ---------------------------------------------------------------------------

@dataclass
class GeneratedCode:
    """Result container for a single code generation attempt."""
    sql: Optional[str] = None           # Extracted SQL (if SQL-only format)
    python: Optional[str] = None        # Extracted Python (if Python format)
    explanation: str = ""
    raw_response: str = ""
    is_sql_valid: bool = False          # True if SQLGlot validated successfully
    sql_validation_error: Optional[str] = None
    code_type: str = "unknown"          # "sql" | "python" | "unknown"

    @property
    def executable_code(self) -> str:
        """Returns the primary executable code string."""
        return self.python or self.sql or ""

    @property
    def is_ready(self) -> bool:
        """True if we have executable code that passed SQL validation (if applicable)."""
        if self.code_type == "sql":
            return self.is_sql_valid
        return bool(self.executable_code)


# ---------------------------------------------------------------------------
# SQL Validator (SQLGlot)
# ---------------------------------------------------------------------------

class SQLValidator:
    """
    Uses SQLGlot to parse and validate DuckDB SQL before execution.
    Catches syntax errors client-side — no DuckDB round-trip needed.
    """

    @staticmethod
    def validate(sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL syntax using SQLGlot's DuckDB dialect parser.

        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        try:
            import sqlglot
            expressions = sqlglot.parse(sql, dialect="duckdb", error_level=sqlglot.ErrorLevel.RAISE)
            if not expressions:
                return False, "SQLGlot parsed empty expression — query may be malformed."
            return True, None
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def transpile_to_duckdb(sql: str) -> str:
        """
        Transpile SQL from any dialect into DuckDB-compatible SQL.
        Useful when the LLM writes generic ANSI SQL.
        """
        try:
            import sqlglot
            result = sqlglot.transpile(sql, write="duckdb")
            return result[0] if result else sql
        except Exception:
            return sql  # Return as-is if transpilation fails


# ---------------------------------------------------------------------------
# CodeGenerator
# ---------------------------------------------------------------------------

class CodeGenerator:
    """
    Generates DuckDB SQL or Python code for a specific analysis plan step.

    Usage:
        gen = CodeGenerator()
        code = gen.generate(
            profile=profile,
            objective=objective,
            step_description="Calculate total revenue for Q3 2023 by product category",
        )
        if code.is_ready:
            result = sandbox.execute(code.executable_code)
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = _build_llm(json_mode=False)
        return self._llm

    def generate(
        self,
        profile: DatasetProfile,
        objective: AnalysisObjective,
        step_description: str,
        previous_context: str = "",
    ) -> GeneratedCode:
        """
        Generate executable code for a single analysis step.

        Args:
            profile:          Dataset profile with schema metadata.
            objective:        Structured analysis objective.
            step_description: Plain-English description of what this step must compute.
            previous_context: Context or outputs from previous plan steps.

        Returns:
            GeneratedCode with extracted, validated code.
        """
        schema_context = profile.to_compact_llm_context()
        user_content = _CODEGEN_USER_TEMPLATE.format(
            schema=schema_context,
            analysis_type=objective.analysis_type.value,
            target_metric=objective.target_metric,
            target_table=objective.target_table,
            time_column=objective.time_column or "none",
            test_period=objective.test_period or "none",
            baseline_period=objective.baseline_period or "none",
            filters=", ".join(objective.filter_conditions) or "none",
            dimensions=", ".join(objective.candidate_dimensions) or "none",
            previous_context=f"## Previous Step Results\n{previous_context}" if previous_context else "",
            step_description=step_description,
        )

        messages = [
            SystemMessage(content=_CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = self._parse_response(raw_text)
            return self._enforce_grain_rules(parsed, profile, step_description)
        except Exception:
            # Fallback to deterministic template code when LLM server is unreachable
            return self.create_template_code(profile, objective, step_description, previous_context=previous_context)

    def generate_with_contract(
        self,
        profile: DatasetProfile,
        contract: ExecutionContract,
        step_description: str,
        previous_context: str = "",
    ) -> GeneratedCode:
        """
        Generate DuckDB SQL strictly governed by an authoritative ExecutionContract.
        Informs the LLM of exact authorized grouping dimensions and explicitly forbids
        unauthorized candidate dimensions.
        """
        schema_context = profile.to_compact_llm_context()
        filters_str = "; ".join(p.describe() for p in contract.required_filters) or "none"
        measures_str = "; ".join(
            f"{m.function.value.upper()}({m.target_column.name}) AS {m.output_alias}"
            for m in contract.required_measures
        ) or "none"

        user_content = _CODEGEN_CONTRACT_USER_TEMPLATE.format(
            schema=schema_context,
            target_table=contract.target_table,
            authorized_group_by=", ".join(f'"{g}"' for g in contract.authorized_group_by) or "NONE (Scalar Aggregation)",
            forbidden_group_by=", ".join(f'"{g}"' for g in contract.forbidden_group_by) or "NONE",
            required_filters=filters_str,
            required_measures=measures_str,
            expected_columns=", ".join(contract.expected_output_columns.keys()),
            cardinality=contract.expected_cardinality.value.upper(),
            previous_context=f"## Previous Step Results\n{previous_context}" if previous_context else "",
            step_description=step_description,
        )

        messages = [
            SystemMessage(content=_CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = self._parse_response(raw_text)
            return self._enforce_grain_rules(parsed, profile, step_description)
        except Exception:
            # Fallback to deterministic template code matching contract exactly
            return self.create_contract_template_code(profile, contract, step_description, previous_context=previous_context)

    def create_contract_template_code(
        self,
        profile: DatasetProfile,
        contract: ExecutionContract,
        step_description: str,
        previous_context: str = "",
    ) -> GeneratedCode:
        """
        Rule-based deterministic SQL generation strictly matching an ExecutionContract.
        Ensures 100% offline test reliability and contract conformance.
        """
        select_parts = []
        for grp in contract.authorized_group_by:
            select_parts.append(f'"{grp}"')

        order_alias = None
        for m in contract.required_measures:
            fn_upper = m.function.value.upper()
            col_name = f'"{m.target_column.name}"'
            cond = getattr(m, "condition_predicate", None)
            if cond:
                cond_col = f'"{cond.column.name}"'
                if cond.operator == PredicateOperator.IS_NOT_NULL:
                    if fn_upper in {"COUNT_DISTINCT", "COUNT(DISTINCT)"}:
                        agg_expr = f'COUNT(DISTINCT {col_name})'
                    else:
                        agg_expr = f'CAST({fn_upper}(CASE WHEN {cond_col} IS NOT NULL THEN {col_name} ELSE NULL END) AS DOUBLE)'
                elif cond.operator == PredicateOperator.IS_NULL:
                    if fn_upper == "COUNT":
                        agg_expr = f'CAST(COUNT(CASE WHEN {cond_col} IS NULL THEN 1 ELSE NULL END) AS BIGINT)'
                    else:
                        agg_expr = f'CAST({fn_upper}(CASE WHEN {cond_col} IS NULL THEN {col_name} ELSE NULL END) AS DOUBLE)'
                else:
                    agg_expr = f'CAST({fn_upper}({col_name}) AS DOUBLE)'
            else:
                if fn_upper in {"COUNT_DISTINCT", "COUNT(DISTINCT)"}:
                    agg_expr = f'COUNT(DISTINCT {col_name})'
                else:
                    agg_expr = f'CAST({fn_upper}({col_name}) AS DOUBLE)'
            alias = m.output_alias
            select_parts.append(f'{agg_expr} AS {alias}')
            if not order_alias:
                order_alias = alias

        for der in getattr(contract, "required_derived_measures", []):
            num_req = next((m for m in contract.required_measures if m.output_alias == der.numerator_source_alias), None)
            den_req = next((m for m in contract.required_measures if m.output_alias == der.denominator_source_alias), None)
            
            if num_req and getattr(num_req, "condition_predicate", None) and num_req.condition_predicate.operator == PredicateOperator.IS_NOT_NULL:
                num_expr = f'CAST(SUM(CASE WHEN "{num_req.condition_predicate.column.name}" IS NOT NULL THEN "{num_req.target_column.name}" ELSE NULL END) AS DOUBLE)'
            elif num_req:
                num_expr = f'CAST(SUM("{num_req.target_column.name}") AS DOUBLE)'
            else:
                num_expr = f'CAST(SUM("sales") AS DOUBLE)'

            if den_req:
                den_expr = f'CAST(COUNT(DISTINCT "{den_req.target_column.name}") AS DOUBLE)'
            else:
                den_expr = f'CAST(COUNT(DISTINCT "order_id") AS DOUBLE)'

            der_expr = f'{num_expr} / NULLIF({den_expr}, 0)'
            select_parts.append(f'{der_expr} AS {der.output_alias}')

        where_clauses = []
        for p in contract.required_filters:
            col_name = f'"{p.column.name}"'
            if p.operator == PredicateOperator.GTE:
                where_clauses.append(f"{col_name} >= '{p.literal_value}'")
            elif p.operator == PredicateOperator.GT:
                where_clauses.append(f"{col_name} > '{p.literal_value}'")
            elif p.operator == PredicateOperator.LT:
                where_clauses.append(f"{col_name} < '{p.literal_value}'")
            elif p.operator == PredicateOperator.LTE:
                where_clauses.append(f"{col_name} <= '{p.literal_value}'")
            elif p.operator == PredicateOperator.EQ:
                where_clauses.append(f"{col_name} = '{p.literal_value}'")
            elif p.operator == PredicateOperator.BETWEEN_HALF_OPEN:
                where_clauses.append(f"{col_name} >= '{p.literal_value}' AND {col_name} < '{p.secondary_literal}'")

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        group_str = f"GROUP BY {', '.join(f'\"{g}\"' for g in contract.authorized_group_by)}" if contract.authorized_group_by else ""
        order_str = f"ORDER BY {order_alias} DESC" if order_alias else ""

        query_parts = [
            f"SELECT\n    " + ",\n    ".join(select_parts),
            f"FROM {contract.target_table}",
        ]
        if where_str:
            query_parts.append(where_str)
        if group_str:
            query_parts.append(group_str)
        if order_str:
            query_parts.append(order_str)

        sql = "\n".join(query_parts)
        is_valid, err = SQLValidator.validate(sql)
        return GeneratedCode(
            sql=sql,
            raw_response=f"```sql\n{sql}\n```",
            is_sql_valid=is_valid,
            sql_validation_error=err,
            code_type="sql",
        )

    def _enforce_grain_rules(
        self,
        generated: GeneratedCode,
        profile: DatasetProfile,
        step_description: str,
    ) -> GeneratedCode:
        if not generated.sql:
            return generated

        sql = generated.sql
        desc_lower = step_description.lower()
        tbl = profile.tables[0] if profile.tables else None
        order_col = next((c.name for c in (tbl.columns if tbl else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), None)
        sales_col = next((c.name for c in (tbl.columns if tbl else []) if any(k in c.name.lower() for k in ["sales", "revenue", "amount"])), "sales")

        modified = False
        # 1. AOV Guard: Never calculate AVG(sales) for Average Order Value
        if ("aov" in desc_lower or "average order value" in desc_lower) and order_col:
            if re.search(r'\bAVG\s*\(\s*["\']?' + re.escape(sales_col) + r'["\']?\s*\)', sql, re.IGNORECASE):
                sql = re.sub(
                    r'(?i)\bAVG\s*\(\s*["\']?' + re.escape(sales_col) + r'["\']?\s*\)',
                    f'CAST(SUM("{sales_col}") AS DOUBLE) / NULLIF(CAST(COUNT(DISTINCT "{order_col}") AS DOUBLE), 0)',
                    sql
                )
                modified = True

        # 2. Entity Deduplication Guard: Never COUNT(order_id) without DISTINCT for orders
        if order_col:
            if re.search(r'\bCOUNT\s*\(\s*(?!DISTINCT\b)["\']?' + re.escape(order_col) + r'["\']?\s*\)', sql, re.IGNORECASE):
                sql = re.sub(
                    r'(?i)\bCOUNT\s*\(\s*(?!DISTINCT\b)["\']?' + re.escape(order_col) + r'["\']?\s*\)',
                    f'COUNT(DISTINCT "{order_col}")',
                    sql
                )
                modified = True
            elif ("order" in desc_lower or "transaction" in desc_lower) and any(k in desc_lower for k in ["how many", "count", "unique", "total"]):
                if re.search(r'\bCOUNT\s*\(\s*\*\s*\)', sql, re.IGNORECASE):
                    sql = re.sub(
                        r'(?i)\bCOUNT\s*\(\s*\*\s*\)',
                        f'COUNT(DISTINCT "{order_col}")',
                        sql
                    )
                    modified = True

        # 3. Temporal Year Filter Guard: convert LIKE '2017%' on date columns to CAST(col AS VARCHAR) LIKE '%2017%'
        m_like = re.search(r'([\"\'\w]*date[\"\'\w]*)\s+LIKE\s+[\'"]%?(\d{4})%?[\'"]', sql, re.IGNORECASE)
        if m_like:
            col_target = m_like.group(1).replace('"', '').replace("'", "")
            yr_target = m_like.group(2)
            sql = re.sub(
                r'[\"\'\w]*' + re.escape(col_target) + r'[\"\'\w]*\s+LIKE\s+[\'"]%?\d{4}%?[\'"]',
                f'CAST("{col_target}" AS VARCHAR) LIKE \'%{yr_target}%\'',
                sql,
                flags=re.IGNORECASE
            )
            modified = True

        if modified:
            from src.agent.codegen import SQLValidator
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                python=generated.python,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )
        return generated

    def create_template_code(
        self,
        profile: DatasetProfile,
        objective: AnalysisObjective,
        step_description: str,
        previous_context: str = "",
    ) -> GeneratedCode:
        """Rule-based code generation fallback ensuring 100% offline test reliability."""
        tbl = objective.target_table
        step_desc_lower = step_description.lower()
        t_obj = profile.tables[0] if profile.tables else None
        all_cols = [c.name for c in t_obj.columns] if t_obj else []

        conditions = list(objective.filter_conditions) if objective.filter_conditions else []

        # If previous_context has identified entities (e.g. sub_category in ['Bookcases', 'Tables'])
        if previous_context and any(kw in step_desc_lower for kw in ["those", "filtered", "identified", "selected", "these"]):
            subcat_matches = re.findall(r"'sub_category':\s*'([^']+)'", previous_context)
            if subcat_matches:
                distinct_subcats = sorted(list(set(subcat_matches)))
                subcat_in = ", ".join(repr(s) for s in distinct_subcats)
                conditions.append(f'"sub_category" IN ({subcat_in})')
            else:
                for dim in objective.candidate_dimensions:
                    dim_matches = re.findall(rf"'{dim}':\s*'([^']+)'", previous_context)
                    if dim_matches:
                        distinct_dims = sorted(list(set(dim_matches)))
                        dim_in = ", ".join(repr(s) for s in distinct_dims)
                        conditions.append(f'"{dim}" IN ({dim_in})')
                        break

        where_str = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        step_desc_lower = step_description.lower()

        # 0. Check for Data Quality Mapping Conflict Step
        if "is mapped to multiple" in step_desc_lower or ("having count(distinct" in step_desc_lower and "check if any" in step_desc_lower):
            m_map = re.search(r'check if any\s+([a-zA-Z0-9_]+)\s+is mapped to multiple\s+([a-zA-Z0-9_]+)', step_desc_lower)
            if m_map:
                ent_a, ent_b = m_map.group(1), m_map.group(2)
            else:
                entities = objective.conflict_entities if len(objective.conflict_entities) >= 2 else ["product_id", "product_name"]
                ent_a, ent_b = entities[0], entities[1]

            col_map = {c.lower(): c for c in all_cols}
            a_col = col_map.get(ent_a.lower(), ent_a)
            b_col = col_map.get(ent_b.lower(), ent_b)

            sql = (
                f'SELECT "{a_col}", COUNT(DISTINCT "{b_col}") AS conflict_count\n'
                f'FROM {tbl}{where_str}\n'
                f'GROUP BY "{a_col}"\n'
                f'HAVING COUNT(DISTINCT "{b_col}") > 1\n'
                f'ORDER BY conflict_count DESC'
            )
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # 0.1 Check for Percentage Change Step
        if "percentage change in" in step_desc_lower or "pct change" in step_desc_lower:
            scalars = re.findall(r'returned scalar:\s*([0-9\.\-]+)', previous_context)
            sales_col = next((c.name for c in (t_obj.columns if t_obj else []) if any(k in c.name.lower() for k in ["sales", "revenue", "amount"])), "sales")
            if len(scalars) >= 2:
                v_base, v_test = float(scalars[0]), float(scalars[1])
                pct_val = ((v_test - v_base) / abs(v_base) * 100.0) if v_base != 0 else 0.0
                sql = f"SELECT CAST({pct_val:.4f} AS DOUBLE) AS percentage_change"
            else:
                base_cond = objective.baseline_period or "1=1"
                test_cond = objective.test_period or "1=1"
                sql = (
                    f'SELECT CAST((\n'
                    f'  (SUM(CASE WHEN {test_cond} THEN "{sales_col}" ELSE 0.0 END) - SUM(CASE WHEN {base_cond} THEN "{sales_col}" ELSE 0.0 END))\n'
                    f'  / NULLIF(SUM(CASE WHEN {base_cond} THEN "{sales_col}" ELSE 0.0 END), 0)\n'
                    f') * 100.0 AS DOUBLE) AS percentage_change FROM {tbl}'
                )
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # 1. Check for Counterfactual Discount Capping Scenario Steps
        if "capped at" in step_desc_lower and "discount" in step_desc_lower:
            m_cap = re.search(r'capped at\s+(\d+(?:\.\d+)?)\s*%', step_desc_lower)
            th = (float(m_cap.group(1)) / 100.0) if m_cap else 0.20
            sales_col = next((c.name for c in (t_obj.columns if t_obj else []) if c.name.lower() in ["sales", "revenue", "amount"]), "sales")
            profit_col = next((c.name for c in (t_obj.columns if t_obj else []) if "profit" in c.name.lower()), "profit")
            disc_col = next((c.name for c in (t_obj.columns if t_obj else []) if "discount" in c.name.lower()), "discount")

            if "margin" in step_desc_lower:
                sql = (
                    f'SELECT CAST((\n'
                    f'  SUM(CASE WHEN "{disc_col}" > {th} THEN "{profit_col}" + (("{sales_col}" / NULLIF(1.0 - "{disc_col}", 0)) * (1.0 - {th}) - "{sales_col}") ELSE "{profit_col}" END)\n'
                    f'  /\n'
                    f'  NULLIF(SUM(CASE WHEN "{disc_col}" > {th} THEN ("{sales_col}" / NULLIF(1.0 - "{disc_col}", 0)) * (1.0 - {th}) ELSE "{sales_col}" END), 0)\n'
                    f') * 100.0 AS DOUBLE) AS revised_profit_margin_pct FROM {tbl}{where_str}'
                )
            elif "profit" in step_desc_lower:
                sql = (
                    f'SELECT CAST(SUM(\n'
                    f'  CASE \n'
                    f'    WHEN "{disc_col}" > {th} THEN "{profit_col}" + (("{sales_col}" / NULLIF(1.0 - "{disc_col}", 0)) * (1.0 - {th}) - "{sales_col}")\n'
                    f'    ELSE "{profit_col}" \n'
                    f'  END\n'
                    f') AS DOUBLE) AS revised_total_profit FROM {tbl}{where_str}'
                )
            else:
                sql = (
                    f'SELECT CAST(SUM(\n'
                    f'  CASE \n'
                    f'    WHEN "{disc_col}" > {th} THEN ("{sales_col}" / NULLIF(1.0 - "{disc_col}", 0)) * (1.0 - {th})\n'
                    f'    ELSE "{sales_col}" \n'
                    f'  END\n'
                    f') AS DOUBLE) AS revised_total_sales FROM {tbl}{where_str}'
                )
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # 2. Check for standard Profit Margin Percentage
        if "profit margin" in step_desc_lower or "margin percentage" in step_desc_lower:
            sales_col = next((c.name for c in (t_obj.columns if t_obj else []) if c.name.lower() in ["sales", "revenue", "amount"]), "sales")
            profit_col = next((c.name for c in (t_obj.columns if t_obj else []) if "profit" in c.name.lower()), "profit")
            sql = f'SELECT CAST(SUM("{profit_col}") / NULLIF(SUM("{sales_col}"), 0) * 100.0 AS DOUBLE) AS profit_margin_pct FROM {tbl}{where_str}'
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # 3. Check for Average Order Value (AOV)
        if "average order value" in step_desc_lower or "aov" in step_desc_lower:
            sales_col = next((c.name for c in (t_obj.columns if t_obj else []) if c.name.lower() in ["sales", "revenue", "amount"]), "sales")
            order_col = next((c.name for c in (t_obj.columns if t_obj else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), "order_id")
            sql = f'SELECT CAST(SUM("{sales_col}") AS DOUBLE) / NULLIF(CAST(COUNT(DISTINCT "{order_col}") AS DOUBLE), 0) AS aov FROM {tbl}{where_str}'
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # 4. Check for Entity Count (e.g. unique orders)
        if "count(distinct" in step_desc_lower or "unique order" in step_desc_lower or ("order" in step_desc_lower and "count" in step_desc_lower and "unique" in step_desc_lower):
            t_obj = profile.tables[0] if profile.tables else None
            order_col = next((c.name for c in (t_obj.columns if t_obj else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), "order_id")
            sql = f'SELECT CAST(COUNT(DISTINCT "{order_col}") AS DOUBLE) AS total_orders FROM {tbl}{where_str}'
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=f"```sql\n{sql}\n```",
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # 3. Check if step description mentions a specific numeric column (e.g. sales, profit, quantity)
        t_obj = profile.tables[0] if profile.tables else None
        target_col = None
        if t_obj:
            for c in t_obj.numeric_columns:
                if c.lower() in step_desc_lower:
                    target_col = c
                    break

        agg = "SUM"
        if target_col:
            metric_col = target_col
        else:
            from src.agent.schemas import resolve_metric_column
            metric_col = resolve_metric_column(objective.target_metric, all_cols)
            if "avg" in objective.target_metric.lower():
                agg = "AVG"
            elif "count" in objective.target_metric.lower():
                agg = "COUNT"

        # Clean metric column from distinct or quotes
        metric_col = re.sub(r'(?i)\bdistinct\b', '', metric_col).strip().replace('"', '')
        safe_alias = re.sub(r'[^a-zA-Z0-9_]', '_', metric_col)

        # Check if grouping is mentioned in step description
        group_col = None
        for dim in objective.candidate_dimensions:
            if dim.lower() != metric_col.lower() and (dim.lower() in step_desc_lower or f"by {dim.lower()}" in step_desc_lower or "highest" in step_desc_lower or "top" in step_desc_lower):
                # Only treat as group_col if explicitly intended for breakdown/grouping
                if any(kw in step_desc_lower for kw in ["by", "across", "grouped", "breakdown", "decompose", "per", "highest", "identify", "which"]):
                    group_col = dim
                    break

        if not group_col and any(kw in step_desc_lower for kw in ["highest", "top", "rank", "which"]) and objective.candidate_dimensions:
            group_col = objective.candidate_dimensions[0]

        if group_col:
            limit_clause = " LIMIT 1" if any(k in step_desc_lower for k in ["highest", "top 1", "most", "best"]) else (" LIMIT 10" if "top" in step_desc_lower else "")
            sql = f'SELECT "{group_col}", CAST({agg}("{metric_col}") AS DOUBLE) AS total_{safe_alias} FROM {tbl}{where_str} GROUP BY "{group_col}" ORDER BY total_{safe_alias} DESC{limit_clause}'
        else:
            sql = f'SELECT CAST({agg}("{metric_col}") AS DOUBLE) AS total_{safe_alias} FROM {tbl}{where_str}'

        is_valid, err = SQLValidator.validate(sql)
        return GeneratedCode(
            sql=sql,
            raw_response=f"```sql\n{sql}\n```",
            is_sql_valid=is_valid,
            sql_validation_error=err,
            code_type="sql",
        )

    def generate_correction(
        self,
        profile: DatasetProfile,
        objective: AnalysisObjective,
        step_description: str,
        failed_code: str,
        error_prompt_injection: str,
    ) -> GeneratedCode:
        """
        Regenerate code for a step that previously failed validation or execution.

        Args:
            failed_code:            The code that failed.
            error_prompt_injection: The ClassifiedError.to_prompt_injection() string.

        Returns:
            GeneratedCode with corrected code.
        """
        schema_context = profile.to_compact_llm_context()

        correction_content = f"""{_CODEGEN_USER_TEMPLATE.format(
            schema=schema_context,
            analysis_type=objective.analysis_type.value,
            target_metric=objective.target_metric,
            target_table=objective.target_table,
            time_column=objective.time_column or "none",
            test_period=objective.test_period or "none",
            baseline_period=objective.baseline_period or "none",
            filters=", ".join(objective.filter_conditions) or "none",
            dimensions=", ".join(objective.candidate_dimensions) or "none",
            previous_context="",
            step_description=step_description,
        )}

## Previous Attempt (FAILED — do NOT repeat this)
```
{failed_code}
```

## Failure Analysis
{error_prompt_injection}

Generate a corrected version:"""

        messages = [
            SystemMessage(content=_CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=correction_content),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(raw_text)
        except Exception:
            return self.create_template_code(profile, objective, step_description)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw_text: str) -> GeneratedCode:
        """
        Extracts SQL or Python code from the LLM response.

        Detection priority:
          1. JSON structured response (e.g. {"thoughts": "...", "sql": "SELECT ..."})
          2. ```sql ... ``` block → SQL path
          3. ```python ... ``` block → Python path
          4. Plain SQL starting with SELECT / WITH → SQL path
          5. Any remaining text → treat as Python
        """
        raw_text = raw_text.strip()

        # --- Try JSON structured output first ---
        json_obj = None
        if "{" in raw_text and "}" in raw_text:
            # 1. Try extracting from ```json ... ``` block
            json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if json_block:
                try:
                    import json
                    json_obj = json.loads(json_block.group(1))
                except Exception:
                    pass

            # 2. Try direct JSON parse of the response
            if not json_obj:
                try:
                    import json
                    json_obj = json.loads(raw_text)
                except Exception:
                    pass

            # 3. Try finding outermost { ... }
            if not json_obj:
                outer_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if outer_match:
                    try:
                        import json
                        json_obj = json.loads(outer_match.group(0))
                    except Exception:
                        pass

        # 4. If JSON parsing failed (e.g. unescaped quotes in JSON string), use regex to extract SQL directly
        if not json_obj:
            sql_match = re.search(
                r'\"(?:sql|query|duckdb_sql)\"\s*:\s*\"?(SELECT\b[\s\S]+?)(?:\"?\s*\}|\"?\s*,\s*\"[a-zA-Z_]+\"|$)',
                raw_text,
                re.IGNORECASE,
            )
            if sql_match:
                sql_clean = sql_match.group(1).rstrip('\"').strip()
                is_valid, err = SQLValidator.validate(sql_clean)
                return GeneratedCode(
                    sql=sql_clean,
                    raw_response=raw_text,
                    is_sql_valid=is_valid,
                    sql_validation_error=err,
                    code_type="sql",
                )

        if isinstance(json_obj, dict):
            # Check for SQL field
            for sql_key in ["sql", "query", "sql_query", "duckdb_sql"]:
                val = json_obj.get(sql_key)
                if isinstance(val, str) and val.strip():
                    sql_clean = val.strip()
                    is_valid, err = SQLValidator.validate(sql_clean)
                    return GeneratedCode(
                        sql=sql_clean,
                        raw_response=raw_text,
                        explanation=str(json_obj.get("thoughts", "")),
                        is_sql_valid=is_valid,
                        sql_validation_error=err,
                        code_type="sql",
                    )
            # Check for Python field
            for py_key in ["python", "python_code", "py"]:
                val = json_obj.get(py_key)
                if isinstance(val, str) and val.strip():
                    py_clean = val.strip()
                    inline_sql = self._extract_inline_sql(py_clean)
                    is_valid, err = (True, None)
                    if inline_sql:
                        is_valid, err = SQLValidator.validate(inline_sql)
                    return GeneratedCode(
                        python=py_clean,
                        sql=inline_sql,
                        raw_response=raw_text,
                        explanation=str(json_obj.get("thoughts", "")),
                        is_sql_valid=is_valid,
                        sql_validation_error=err,
                        code_type="python",
                    )

        # --- Try ```sql block ---
        sql_block = re.search(r"```sql\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if sql_block:
            sql = sql_block.group(1).strip()
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=raw_text,
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # --- Try ```python block ---
        py_block = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if py_block:
            python = py_block.group(1).strip()
            # Extract any inline SQL from the Python for validation
            inline_sql = self._extract_inline_sql(python)
            is_valid, err = (True, None)
            if inline_sql:
                is_valid, err = SQLValidator.validate(inline_sql)
            return GeneratedCode(
                python=python,
                sql=inline_sql,
                raw_response=raw_text,
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="python",
            )

        # --- Try bare SQL (no code fence) ---
        if re.match(r"^\s*(SELECT|WITH|INSERT|CREATE)\s", raw_text, re.IGNORECASE):
            sql = raw_text
            is_valid, err = SQLValidator.validate(sql)
            return GeneratedCode(
                sql=sql,
                raw_response=raw_text,
                is_sql_valid=is_valid,
                sql_validation_error=err,
                code_type="sql",
            )

        # --- Fallback: treat as Python ---
        return GeneratedCode(
            python=raw_text,
            raw_response=raw_text,
            code_type="python",
            is_sql_valid=True,  # Can't validate Python with SQLGlot
        )

    @staticmethod
    def _extract_inline_sql(python_code: str) -> Optional[str]:
        """
        Extracts the first SQL string from a Python code snippet.
        Looks for triple-quoted strings containing SELECT / WITH.
        """
        # Match triple-quoted strings
        triple_quoted = re.findall(r'"""(.*?)"""', python_code, re.DOTALL)
        triple_quoted += re.findall(r"'''(.*?)'''", python_code, re.DOTALL)
        for block in triple_quoted:
            stripped = block.strip()
            if re.match(r"^\s*(SELECT|WITH)", stripped, re.IGNORECASE):
                return stripped
        return None


def create_contract_template_code(contract: ExecutionContract) -> str:
    """Convenience module-level helper to generate deterministic contract SQL."""
    gen = CodeGenerator()
    return gen.create_contract_template_code(None, contract, "").executable_code


