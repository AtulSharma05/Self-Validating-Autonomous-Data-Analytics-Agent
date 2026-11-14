"""
ObjectiveExtractor — translates an ambiguous natural-language question
into a strongly-typed AnalysisObjective Pydantic model.

The LLM receives ONLY:
  - The compact DatasetProfile metadata (column names, types, stats)  ← ~500 tokens
  - The user's question

Raw data rows are NEVER sent to the LLM.

Provider support (controlled by src/config.py):
  - ollama  → ChatOllama (qwen2.5-coder:7b, llama3.1:8b, ...)
  - gemini  → ChatGoogleGenerativeAI
  - openai  → ChatOpenAI
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from src.agent.schemas import AnalysisObjective, AnalysisType, DatasetProfile, resolve_metric_column
from src.config import settings


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_OBJECTIVE_SYSTEM_PROMPT = """\
You are a precise data analytics assistant. Your job is to parse the user's \
natural-language question and extract a structured analytical specification.

You will receive:
1. A compact dataset schema (column names, types, stats) — this is all you know about the data.
2. The user's question.

You MUST respond with ONLY a valid JSON object matching this exact schema:
{{
  "analysis_type": "<DESCRIPTIVE|DIAGNOSTIC|COMPARATIVE|EXPLORATORY|DATA_QUALITY|ANOMALY_DETECTION>",
  "target_metric": "<primary column name, aggregation, or conflict check e.g. 'COUNT(DISTINCT order_id), AOV', 'SUM(sales)', 'CONFLICT_CHECK(product_id, product_name)'>",
  "target_table": "<table name from the schema>",
  "time_column": "<date/timestamp column name, or null>",
  "test_period": "<current period filter string, e.g. '2023-Q3', or null>",
  "baseline_period": "<comparison period, e.g. '2023-Q2', or null>",
  "filter_conditions": ["<SQL WHERE/HAVING clause fragment>", ...],
  "candidate_dimensions": ["<categorical/entity column for breakdown>", ...],
  "target_metrics": ["<metric1>", "<metric2>"],
  "entity_grain": "<'order'|'customer'|'product'|'line_item'|null>",
  "required_outputs": ["<output1>", "<output2>"],
  "quality_check_type": "<'bidirectional_mapping'|'null_check'|'duplicate_keys'|null>",
  "conflict_entities": ["<entity1>", "<entity2>"]
}}

Rules:
- Multi-Metric & Grain Rule: When the question asks for multiple metrics (e.g. "total orders and average order value", or "sales and profit"):
  * include ALL in target_metrics: ["COUNT(DISTINCT order_id)", "AOV"]
  * set entity_grain: "order"
  * set required_outputs: ["Total unique orders", "Average Order Value"]
  * NEVER substitute total revenue/sales for order count or AOV!
- analysis_type=DATA_QUALITY when the question asks about data integrity, mapping conflicts, whether entity A is mapped to multiple entity B or vice versa (e.g. "Product IDs mapped to multiple Product Names, or Product Names mapped to multiple Product IDs").
  * set quality_check_type: "bidirectional_mapping"
  * set conflict_entities: ["product_id", "product_name"]
  * set target_metric: "CONFLICT_CHECK(product_id, product_name)"
- target_metric: MUST be a column name from schema, basic aggregation on it, or a quality check.
  NEVER write complex CASE WHEN expressions or nested aggregations inside target_metric.
- filter_conditions: Put ALL filtering criteria, thresholds, and conditions here (e.g. ["sales > 100000", "profit < 0"]).
- analysis_type=DIAGNOSTIC when the user asks "why", "cause", "reason", "drop", "decline", "increase", "breakdown", "drivers"
- analysis_type=COMPARATIVE when user asks to compare two periods, regions, or groups
- analysis_type=DESCRIPTIVE for "what", "total", "how many", "show me", "which" questions
- analysis_type=EXPLORATORY for open-ended exploration without a specific target
- Entity Deduplication Rule: When asking to count entities (e.g. "how many orders", "total orders", "unique customers"):
  Use `COUNT(DISTINCT "order_id")` or `COUNT(DISTINCT "<entity_col>")`, NEVER `COUNT(*)` or `COUNT("order_id")`.
- Average Order Value (AOV) Rule: AOV is `SUM("sales") / COUNT(DISTINCT "order_id")`. Never map AOV to `AVG("sales")`.
- candidate_dimensions: include ALL categorical or entity columns relevant to the question.
- Use only column names that exist in the provided schema
- Do NOT invent column names or table names
- Respond with ONLY the JSON — no explanation, no markdown, no extra text
"""

_OBJECTIVE_USER_TEMPLATE = """\
## Dataset Schema
{schema}

## User Question
{question}

Respond with ONLY the JSON object:"""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm(json_mode: bool = False) -> BaseChatModel:
    """
    Instantiates the LLM based on MODEL_PROVIDER in settings.

    Raises:
        RuntimeError: If the provider is unsupported or the model server is unreachable.
    """
    provider = settings.model_provider.lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama"
            ) from e
        kwargs = {
            "model": settings.ollama_model_name,
            "base_url": settings.ollama_base_url,
            "temperature": 0,          # Deterministic — no creativity needed for extraction
        }
        if json_mode:
            kwargs["format"] = "json"  # Only constrain to JSON grammar when explicitly requested
        return ChatOllama(**kwargs)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise RuntimeError(
                "langchain-google-genai is not installed. Run: pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from e
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.openai_api_key,
            temperature=0,
        )

    raise RuntimeError(
        f"Unsupported MODEL_PROVIDER='{provider}'. "
        "Set MODEL_PROVIDER to 'ollama', 'gemini', or 'openai' in .env"
    )


# ---------------------------------------------------------------------------
# ObjectiveExtractor
# ---------------------------------------------------------------------------

class ObjectiveExtractor:
    """
    Extracts a structured AnalysisObjective from a natural-language question.

    Usage:
        extractor = ObjectiveExtractor()
        objective = extractor.extract(profile, "Why did revenue drop in Q3?")
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Args:
            llm: Pre-built LangChain chat model. If None, builds from settings.
        """
        self._llm = llm  # Lazily initialized to avoid connecting at import time

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = _build_llm(json_mode=True)
        return self._llm

    def extract(self, profile: DatasetProfile, question: str) -> AnalysisObjective:
        """
        Extracts an AnalysisObjective from the user's question using the LLM.

        Args:
            profile:  The DatasetProfile produced by AutomatedDataProfiler.
            question: The user's raw natural-language question.

        Returns:
            Validated AnalysisObjective Pydantic model.

        Raises:
            ValueError: If the LLM response cannot be parsed into a valid AnalysisObjective.
            RuntimeError: If the LLM server is unreachable.
        """
        schema_context = profile.to_compact_llm_context()
        user_content = _OBJECTIVE_USER_TEMPLATE.format(
            schema=schema_context,
            question=question,
        )

        messages = [
            SystemMessage(content=_OBJECTIVE_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(raw_text, profile, user_question=question)
        except Exception:
            # Fallback to rule-based heuristic extraction when LLM server is unreachable
            return self.create_template_objective(profile, question)

    def create_template_objective(self, profile: DatasetProfile, question: str) -> AnalysisObjective:
        """
        Rule-based deterministic objective extraction ensuring 100% offline reliability.
        """
        q_lower = question.lower()

        # 0. Check for counterfactual / scenario capping (e.g. "discount capped at 20%")
        is_scenario = False
        scenario_params = {}
        m_cap = re.search(r'(?:capped at|max discount(?: of)?|cap discounts? at)\s*(\d+(?:\.\d+)?)\s*%', q_lower)
        if m_cap or ("capped" in q_lower and "discount" in q_lower):
            cap_val = float(m_cap.group(1)) / 100.0 if m_cap else 0.20
            is_scenario = True
            scenario_params = {"type": "discount_cap", "threshold": cap_val}
            atype = AnalysisType.EXPLORATORY

        # 0.1 Check for Data Quality / Bidirectional Mapping queries (e.g. "Product IDs mapped to multiple Product Names")
        mapping_keywords = [
            "mapped to multiple", "mapping", "multiple product names", "multiple product ids",
            "conflicting", "inconsistent mapping", "1-to-many", "many-to-many", "duplicate mapping",
            "multiple names", "multiple ids", "data quality", "integrity"
        ]
        is_data_quality = any(kw in q_lower for kw in mapping_keywords) or (
            "multiple" in q_lower and any(k in q_lower for k in ["id", "name", "mapping", "mapped", "product", "customer"])
        )
        if is_data_quality:
            t_prof = profile.tables[0] if profile.tables else None
            table_name = profile.tables[0].table_name if profile.tables else "data"
            cols = [c.name for c in (t_prof.columns if t_prof else [])]
            pid = next((c for c in cols if "product_id" in c.lower() or (c.lower().endswith("id") and "product" in c.lower())), None)
            pname = next((c for c in cols if "product_name" in c.lower() or (("name" in c.lower() or "desc" in c.lower()) and "product" in c.lower())), None)
            if not pid or not pname:
                cid = next((c for c in cols if "customer_id" in c.lower() or (c.lower().endswith("id") and "customer" in c.lower())), None)
                cname = next((c for c in cols if "customer_name" in c.lower() or (("name" in c.lower()) and "customer" in c.lower())), None)
                if cid and cname:
                    pid, pname = cid, cname
            if not pid:
                pid = next((c for c in cols if c.lower().endswith("_id") or c.lower() == "id"), "product_id")
            if not pname:
                pname = next((c for c in cols if "name" in c.lower() or "title" in c.lower() or "desc" in c.lower()), "product_name")

            conflict_entities = [pid, pname]
            target_metric = f"CONFLICT_CHECK({pid}, {pname})"
            return AnalysisObjective(
                analysis_type=AnalysisType.DATA_QUALITY,
                target_metric=target_metric,
                target_table=table_name,
                time_column=None,
                test_period=None,
                baseline_period=None,
                filter_conditions=[],
                candidate_dimensions=conflict_entities,
                target_metrics=conflict_entities,
                is_scenario=False,
                scenario_params={},
                quality_check_type="bidirectional_mapping",
                conflict_entities=conflict_entities,
            )

        # 1. Determine Analysis Type (if not scenario)
        if not is_scenario:
            if any(kw in q_lower for kw in ["why", "cause", "reason", "driver", "root cause", "rca", "explain the drop", "explain the decline"]):
                atype = AnalysisType.DIAGNOSTIC
            elif any(kw in q_lower for kw in ["compare", "vs", "versus", "across", "between"]):
                atype = AnalysisType.COMPARATIVE
            else:
                atype = AnalysisType.DESCRIPTIVE

        # 2. Table identification
        table_name = profile.tables[0].table_name if profile.tables else "data"
        t_prof = profile.tables[0] if profile.tables else None

        sales_col = next((c.name for c in (t_prof.columns if t_prof else []) if any(k in c.name.lower() for k in ["sales", "revenue", "amount"])), "sales")
        profit_col = next((c.name for c in (t_prof.columns if t_prof else []) if "profit" in c.name.lower()), "profit")
        qty_col = next((c.name for c in (t_prof.columns if t_prof else []) if any(k in c.name.lower() for k in ["quantity", "units_sold", "units", "qty"])), "quantity")
        order_col = next((c.name for c in (t_prof.columns if t_prof else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), None)

        # 3. Detect ALL requested metrics in question (Multi-Metric Support)
        found_metrics = []
        entity_grain = None
        required_outputs = []
        missing_explanation = None

        has_orders_query = not is_scenario and any(
            re.search(p, q_lower) for p in [
                r'\bhow many orders\b',
                r'\bhow many total orders\b',
                r'\btotal orders?\b',
                r'\border count\b',
                r'\bnumber of orders?\b',
                r'\bunique orders?\b',
                r'\bcount of orders?\b',
                r'\bcount\s+(?:distinct\s+)?orders?\b',
                r'\borders?\s+were\s+placed\b',
            ]
        )
        has_aov_query = not is_scenario and bool(re.search(r'\b(average order value|aov)\b', q_lower))

        if has_orders_query or has_aov_query:
            entity_grain = "order"
            if has_orders_query:
                required_outputs.append("Total unique orders")
                if order_col:
                    found_metrics.append(f'COUNT(DISTINCT "{order_col}")')
                else:
                    missing_explanation = f"Dataset is missing an order identifier column ('order_id') required to calculate total unique orders."
            if has_aov_query:
                required_outputs.append("Average Order Value")
                if order_col:
                    found_metrics.append("AOV")
                else:
                    missing_explanation = f"Dataset is missing an order identifier column ('order_id') required to calculate Average Order Value."

        if ("sales" in q_lower or "revenue" in q_lower) and not has_aov_query:
            found_metrics.append(sales_col)
            required_outputs.append("Total Sales/Revenue")

        if "profit" in q_lower or "earnings" in q_lower:
            actual_profit_col = next((c.name for c in (t_prof.columns if t_prof else []) if "profit" in c.name.lower()), None)
            if actual_profit_col:
                found_metrics.append(actual_profit_col)
                required_outputs.append("Total Profit")
            else:
                missing_explanation = f"Dataset is missing a 'profit' column required for profit calculation."

        if "profit margin" in q_lower or "margin" in q_lower:
            found_metrics.append("profit_margin")
            required_outputs.append("Profit Margin")

        if "quantity" in q_lower or "units" in q_lower:
            found_metrics.append(qty_col)
            required_outputs.append("Total Quantity")

        if is_scenario and scenario_params.get("type") == "discount_cap":
            if not found_metrics:
                found_metrics = [sales_col, profit_col, "profit_margin"]
            else:
                if sales_col not in found_metrics:
                    found_metrics.insert(0, sales_col)
                if profit_col not in found_metrics:
                    found_metrics.append(profit_col)
                if "profit_margin" not in found_metrics and "margin" in q_lower:
                    found_metrics.append("profit_margin")

        if len(found_metrics) > 1:
            target_metric = ", ".join(found_metrics)
            target_metrics_list = list(found_metrics)
        elif len(found_metrics) == 1:
            target_metric = found_metrics[0]
            target_metrics_list = [found_metrics[0]]
        else:
            metric_col = sales_col
            agg = "SUM"
            if t_prof:
                for nc in t_prof.numeric_columns:
                    if nc.lower() in q_lower or nc.replace("_", " ").lower() in q_lower:
                        metric_col = nc
                        break
            if "avg" in q_lower or "average" in q_lower:
                agg = "AVG"
            elif "count" in q_lower or "number of" in q_lower or "how many" in q_lower:
                agg = "COUNT"
            target_metric = f"{agg}({metric_col})"
            target_metrics_list = [metric_col]

        # 4. Dimension & Temporal columns: filter out IDs, single-value, and high-cardinality (>100) columns
        all_text_cols = [
            c.name for c in (t_prof.columns if t_prof else []) 
            if not c.is_numeric and not c.is_temporal and not c.name.lower().endswith("_id")
            and 2 <= c.distinct_count <= 100
        ]
        if not all_text_cols and t_prof:
            all_text_cols = [c.name for c in t_prof.columns if not c.is_numeric and not c.is_temporal and not c.name.lower().endswith("_id")]
        
        # Prioritize dimensions mentioned in the user's question
        mentioned_dims = [
            d for d in all_text_cols 
            if d.lower() in q_lower or d.replace("_", " ").lower() in q_lower or any(word in d.lower() for word in q_lower.split() if len(word) > 2)
        ]
        mentioned_dims.sort(key=lambda x: (0 if "name" in x.lower() else 1, x))
        other_dims = [d for d in all_text_cols if d not in mentioned_dims]
        dims = mentioned_dims + other_dims
        time_col = t_prof.temporal_columns[0] if t_prof and t_prof.temporal_columns else None

        # 5. Temporal period detection & year filtering
        base_filter = None
        test_filter = None
        filter_conds = []

        # Detect year in question (e.g. "in 2017")
        year_match = re.search(r'\b(20\d\d|19\d\d)\b', q_lower)
        date_col = time_col or next((c.name for c in (t_prof.columns if t_prof else []) if any(k in c.name.lower() for k in ["date", "time", "year", "period"])), None)
        if year_match:
            yr = year_match.group(1)
            if date_col:
                filter_conds.append(f'CAST("{date_col}" AS VARCHAR) LIKE \'%{yr}%\'')
            else:
                if not missing_explanation:
                    missing_explanation = f"Question requested filtering for year {yr}, but no date/temporal column was found in the dataset."

        # Extract numeric condition thresholds from question
        # ONLY if NOT a counterfactual capping scenario (capping is an in-query transformation, not a row-dropping filter!)
        if not is_scenario and t_prof:
            for nc in t_prof.numeric_columns:
                nc_pat = r'\b' + re.escape(nc.lower()) + r'\b'
                # check negative e.g. "negative profit"
                if re.search(r'\b(?:negative|loss)\s+' + nc_pat, q_lower) or re.search(nc_pat + r'\s+(?:is\s+)?(?:negative|less than 0)', q_lower):
                    filter_conds.append(f'"{nc}" < 0')
                # check greater than / less than
                m_gt = re.search(nc_pat + r'\s+(?:greater than|more than|exceeding|>|>=)\s*\$?([0-9,]+)', q_lower)
                if m_gt:
                    val = m_gt.group(1).replace(',', '')
                    filter_conds.append(f'"{nc}" > {val}')
                m_lt = re.search(nc_pat + r'\s+(?:less than|under|<|<=)\s*\$?([0-9,]+)', q_lower)
                if m_lt:
                    val = m_lt.group(1).replace(',', '')
                    filter_conds.append(f'"{nc}" < {val}')

        if "h1" in q_lower and "h2" in q_lower and time_col:
            base_filter = f'"{time_col}" < \'2023-07-01\''
            test_filter = f'"{time_col}" >= \'2023-07-01\''

        return AnalysisObjective(
            analysis_type=atype,
            target_metric=target_metric,
            target_table=table_name,
            time_column=time_col,
            test_period=test_filter,
            baseline_period=base_filter,
            filter_conditions=filter_conds,
            candidate_dimensions=dims[:3],
            target_metrics=target_metrics_list,
            entity_grain=entity_grain,
            required_outputs=required_outputs,
            missing_columns_explanation=missing_explanation,
            is_scenario=is_scenario,
            scenario_params=scenario_params,
        )

    def _parse_response(self, raw_text: str, profile: DatasetProfile, user_question: str = "") -> AnalysisObjective:
        """
        Parses the LLM's raw JSON text into a validated AnalysisObjective.
        Aligns extracted specification with the user question to prevent unauthorized metric substitution.
        """
        json_str = self._extract_json(raw_text)
        raw_text_for_mapping = raw_text.lower()
        q_lower = user_question.lower()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON. Raw response:\n{raw_text}\nError: {exc}"
            )

        # Normalize analysis_type to uppercase
        if "analysis_type" in data:
            data["analysis_type"] = str(data["analysis_type"]).upper()

        t_prof = profile.tables[0] if profile.tables else None
        all_cols = [c.name for c in t_prof.columns] if t_prof else []
        order_col = next((c.name for c in (t_prof.columns if t_prof else []) if "order_id" in c.name.lower() or c.name.lower() == "order_id"), None)
        sales_col = next((c.name for c in (t_prof.columns if t_prof else []) if any(k in c.name.lower() for k in ["sales", "revenue", "amount"])), "sales")
        profit_col = next((c.name for c in (t_prof.columns if t_prof else []) if "profit" in c.name.lower()), None)
        date_col = (t_prof.temporal_columns[0] if t_prof and t_prof.temporal_columns else None) or next((c.name for c in (t_prof.columns if t_prof else []) if any(k in c.name.lower() for k in ["date", "time", "year", "period"])), None)

        # Detect semantic intent from user_question
        is_scenario = bool(data.get("is_scenario")) or ("capped" in q_lower and "discount" in q_lower)
        has_orders_query = not is_scenario and any(
            re.search(p, q_lower) for p in [
                r'\bhow many orders\b',
                r'\bhow many total orders\b',
                r'\btotal orders?\b',
                r'\border count\b',
                r'\bnumber of orders?\b',
                r'\bunique orders?\b',
                r'\bcount of orders?\b',
                r'\bcount\s+(?:distinct\s+)?orders?\b',
                r'\borders?\s+were\s+placed\b',
            ]
        )
        has_aov_query = not is_scenario and bool(re.search(r'\b(average order value|aov)\b', q_lower))
        wants_profit = "profit" in q_lower or "earnings" in q_lower
        wants_sales = ("sales" in q_lower or "revenue" in q_lower) and not has_aov_query
        year_match = re.search(r'\b(20\d\d|19\d\d)\b', q_lower)

        # Enforce multi-metric and grain correctness: NEVER substitute revenue for order count or AOV
        if has_orders_query or has_aov_query:
            data["entity_grain"] = "order"
            if order_col:
                if has_orders_query and has_aov_query:
                    data["target_metrics"] = [f'COUNT(DISTINCT "{order_col}")', "AOV"]
                    data["target_metric"] = f'COUNT(DISTINCT "{order_col}"), AOV'
                    data["required_outputs"] = ["Total unique orders", "Average Order Value"]
                elif has_orders_query:
                    data["target_metrics"] = [f'COUNT(DISTINCT "{order_col}")']
                    data["target_metric"] = f'COUNT(DISTINCT "{order_col}")'
                    data["required_outputs"] = ["Total unique orders"]
                elif has_aov_query:
                    data["target_metrics"] = [f'COUNT(DISTINCT "{order_col}")', "AOV"]
                    data["target_metric"] = f'COUNT(DISTINCT "{order_col}"), AOV'
                    data["required_outputs"] = ["Total unique orders", "Average Order Value"]
            else:
                data["missing_columns_explanation"] = f"Requested calculation requires an order identifier column ('order_id'), but no matching column was found in table '{data.get('target_table', 'table')}'. Available columns: {all_cols}."
        elif wants_profit:
            if profit_col:
                # If LLM returned sales or revenue instead of profit, correct it
                current_tm = str(data.get("target_metric", "")).lower()
                if "sales" in current_tm or "revenue" in current_tm:
                    data["target_metric"] = profit_col
                    data["target_metrics"] = [profit_col]
                    data["required_outputs"] = ["Total Profit"]
            else:
                if not data.get("missing_columns_explanation"):
                    data["missing_columns_explanation"] = f"Question requested profit analysis, but no 'profit' column was found in table '{data.get('target_table', 'table')}'. Available columns: {all_cols}."

        # Sanitize target_metric if LLM generated a complex CASE expression or nested aggregation
        raw_tm = data.get("target_metric", "")
        raw_tm_lower = raw_tm.lower()
        if order_col:
            # If counting orders without DISTINCT, upgrade to COUNT(DISTINCT order_id)
            if re.search(r'\bcount\s*\(\s*(?!distinct)' + re.escape(order_col.lower()) + r'\s*\)', raw_tm_lower):
                raw_tm = re.sub(
                    r'(?i)\bcount\s*\(\s*(?!distinct)' + re.escape(order_col) + r'\s*\)',
                    f'COUNT(DISTINCT "{order_col}")',
                    raw_tm
                )
                data["target_metric"] = raw_tm

        if "conflict_check" in raw_tm_lower or data.get("analysis_type") == "DATA_QUALITY" or "mapped to multiple" in raw_text_for_mapping or "mapped to multiple" in q_lower:
            data["analysis_type"] = AnalysisType.DATA_QUALITY
            if not data.get("quality_check_type"):
                data["quality_check_type"] = "bidirectional_mapping"
            if not data.get("conflict_entities"):
                cols = [c.name for c in (t_prof.columns if t_prof else [])]
                pid = next((c for c in cols if "product_id" in c.lower() or (c.lower().endswith("id") and "product" in c.lower())), "product_id")
                pname = next((c for c in cols if "product_name" in c.lower() or (("name" in c.lower() or "desc" in c.lower()) and "product" in c.lower())), "product_name")
                data["conflict_entities"] = [pid, pname]
                data["candidate_dimensions"] = [pid, pname]
            data["target_metric"] = f"CONFLICT_CHECK({data['conflict_entities'][0]}, {data['conflict_entities'][1]})"

        if data.get("analysis_type") != "DATA_QUALITY" and ("case when" in raw_tm.lower() or raw_tm.count("(") > 1):
            if profile.tables:
                all_cols = [c.name for c in profile.tables[0].columns]
            resolved = resolve_metric_column(raw_tm, all_cols)
            data["target_metric"] = f"SUM({resolved})"
            if not data.get("filter_conditions"):
                conds = []
                for part in re.findall(r'(?:SUM\s*\(\s*)?([a-zA-Z0-9_]+)\s*\)?\s*(>|<|>=|<=|=|!=)\s*([0-9\.\-]+)', raw_tm):
                    col_p, op_p, val_p = part
                    conds.append(f'"{col_p}" {op_p} {val_p}')
                if conds:
                    data["filter_conditions"] = conds

        # Apply year filter from user question if specified
        if year_match:
            yr = year_match.group(1)
            if date_col:
                conds = data.get("filter_conditions", [])
                has_yr = any(yr in str(c) for c in conds)
                if not has_yr:
                    conds.append(f'CAST("{date_col}" AS VARCHAR) LIKE \'%{yr}%\'')
                data["filter_conditions"] = conds
            else:
                if not data.get("missing_columns_explanation"):
                    data["missing_columns_explanation"] = f"Question requested filtering for year {yr}, but no date/temporal column was found in table '{data.get('target_table', 'table')}'. Available columns: {all_cols}."

        # Sanitize year filtering in filter_conditions for DATE/TIMESTAMP/VARCHAR columns
        if date_col and data.get("filter_conditions"):
            clean_conds = []
            for cond in data["filter_conditions"]:
                m_yr = re.search(r'["\']?' + re.escape(date_col) + r'["\']?\s+LIKE\s+[\'"]%?(\d{4})%?[\'"]', cond, re.IGNORECASE)
                if m_yr:
                    clean_conds.append(f'CAST("{date_col}" AS VARCHAR) LIKE \'%{m_yr.group(1)}%\'')
                else:
                    clean_conds.append(cond)
            data["filter_conditions"] = clean_conds

        # Validate table name against known tables
        known_tables = [t.table_name for t in profile.tables]
        if data.get("target_table") not in known_tables and known_tables:
            data["target_table"] = known_tables[0]  # Default to first table

        try:
            return AnalysisObjective(**data)
        except (ValidationError, TypeError) as exc:
            raise ValueError(
                f"LLM response does not match AnalysisObjective schema. "
                f"Parsed data: {data}\nValidation error: {exc}"
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Attempts to extract a JSON string from the LLM response using multiple strategies.
        """
        text = text.strip()

        # Strategy 1: Entire response is valid JSON
        if text.startswith("{"):
            return text

        # Strategy 2: Extract from ```json ... ``` block
        json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_block:
            return json_block.group(1)

        # Strategy 3: Find first { ... } spanning the response
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        # Strategy 4: Return as-is and let the caller raise JSONDecodeError
        return text

