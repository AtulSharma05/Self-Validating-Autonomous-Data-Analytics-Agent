# Technical Implementation Specification: Self-Validating Autonomous Data Analytics Agent

**Project:** A Self-Validating Autonomous Data Analytics Agent for Diagnostic and Evidence-Based Decision Support  
**Project Code:** Major Project - 15B19CI791 (B.Tech 7th Semester, JIIT Sector 128)  
**Authors:** Parth Tiwari (9923103258), Atul Sharma (9923103263)  
**Supervisor:** Mr. Noor Mohammad (Assistant Professor, CSE)  

---

## Table of Contents
1. [System Architectural Overview](#system-architectural-overview)
2. [Feature 1: Data Acquisition, Ingestion & In-Memory Storage](#feature-1-data-acquisition-ingestion--in-memory-storage)
3. [Feature 2: Automated Structural & Statistical Data Profiling](#feature-2-automated-structural--statistical-data-profiling)
4. [Feature 3: Natural-Language Question Understanding & Objective Extraction](#feature-3-natural-language-question-understanding--objective-extraction)
5. [Feature 4: Autonomous Multi-Step Planning (LangGraph State Machine)](#feature-4-autonomous-multi-step-planning-langgraph-state-machine)
6. [Feature 5: Python / SQL Code Generation](#feature-5-python--sql-code-generation)
7. [Feature 6: Sandboxed Execution Environment](#feature-6-sandboxed-execution-environment)
8. [Feature 7: Deterministic Neuro-Symbolic Validation Engine](#feature-7-deterministic-neuro-symbolic-validation-engine)
9. [Feature 8: Iterative Self-Correction & Error Recovery Engine](#feature-8-iterative-self-correction--error-recovery-engine)
10. [Feature 9: Hypothesis-Based Root-Cause Analysis (RCA) & Decomposition](#feature-9-hypothesis-based-root-cause-analysis-rca--decomposition)
11. [Feature 10: Evidence Collection & Provenance DAG Tracking](#feature-10-evidence-collection--provenance-dag-tracking)
12. [Feature 11: Evidence Sufficiency & Mathematical Confidence Scoring](#feature-11-evidence-sufficiency--mathematical-confidence-scoring)
13. [Feature 12: Interactive Evidence-Based Reporting & Visualizations](#feature-12-interactive-evidence-based-reporting--visualizations)
14. [Feature 13: Benchmark Suite & 5-Way Ablation Study Framework](#feature-13-benchmark-suite--5-way-ablation-study-framework)
15. [Master Dependency Matrix & Requirements](#master-dependency-matrix--requirements)

---

## System Architectural Overview

```
                      +-------------------------------------------------------------+
                      |                      USER INTERACTION                       |
                      |          Streamlit Dashboard / FastAPI REST API             |
                      +-----------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------+
| 1. DATA INGESTION & PROFILING                                                                     |
|    - Upload CSV/Excel/Parquet                                                                     |
|    - DuckDB & Pandas load dataset in-memory (Never sent raw to LLM)                               |
|    - Automated Profiler computes compact JSON Schema, Null Ratios, Quantiles, and Cardinalities   |
+-------------------------------------------+-------------------------------------------------------+
                                            |
                                            v (Metadata JSON + User Query)
+---------------------------------------------------------------------------------------------------+
| 2. QUESTION UNDERSTANDING & LANGGRAPH ORCHESTRATION                                              |
|    - Pydantic structured extraction of target metric, comparison baseline, & analysis type        |
|    - LangGraph State Machine manages: Plan -> CodeGen -> Exec -> Validate -> Correct -> RCA      |
+-------------------------------------------+-------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
| 3. EXECUTION & DETERMINISTIC VALIDATION LOOP                                                      |
|    - Sandboxed Execution (Docker / isolated subprocess with resource & timeout limits)            |
|    - Deterministic Validator checks:                                                              |
|        * Invariants (Sum of parts == Whole)                                                       |
|        * Schema & Boundary rules (No negative sales, no future dates)                             |
|        * Dual-Path verification (DuckDB SQL aggregation vs Pandas aggregation)                     |
|    - If Invalid -> Structured Error Classification -> LangGraph Self-Correction Loop (Max 3)     |
+-------------------------------------------+-------------------------------------------------------+
                                            | (Valid Intermediate Result)
                                            v
+---------------------------------------------------------------------------------------------------+
| 4. DIAGNOSTIC ROOT-CAUSE ANALYSIS (RCA)                                                           |
|    - Hypothesis formulation (Pareto decomposition across high-variance dimensions)                 |
|    - Parallel sub-query execution (Volume vs Price, Region, Product Category)                      |
|    - Factor attribution ranking (Variance Explained %)                                            |
+-------------------------------------------+-------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
| 5. PROVENANCE, CONFIDENCE SCORING & REPORTING                                                     |
|    - Provenance DAG: Records step_id, query, execution timestamp, input slice, and validation hash|
|    - Multi-factor Evidence Confidence Score (ECS = w1*Exec + w2*DataQuality + w3*Attr - w4*Contra)|
|    - Rich interactive UI: Executive summary, Plotly decomposition charts, audit-trace logs        |
+---------------------------------------------------------------------------------------------------+
```

---

## Feature 1: Data Acquisition, Ingestion & In-Memory Storage

### Goal
Efficiently load raw structured files (CSV, Excel `.xlsx`, Parquet) into an analytical engine without exposing raw rows to the LLM, preserving privacy and keeping token usage near-zero.

### In Which Way (Implementation Logic)
1. User uploads files via Streamlit UI or FastAPI endpoint `/api/upload`.
2. A temporary session UUID is generated per session.
3. Files are stored on a local temporary volume (`/tmp/analytics_sessions/<session_id>/`).
4. Ingestion registers the files as virtual tables directly inside an in-memory **DuckDB** connection (`duckdb.connect(database=':memory:')`).
5. DuckDB's native zero-copy scanning (`read_csv_auto()`, `read_parquet()`) ensures that even 1M+ rows load in under 150ms.

### Using What (Libraries & Tools)
* **DuckDB Python Client (`duckdb`)**: Primary analytical database engine.
* **Pandas (`pandas`) / OpenPyXL (`openpyxl`)**: For converting Excel multi-sheet workbooks into DuckDB tables.
* **Pydantic**: For file upload metadata validation.

### Data Contract / Code Structure
```python
# src/ingestion/loader.py
import duckdb
from pathlib import Path

class DataIngestionEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.con = duckdb.connect(database=':memory:')
        
    def load_file(self, file_path: str, table_name: str) -> dict:
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
        elif ext == ".parquet":
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
        elif ext in [".xlsx", ".xls"]:
            import pandas as pd
            df = pd.read_excel(file_path)
            self.con.register(table_name, df)
        
        row_count = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return {"table_name": table_name, "row_count": row_count}
```

---

## Feature 2: Automated Structural & Statistical Data Profiling

### Goal
Generate a rich, token-efficient metadata schema profile of all tables. The LLM sees *only* this statistical summary—never the underlying private records.

### In Which Way (Implementation Logic)
1. Execute DuckDB's `SUMMARIZE <table_name>` query to instantly extract:
   * Column names, inferred data types (`BIGINT`, `VARCHAR`, `TIMESTAMP`, `DOUBLE`).
   * Missing value count and `null_percentage`.
   * Statistical metrics: min, max, approximate unique values (`approx_unique`), mean, standard deviation, and quartiles (q25, q50, q75).
2. Compute categorical cardinalities: columns with $< 30$ unique values are flagged as `Categorical Dimensions` and their distinct categories are included as context.
3. Identify potential date/timestamp columns and compute minimum and maximum date ranges.
4. Export the resulting metadata as a compact JSON object (~500–1000 tokens) formatted specifically for the LLM system prompt.

### Using What (Libraries & Tools)
* **DuckDB**: Fast SQL-based summary queries (`SUMMARIZE`, `approx_count_distinct`).
* **NumPy**: For array quantiles and outlier boundary calculations (IQR rule: $Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$).

### Output Data Contract
```json
{
  "tables": [
    {
      "name": "sales_transactions",
      "row_count": 125430,
      "columns": [
        {"name": "order_id", "type": "VARCHAR", "is_primary_key": true, "null_pct": 0.0},
        {"name": "order_date", "type": "TIMESTAMP", "min": "2023-01-01", "max": "2023-12-31"},
        {"name": "region", "type": "VARCHAR", "distinct_values": ["North", "South", "East", "West"], "cardinality": 4},
        {"name": "revenue", "type": "DOUBLE", "min": 5.0, "max": 14200.0, "mean": 450.2, "null_pct": 0.001},
        {"name": "units_sold", "type": "INTEGER", "min": 1, "max": 120}
      ]
    }
  ]
}
```

---

## Feature 3: Natural-Language Question Understanding & Objective Extraction

### Goal
Translate an unstructured, ambiguous user question (e.g., *"Why did revenue plunge in Q3 compared to Q2?"*) into an unambiguous, strongly-typed analytical specification.

### In Which Way (Implementation Logic)
1. Use **Instructor** or **LangChain structured output** backed by Pydantic models.
2. The LLM receives the system prompt with the Dataset Profile JSON + User Question.
3. The LLM fills out a validated `AnalysisObjective` schema:
   * Identifies `analysis_type`: `DESCRIPTIVE` (what happened?), `DIAGNOSTIC` (why did it happen?), `COMPARATIVE`, or `PREDICTIVE`.
   * Identifies target metrics (e.g., `SUM(revenue)`).
   * Identifies comparison temporal windows (Current: `2023-Q3`, Baseline: `2023-Q2` or `2022-Q3`).
   * Extracts filter dimensions (e.g., specific regions, product categories).
   * Identifies candidate drill-down dimensions for root-cause testing.

### Using What (Libraries & Tools)
* **Pydantic v2**: Strict schema definition and type coercion.
* **LLM Engine**: Google Gemini 2.5/3 Flash (via `@google/genai` or `langchain-google-genai`) with `response_schema` or `with_structured_output()`.

### Code Implementation Schema
```python
# src/agent/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AnalysisType(str, Enum):
    DESCRIPTIVE = "DESCRIPTIVE"
    DIAGNOSTIC = "DIAGNOSTIC"
    COMPARATIVE = "COMPARATIVE"

class AnalysisObjective(BaseModel):
    analysis_type: AnalysisType = Field(description="Type of analytics requested")
    target_metric: str = Field(description="Primary metric to analyze, e.g. revenue, profit, quantity")
    target_table: str = Field(description="Primary table containing the target metric")
    aggregation_func: str = Field(description="Aggregation function: SUM, AVG, COUNT")
    test_period: str = Field(description="Filter or time window for the period of interest, e.g. '2023-Q3'")
    baseline_period: Optional[str] = Field(None, description="Baseline period for comparison, e.g. '2023-Q2'")
    candidate_dimensions: List[str] = Field(default_factory=list, description="Categorical columns for root-cause breakdown")
```

---

## Feature 4: Autonomous Multi-Step Planning (LangGraph State Machine)

### Goal
Decompose high-level diagnostic objectives into an ordered execution graph of discrete analytical tasks, dynamically branching and looping based on intermediate findings.

### In Which Way (Implementation Logic)
1. Construct a stateful graph using **LangGraph**.
2. Define a shared `AnalyticsAgentState` containing the conversation history, data schema, analysis plan, execution results, validation errors, and provenance DAG.
3. Graph Nodes:
   * `plan_node`: Generates an ordered list of tasks (e.g., Step 1: Calculate baseline, Step 2: Calculate test period, Step 3: Compute delta, Step 4: Attribute delta across dimensions).
   * `codegen_node`: Writes SQL/Pandas code for the current active step.
   * `sandbox_exec_node`: Runs code safely against DuckDB.
   * `validate_node`: Executes deterministic checks.
   * `correct_node`: Fixes code if validation fails (conditional edge: loops back to `codegen_node` if $attempts < 3$, else fails gracefully).
   * `rca_node`: Generates and ranks hypotheses if the question is diagnostic.
   * `report_node`: Synthesizes final report with evidence and confidence scores.

### Using What (Libraries & Tools)
* **LangGraph (`langgraph`)**: Orchestrating cycles, state updates, and conditional routing.
* **LangChain Core (`langchain_core`)**: Message state management.

### Graph Architecture Diagram
```
[START]
   |
   v
[plan_node] ---------> [codegen_node]
                           |
                           v
                   [sandbox_exec_node]
                           |
                           v
                   [validate_node]
                      /         \
            (Invalid /           (Valid)
             Attempts < 3)          \
                    v                v
            [correct_node]       [Is Diagnostic?]
                    |             /            \
                    |       (Yes) /              \ (No)
                    +---> [rca_node]              v
                              \             [report_node]
                               \                  ^
                                v                 |
                           [provenance_node] -----+
                                                  |
                                                  v
                                                [END]
```

---

## Feature 5: Python / SQL Code Generation

### Goal
Generate precise, performant DuckDB SQL queries and accompanying Python data manipulation code tailored to the validated plan step.

### In Which Way (Implementation Logic)
1. The planner assigns a specific sub-goal to the CodeGen module (e.g., *"Calculate total revenue for Q2 and Q3 grouped by product_category"*).
2. The prompt supplies:
   * The exact DuckDB SQL dialect rules (e.g., use `STRFTIME`, zero-division protection via `NULLIF(denominator, 0)`).
   * Exact table schemas and verified column names from Feature 2.
   * Coding rules: No arbitrary assumptions, always assign results to a standard variable name (`result_df`), return scalar metrics in a standard JSON key.
3. The LLM outputs structured code enclosed in markdown blocks or Pydantic fields.

### Using What (Libraries & Tools)
* **Gemini 2.5/3 Flash**: High coding benchmarks, low latency, large context window.
* **SQLGlot (`sqlglot`)**: Pre-execution AST parser that validates DuckDB SQL syntax before it even touches the database engine.

---

## Feature 6: Sandboxed Execution Environment

### Goal
Execute generated Python and SQL code without risking host machine compromise, endless hanging loops, or memory exhaustion.

### In Which Way (Implementation Logic)
1. **Two-Tier Execution Strategy**:
   * **Tier 1 (Fast In-Process DuckDB Sandbox):** Pure SQL queries run directly in an in-memory DuckDB connection with strict memory limits (`SET max_memory = '2GB'`) and execution timeouts via Python `concurrent.futures`.
   * **Tier 2 (Isolated Python Container / Subprocess):** If code requires arbitrary Python execution (e.g., complex stats with SciPy or visualization generation):
     * Code is executed in an isolated temporary container (Docker with network disabled `--network none`, memory capped `-m 1024m`, CPU capped `--cpus 1.0`, read-only dataset mount).
     * Alternatively, for lower latency in development: an isolated Python `subprocess` using `resource.setrlimit()` (POSIX memory and CPU time constraints) inside a temporary working directory.
2. Captures `stdout`, `stderr`, execution time, and serialized result payloads (JSON/Arrow/DataFrame).

### Using What (Libraries & Tools)
* **Python `concurrent.futures.ThreadPoolExecutor`**: Hard timeout enforcement (e.g., 10 seconds max per query).
* **Docker SDK for Python (`docker`)**: Containerized sandbox for non-standard code execution.
* **PyArrow (`pyarrow`)**: Fast zero-copy data exchange between DuckDB and Pandas.

---

## Feature 7: Deterministic Neuro-Symbolic Validation Engine

### Goal
**This is the core research novelty of the project.** The system never asks the LLM *"Did you calculate this correctly?"*. Instead, an independent algorithmic engine verifies results deterministically using formal data invariants.

### In Which Way (Implementation Logic)
The engine applies three independent layers of verification:

#### 1. Mathematical Invariant Reconciliation (Sum-of-Parts Rule)
When computing dimensional breakdowns (e.g., revenue by region), the validator automatically triggers an independent background query calculating the unsegmented grand total:
$$\left| \sum_{r \in \text{Regions}} \text{Revenue}_r - \text{Total Revenue} \right| < \epsilon \quad (\epsilon = 10^{-2})$$
*If the sum of slices does not equal the whole, the query is immediately invalidated* (detects faulty joins that dropped rows or double-counted records due to Cartesian products).

#### 2. Domain & Boundary Rule Checks
* **Non-Negativity Constraint:** Metrics like `revenue`, `units_sold`, `inventory_count` must satisfy $\ge 0$.
* **Ratio Constraints:** Percentages, margins, and discounts must fall within $[0, 100]\%$.
* **Temporal Monotonicity:** Date ranges analyzed must exist within the dataset's known bounds.

#### 3. Dual-Path Verification (Orthogonal Execution)
For critical numerical claims:
* **Path A:** Executed via generated DuckDB SQL query.
* **Path B:** An independent, hardcoded vectorized Pandas aggregation is executed on the same table slice.
* The validator asserts:
  $$\text{assert } |Result_A - Result_B| < 10^{-4}$$

### Using What (Libraries & Tools)
* **Pydantic**: Invariant rule definition.
* **Custom Python Rule Engine (`src/validation/invariant_rules.py`)**: Vectorized numpy assertion routines.

### Code Implementation Example
```python
# src/validation/validator.py
class ValidationResult(BaseModel):
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    audit_details: dict = {}

class DeterministicValidator:
    @staticmethod
    def verify_sum_of_parts(con, table_name: str, metric_col: str, group_col: str, filters: str = "1=1") -> ValidationResult:
        # Grand total
        total_query = f"SELECT SUM({metric_col}) FROM {table_name} WHERE {filters}"
        grand_total = con.execute(total_query).fetchone()[0] or 0.0
        
        # Segment sum
        parts_query = f"SELECT SUM(part_sum) FROM (SELECT SUM({metric_col}) as part_sum FROM {table_name} WHERE {filters} GROUP BY {group_col})"
        parts_total = con.execute(parts_query).fetchone()[0] or 0.0
        
        diff = abs(grand_total - parts_total)
        if diff > 0.01:
            return ValidationResult(
                is_valid=False,
                error_type="INVARIANT_VIOLATION_SUM_OF_PARTS",
                error_message=f"Sum of {group_col} parts ({parts_total}) does not equal grand total ({grand_total}). Discrepancy: {diff}",
                audit_details={"grand_total": grand_total, "parts_total": parts_total, "diff": diff}
            )
        return ValidationResult(is_valid=True, audit_details={"reconciled_total": grand_total})
```

---

## Feature 8: Iterative Self-Correction & Error Recovery Engine

### Goal
When execution crashes or deterministic validation fails, automatically analyze the failure, construct actionable feedback, and regenerate the corrected query without human intervention.

### In Which Way (Implementation Logic)
1. **Error Classification:** The error is parsed into one of 4 standardized categories:
   * `SYNTAX_ERROR`: SQL parsing or Python runtime exception (e.g., column does not exist).
   * `INVARIANT_ERROR`: Mathematical discrepancy (e.g., sum of parts $\ne$ total).
   * `BOUNDARY_ERROR`: Impossible values detected (e.g., negative inventory).
   * `EMPTY_RESULT_ERROR`: Query returned 0 rows because of overly restrictive filters.
2. **Targeted Prompt Injection:** The LangGraph state injects:
   * The failed query.
   * The specific error type and exact discrepancy message.
   * Suggested correction guidance (e.g., *"Table sales uses 'order_amount' instead of 'revenue'; use 'LEFT JOIN' instead of 'INNER JOIN' to prevent dropping null records"*).
3. **Bounded Iteration Loop:** A loop counter increments. If the agent fails 3 times on the same sub-task, it stops the loop, logs an `UNRESOLVED_DISCREPANCY` alert, and switches to a fallback query strategy.

### Using What (Libraries & Tools)
* **LangGraph Conditional Edges**: `should_continue(state) -> "correct_node" | "rca_node" | "fail_gracefully"`.

---

## Feature 9: Hypothesis-Based Root-Cause Analysis (RCA) & Decomposition

### Goal
Transform a simple metric change into an intelligent breakdown that investigates multiple competing hypotheses and identifies which dimensions explain the variance.

### In Which Way (Implementation Logic)
1. **Variance Decomposition Formulation:**
   Given a delta in total metric between Baseline $B$ and Test Period $T$:
   $$\Delta M = M_T - M_B$$
2. **Dimension Screening:**
   The profiler provides candidate categorical dimensions: `Region`, `Product_Category`, `Customer_Segment`, `Sales_Channel`.
3. **Variance Contribution Calculation:**
   For each category $c$ in dimension $D$, calculate its absolute delta and percentage contribution:
   $$\text{Contribution}(c) = \frac{M_{T, c} - M_{B, c}}{\Delta M} \times 100\%$$
4. **Volume vs. Price Decomposition (for Sales/Revenue queries):**
   $$\Delta \text{Revenue} = \underbrace{(Q_T - Q_B) \times P_B}_{\text{Volume Effect}} + \underbrace{(P_T - P_B) \times Q_T}_{\text{Price Effect}}$$
5. **Ranking & Hypothesis Filtering:**
   Factors are ranked in descending order of explained variance. Only factors with $|\text{Contribution}| \ge 10\%$ are promoted to the final report as "Primary Contributing Factors".

### Using What (Libraries & Tools)
* **DuckDB Analytical SQL**: Window functions (`LAG`, `OVER`), grouping sets, and rolling aggregations.
* **Pandas**: Vectorized contribution calculations and waterfall table generation.

---

## Feature 10: Evidence Collection & Provenance DAG Tracking

### Goal
Guarantee complete auditability. Every single number, chart, and bullet point in the final report can be traced back to the exact code, dataset slice, and validation timestamp that generated it.

### In Which Way (Implementation Logic)
1. Maintain a directed acyclic graph (DAG) of analytical steps inside `AgentState["provenance_store"]`.
2. For every executed operation, generate an immutable `EvidenceArtifact`:
   * `step_id`: Monotonically increasing index (e.g., `step_004`).
   * `query_sql`: Exact SQL executed.
   * `raw_result`: Scalar or tabular output.
   * `validation_status`: `VALIDATED_DETERMINISTICALLY` or `FAILED`.
   * `checksum_hash`: SHA-256 hash of the input table slice and query string.
3. When the LLM generates the final textual report, it is forced to cite evidence tokens (e.g., `[Evidence #step_004]`).

### Using What (Libraries & Tools)
* **Python `hashlib`**: SHA-256 cryptographic provenance hashing.
* **NetworkX (`networkx`)**: Managing the DAG of dependencies between analysis steps.

---

## Feature 11: Evidence Sufficiency & Mathematical Confidence Scoring

### Goal
Provide a transparent, mathematically grounded **Evidence Confidence Score ($ECS \in [0, 100]\%$)** to replace arbitrary LLM confidence estimations.

### In Which Way (Implementation Logic)
The score is computed using a formal weighted multi-factor formula:

$$ECS = 100 \times \left( w_1 \cdot V_{\text{exec}} + w_2 \cdot D_{\text{cov}} + w_3 \cdot E_{\text{attr}} - w_4 \cdot C_{\text{contradict}} \right)$$

| Component | Metric | Formulation | Weight ($w_i$) |
| :--- | :--- | :--- | :--- |
| **Validation Score ($V_{\text{exec}}$)** | Proportion of deterministic invariant checks passed | $\frac{\text{Passed Checks}}{\text{Total Checks Tested}}$ | **0.40** |
| **Data Completeness ($D_{\text{cov}}$)** | Cleanliness of relevant columns | $1.0 - \text{null\_rate}_{\text{key\_cols}}$ | **0.25** |
| **Attribution Explanatory Power ($E_{\text{attr}}$)** | Share of observed variance explained by identified factors | $\min\left(1.0, \frac{\sum |\Delta \text{Top Factors}|}{|\Delta \text{Total}|}\right)$ | **0.25** |
| **Contradiction Penalty ($C_{\text{contradict}}$)** | Presence of conflicting sub-trends | $1.0$ if opposing trends exceed threshold, else $0.0$ | **0.10** |

* **Sufficiency Threshold:**
  * If $ECS \ge 75\% \implies$ **High Confidence Evidence**.
  * If $50\% \le ECS < 75\% \implies$ **Moderate Confidence (Flagged with caveats)**.
  * If $ECS < 50\% \implies$ Output warning: *"Insufficient evidence to determine a reliable contributing factor."*

### Using What (Libraries & Tools)
* Custom Python mathematical module (`src/reporting/confidence.py`).

---

## Feature 12: Interactive Evidence-Based Reporting & Visualizations

### Goal
Present the business decision-maker with a clean, executive-level dashboard showing the findings, interactive charts, validation stamps, and full audit provenance.

### In Which Way (Implementation Logic)
1. **Frontend Architecture:**
   * Built with **Streamlit** (interactive prototype) and styled with modern clean typography, dark/light cards, and glassmorphic badges.
2. **Visualization Engine:**
   * Dynamic **Plotly** figures:
     * **Waterfall Chart**: Demonstrates baseline metric flowing to test metric through positive and negative contributing factors.
     * **Multi-Period Trend Line**: Metric over time with annotated change points.
     * **Segmented Bar Chart**: Comparative breakdown of top decline drivers.
3. **Traceability Inspector:**
   * Clicking on any claim or card opens an expandable "Audit Provenance" drawer showing the exact SQL query, execution duration, and deterministic verification checkmarks ($\checkmark$).

### Using What (Libraries & Tools)
* **Streamlit (`streamlit`)**: Rapid, interactive user interface.
* **Plotly Express & Graph Objects (`plotly`)**: Interactive, exportable charts (JSON/HTML).
* **FastAPI (`fastapi`) & Uvicorn**: Underlying REST API serving the agent logic.

---

## Feature 13: Benchmark Suite & 5-Way Ablation Study Framework

### Goal
Provide empirical, quantifiable proof for your thesis and project evaluation that the proposed self-validating agent outperforms standard LLM approaches.

### In Which Way (Implementation Logic)
1. **Question Benchmark Suite (50–100 Questions):**
   * Category 1: Single-table descriptive queries (Easy).
   * Category 2: Multi-step comparative queries (Medium).
   * Category 3: Diagnostic Root-Cause queries with multiple factors (Hard).
   * Category 4: Questions on noisy datasets with intentional missing values/anomalies (Stress-test).
   * Category 5: Questions with insufficient data (Testing if the agent appropriately refrains from guessing).
2. **5-Way Ablation Configuration:**
   * **System A:** LLM Only (Vanilla prompt, direct answer).
   * **System B:** LLM + Code Execution (Standard Code Interpreter / PandasAI style).
   * **System C:** LLM + Code Execution + Deterministic Validation (No self-correction).
   * **System D:** LLM + Validation + Self-Correction Loop.
   * **System E (Proposed):** Complete System (LangGraph + Validation + Self-Correction + Diagnostic RCA + Provenance).
3. **Automated Evaluation Runner:**
   * Runs all 50 questions across Systems A through E.
   * Logs execution time, API cost, calculation accuracy (% matching ground-truth), and unsupported claim rate (hallucination rate).
   * Generates evaluation comparison charts and markdown summary tables.

### Using What (Libraries & Tools)
* **Pytest (`pytest`)**: Automated test runner.
* **DABStep Benchmark / Synthetic Ground-Truth Datasets**: Test fixtures.

---

## Master Dependency Matrix & Requirements

| Library / Tool | Version | Purpose in this Project |
| :--- | :--- | :--- |
| **`python`** | `>=3.10, <3.12` | Core programming language |
| **`duckdb`** | `^1.1.0` | Ultra-fast in-memory analytical SQL engine over CSV/Parquet |
| **`pandas`** | `^2.2.0` | In-memory data manipulation, dual-path verification |
| **`numpy`** | `^1.26.0` | Numerical calculations, quantile distributions, IQR bounds |
| **`langgraph`** | `^0.2.0` | Stateful cyclic multi-agent graph orchestration |
| **`langchain-core`** | `^0.3.0` | Base agent abstractions, runnables, and prompts |
| **`google-genai`** | Latest | Official SDK for Gemini 2.5/3 Flash API execution |
| **`pydantic`** | `^2.8.0` | Strict schema validation, typed state, and structured outputs |
| **`sqlglot`** | `^25.0.0` | SQL parser, transpiler, and pre-execution syntax validator |
| **`plotly`** | `^5.22.0` | Interactive charts (waterfall, bar charts, time-series) |
| **`streamlit`** | `^1.38.0` | Web dashboard prototype and interactive analysis UI |
| **`fastapi`** | `^0.112.0` | REST API backend for modular decoupled architecture |
| **`docker`** | `^7.1.0` | Sandboxed Python execution environment |
| **`openpyxl`** | `^3.1.5` | Excel (.xlsx) ingestion support |
| **`pytest`** | `^8.3.0` | Unit testing and benchmark evaluation suite |

---

## Next Steps for Development

1. **Step 1:** Create project skeleton following this directory structure.
2. **Step 2:** Write `requirements.txt` with the master dependencies above.
3. **Step 3:** Implement the `DataIngestionEngine` and `AutomatedDataProfiler` (Features 1 & 2).
4. **Step 4:** Implement `DeterministicValidator` and `InvariantRules` (Feature 7).
5. **Step 5:** Build the LangGraph workflow assembling the full loop.
