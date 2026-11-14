# Self-Validating Autonomous Data Analytics Agent
### *Diagnostic and Evidence-Based Decision Support for Tabular Data*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.1.0+-yellow.svg)](https://duckdb.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0+-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0+-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-124%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Major Project (Course Code: 15B19CI791)**  
> **Institution:** Jaypee Institute of Information Technology (JIIT), Sector 128, Noida  
> **Authors:** Atul Sharma (Enrolment No: 9923103263) & Parth Tiwari (Enrolment No: 9923103258)  
> **Supervisor:** Mr. Noor Mohammad (Assistant Professor, Department of CSE & IT)  

---

## 📖 Executive Summary

Large Language Model (LLM) analytical assistants commonly suffer from hallucinations, silent calculation failures, and lack of auditable provenance when tasked with diagnosing real-world business datasets.

The **Self-Validating Autonomous Data Analytics Agent** is a neuro-symbolic system that bridges high-level semantic reasoning with deterministic invariant verification. Rather than passing raw enterprise datasets directly to LLMs, it leverages zero-copy in-memory engines (**DuckDB** & **Pandas**), plans multi-step analytical trajectories via a **LangGraph** cyclic state machine, verifies intermediate calculations against algebraic invariants, performs hypothesis-based Root-Cause Analysis (RCA), and generates a cryptographically hashed **Provenance DAG** paired with a mathematically grounded **Evidence Confidence Score ($ECS$)**.

---

## 🏛️ System Architecture

```text
                                  +-------------------------------------------------------------+
                                  |                      USER INTERACTION                       |
                                  |          Streamlit Dashboard / FastAPI REST API             |
                                  +-----------------------------+-------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------------+
| 1. DATA INGESTION & PROFILING                                                                                                 |
|    - Upload CSV / Excel (.xlsx) / Parquet into ephemeral analytical sessions                                                  |
|    - DuckDB registers zero-copy in-memory tables (Raw data never passed to LLM context)                                       |
|    - Automated Profiler computes compact JSON schemas, quantile summaries, null ratios, and column cardinalities              |
+---------------------------------------------------------------+---------------------------------------------------------------+
                                                                |
                                                                v (Metadata JSON + User Query)
+-------------------------------------------------------------------------------------------------------------------------------+
| 2. QUESTION UNDERSTANDING & LANGGRAPH ORCHESTRATION                                                                          |
|    - Pydantic schema extracts target metrics, dimension grains, filters, and analysis type (Descriptive vs Diagnostic)        |
|    - LangGraph State Machine manages: Plan -> CodeGen -> Exec -> Validate -> Correct -> RCA -> Synthesize                   |
+---------------------------------------------------------------+---------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------------+
| 3. SANDBOXED EXECUTION & DETERMINISTIC VALIDATION                                                                             |
|    - Isolated subprocess & restricted AST execution with strict resource & timeout limits                                    |
|    - Deterministic Validator executes algebraic invariant checks:                                                             |
|        * Invariant 1: Sum of parts == Whole ($|\sum X_i - X_{total}| < \epsilon$)                                            |
|        * Invariant 2: Domain boundary rules (Non-negative sales, bounded margins, valid date ranges)                           |
|        * Invariant 3: Dual-path verification (DuckDB SQL aggregation cross-checked against Pandas aggregation)                |
|    - Structured Error Classification & Self-Correction Loop (Up to 3 automated retries with exact runtime feedback)           |
+---------------------------------------------------------------+---------------------------------------------------------------+
                                                                | (Validated Intermediate Results)
                                                                v
+-------------------------------------------------------------------------------------------------------------------------------+
| 4. DIAGNOSTIC ROOT-CAUSE ANALYSIS (RCA) ENGINE                                                                                |
|    - Automated Pareto decomposition across high-variance dimensions                                                           |
|    - Volume-Price mix effect decomposition ($\Delta \text{Revenue} = \Delta P \cdot V_{base} + \Delta V \cdot P_{base} + \dots$)|
|    - Factor attribution ranking (Variance Explained % and top negative/positive drivers)                                     |
+---------------------------------------------------------------+---------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------------+
| 5. PROVENANCE, CONFIDENCE SCORING & INTERACTIVE REPORTING                                                                     |
|    - Provenance DAG: Records step lineage, query hashes, timestamps, slice parameters, and verification tokens                |
|    - Evidence Confidence Score: $ECS = w_1 S_{exec} + w_2 S_{quality} + w_3 S_{attribution} - w_4 S_{contradiction} \in [0, 1]$|
|    - Streamlit 8-Section Dashboard: Executive KPIs, Plotly Waterfall & Attribution Charts, SQL Inspection, Audit DAG          |
+-------------------------------------------------------------------------------------------------------------------------------+
```

---

## ✨ Key Features & Capabilities

### 1. Zero-Copy Ingestion & Automated Statistical Profiling
- Upload CSV, Excel (`.xlsx`, `.xls`), or Parquet files.
- Ingests into an in-memory session managed by **DuckDB** with zero-copy table scanning.
- Computes structural metrics (types, row counts) and statistical distributions (null ratios, distinct counts, min/max/mean/quantiles, high-cardinality flags) without sending sensitive row data to LLM context.

### 2. Natural-Language Question Understanding & Objective Extraction
- Maps conversational questions (e.g., *"Why did total profit margin drop in Q3 2024 despite higher revenue?"*) into strongly-typed Pydantic schemas.
- Disambiguates column targets, baseline vs comparison time periods, dimension grains, filter constraints, and analysis intent (`DESCRIPTIVE` vs `DIAGNOSTIC`).

### 3. Cyclic Multi-Step LangGraph Orchestrator
- Implements an iterative, stateful execution graph using **LangGraph**.
- Node pipeline: `objective_node` $\rightarrow$ `plan_node` $\rightarrow$ `codegen_node` $\rightarrow$ `exec_node` $\rightarrow$ `validate_node` $\rightarrow$ `correct_node` $\rightarrow$ `rca_node` $\rightarrow$ `synthesize_node`.
- Dynamically cycles between validation failure and self-correction before advancing to subsequent analytical sub-tasks.

### 4. Sandboxed Code Execution
- Generates clean DuckDB SQL and restricted Python code.
- Executes within an isolated subprocess environment with strict memory ceiling, timeout limits (default: 30s), and AST-based prohibition of malicious imports (e.g., `os`, `sys`, socket, file write operations).

### 5. Deterministic Neuro-Symbolic Validation Engine
- **Algebraic Invariants:** Enforces mathematical consistency across grouped subsets ($\sum \text{subgroup} = \text{total}$).
- **Boundary & Schema Rules:** Detects domain violations (negative prices, impossible date sequences, null outputs).
- **Dual-Path Verification:** Compares SQL aggregation outputs directly with vectorized Pandas execution; discrepancies trigger self-correction.

### 6. Iterative Self-Correction & Error Recovery
- Classifies failures into concrete categories: `SYNTAX_ERROR`, `SCHEMA_MISMATCH`, `INVARIANT_VIOLATION`, `EMPTY_RESULT`, or `TIMEOUT`.
- Feeds structured error payloads back into the code generator to produce corrected queries without restarting the workflow.

### 7. Hypothesis-Based Root-Cause Analysis (RCA)
- Automatically isolates drivers of performance swings using:
  - **Pareto Dimension Decomposition:** Identifies which categories, regions, or segments explain $\ge 80\%$ of variance.
  - **Volume-Price Mix Decomposition:** Disentangles pure volume effects from pricing pressure and cross-product shifts.
  - **Attribution Ranking:** Quantifies net contribution percentages for each candidate dimension.

### 8. Provenance DAG & Evidence Confidence Score ($ECS$)
- Tracks end-to-end analytical steps as directed acyclic graph (DAG) artifacts using **NetworkX**.
- Calculates a mathematically normalized **Evidence Confidence Score ($ECS \in [0.0, 1.0]$)**:
  $$\text{ECS} = w_1 S_{\text{exec}} + w_2 S_{\text{quality}} + w_3 S_{\text{attribution}} - w_4 S_{\text{contradiction}}$$
  where weights default to $w_1=0.35, w_2=0.25, w_3=0.25, w_4=0.15$.

### 9. Dual Interface: Web Dashboard & REST API
- **Streamlit Web UI:** 8 modular sections (Executive Summary, KPI Cards, Factor Waterfall & Attribution Charts, SQL/Python Code Inspector, Provenance Audit Log, and Model Health).
- **FastAPI REST API:** Full programmatic endpoints for asynchronous file uploads, dataset profiling, and interactive analytical queries.

### 10. 5-Way Ablation Benchmark Framework
- Includes a dedicated evaluation suite comparing the agent against alternative architectures:
  - **System A:** Vanilla LLM (Approximation only, zero execution, high hallucination rate).
  - **System B:** Naive Code Execution (PandasAI / Code Interpreter without invariant validation).
  - **System C:** Code Execution + Deterministic Validation (no self-correction loop).
  - **System D:** Code Execution + Validation + Self-Correction (max 3 retries).
  - **System E:** **Proposed System** (LangGraph + Invariants + Self-Correction + RCA + Provenance + ECS).

---

## 📂 Repository Structure

```text
Major/
├── .env.example                       # Template for environment configuration
├── .gitignore                          # Standard Python / OS ignore rules
├── FEATURE_IMPLEMENTATION_SPEC.md      # Detailed 15-feature architectural spec
├── LOCAL_MODEL_SETUP.md                # Guide for running local models via Ollama
├── Major Synopsis DRAFT - Google Docs.pdf  # Project academic synopsis
├── Major_MidTermReport.pdf             # Mid-term project report
├── README.md                           # Master project documentation
├── Self_Validating_Analytics_Agent.pptx# Project presentation slide deck
├── Self_Validating_Analytics_Agent_pdf.pdf # Exported presentation slides
├── pytest.ini                          # Pytest configuration and test filters
├── requirements.txt                    # Project dependencies
├── data/
│   ├── benchmarks/
│   │   └── benchmark_questions.json    # Standardized evaluation questions
│   ├── synthetic/
│   │   └── sample_retail_sales.csv     # Synthetic retail dataset with anomalies
│   └── raw/                            # Directory for user-uploaded datasets
├── src/
│   ├── __init__.py
│   ├── config.py                       # Pydantic settings & environment manager
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── codegen.py                  # SQL & Python code generation
│   │   ├── connection.py               # Ollama / Cloud model connection manager
│   │   ├── graph.py                    # LangGraph cyclic state machine
│   │   ├── objective_extractor.py      # NL -> Pydantic objective parser
│   │   ├── planner.py                  # Multi-step analytical plan builder
│   │   ├── rca.py                      # Diagnostic Root-Cause Analysis engine
│   │   ├── schemas.py                  # Pydantic data models & contracts
│   │   └── state.py                    # LangGraph agent state schema
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application factory & routes
│   │   ├── session.py                  # DuckDB session life-cycle manager
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── analyze.py              # /api/analyze endpoints
│   │       └── upload.py               # /api/upload & /api/profile endpoints
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                      # Streamlit 8-section dark-slate dashboard
│   │   └── charts.py                   # Plotly waterfall, volume-price & factor charts
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── error_classifier.py         # Runtime & semantic error categorization
│   │   └── sandbox.py                  # Subprocess & AST-restricted execution
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                   # DuckDB zero-copy data loader
│   │   └── profiler.py                 # Automated structural & statistical profiler
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── confidence.py               # Evidence Confidence Score (ECS) engine
│   │   └── provenance.py               # Cryptographic Provenance DAG tracker
│   └── validation/
│       ├── __init__.py
│       ├── invariant_rules.py          # Sum-of-parts, boundary & dual-path rules
│       └── validator.py                # Deterministic & semantic coverage validator
└── tests/
    ├── __init__.py
    ├── benchmark/
    │   ├── __init__.py
    │   ├── evaluator.py                # 5-way ablation study evaluator
    │   └── runner.py                   # Benchmark execution script
    ├── test_api.py                     # FastAPI endpoint test suite
    ├── test_benchmark.py               # Benchmark suite integration tests
    ├── test_codegen.py                 # Code generator unit tests
    ├── test_connection.py              # LLM connectivity & fallback tests
    ├── test_dashboard.py               # Streamlit chart & session tests
    ├── test_end_to_end_phases_1_to_3.py# Full pipeline end-to-end integration
    ├── test_entity_grain_and_rca.py    # Dimension resolution & RCA logic tests
    ├── test_graph.py                   # LangGraph state machine node tests
    ├── test_ingestion.py               # File loading & session management tests
    ├── test_profiler.py                # Profiler metric & quantile tests
    ├── test_question_understanding_and_validation.py # NL parsing & invariant tests
    ├── test_rca.py                     # Root-Cause Analysis unit tests
    ├── test_reporting.py               # ECS scoring & Provenance DAG tests
    ├── test_sandbox.py                 # Python & SQL execution sandbox tests
    ├── test_semantic_correctness_and_evidence.py # Semantic coverage tests
    └── test_validation.py              # Deterministic validator test suite
```

---

## 🚀 Getting Started

### Prerequisites
- **Python:** 3.10, 3.11, 3.12, or 3.13
- **Ollama** (for local offline LLM inference) or API keys for Google Gemini / OpenAI.

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/AtulSharma05/Self-Validating-Autonomous-Data-Analytics-Agent.git
cd Self-Validating-Autonomous-Data-Analytics-Agent
```

---

### Step 2: Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell)
# python -m venv .venv
# .venv\Scripts\activate
```

---

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables
Copy `.env.example` to create your local `.env`:
```bash
cp .env.example .env
```

Default configuration for **Ollama** (Free, fully offline, no API key required):
```env
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5-coder:7b
OLLAMA_FALLBACK_MODEL_NAME=llama3.1:8b

DUCKDB_MAX_MEMORY=4GB
EXECUTION_TIMEOUT_SECONDS=30
MAX_CORRECTION_RETRIES=3
```

*(Optional: Set `MODEL_PROVIDER=gemini` with `GEMINI_API_KEY=...` or `MODEL_PROVIDER=openai` with `OPENAI_API_KEY=...` for cloud inference).*

---

### Step 5: Start Local LLM with Ollama (Recommended)
1. Install Ollama from [ollama.com](https://ollama.com/download) or via Homebrew on macOS:
   ```bash
   brew install ollama
   ```
2. Start the Ollama background daemon:
   ```bash
   ollama serve
   ```
3. Pull the recommended coding & reasoning models:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama pull llama3.1:8b
   ```
*(For detailed hardware sizing and instructions, see [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md)).*

---

## 🖥️ Running the Application

### Option A: Launch Interactive Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser at `http://localhost:8501`.
- **Upload Dataset:** Drag & drop CSV, Excel, or Parquet files (or click *"Load Sample Retail Dataset"*).
- **Inspect Profiling:** Review inferred data types, null rates, and distribution summaries.
- **Ask Natural-Language Questions:** Ask descriptive or diagnostic questions.
- **Inspect Verifications:** View step-by-step SQL/Python code, invariant verification checks, factor attribution waterfall charts, and cryptographic DAG provenance.

---

### Option B: Launch FastAPI REST Server
```bash
uvicorn src.api.main:app --reload --port 8000
```
Access the interactive Swagger API documentation at:
`http://localhost:8000/docs`

**Core Endpoints:**
- `GET  /api/health` — System status, active sessions, and Ollama connection state.
- `POST /api/upload` — Upload CSV / Excel / Parquet file and initialize DuckDB session.
- `GET  /api/profile/{session_id}` — Retrieve structural & statistical profile JSON.
- `POST /api/analyze` — Submit natural-language question for autonomous execution.
- `GET  /api/provenance/{session_id}` — Export Provenance DAG and audit trail.

---

## 🧪 Testing & Benchmark Evaluation

### Run Test Suite
The repository includes a comprehensive test suite (126 tests) covering sandboxed execution, invariant validation, semantic coverage, error recovery, and REST routes:

```bash
pytest
```
*Expected output: `124 passed, 2 deselected in ~3.3s`.*

Run tests with verbose logs and coverage:
```bash
pytest -v --durations=10
```

### Run 5-Way Ablation Benchmark
Evaluate the agent against baseline paradigms (System A through System E) across the benchmark question suite:
```bash
python -m tests.benchmark.runner
```

---

## 📊 Evaluation & Comparative Study

The proposed architecture addresses critical weaknesses identified in classical LLM analytics:

| Dimension | System A (Vanilla LLM) | System B (Standard Code Exec) | System C (Exec + Validation) | System D (Validation + Retry) | System E (Proposed Agent) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Execution Engine** | None (LLM hallucination) | Subprocess / Sandbox | Subprocess / Sandbox | Subprocess / Sandbox | **DuckDB + Subprocess Sandbox** |
| **Data Privacy** | Raw data in LLM prompt | Raw data in environment | Raw data in environment | Raw data in environment | **Zero-Copy In-Memory (Metadata only to LLM)** |
| **Invariant Checking** | ❌ None | ❌ None | ✅ Deterministic Rules | ✅ Deterministic Rules | **✅ Invariants + Dual-Path Cross-Validation** |
| **Self-Correction** | ❌ None | ❌ None | ❌ Halts on Error | ✅ Up to 3 Retries | **✅ Structured Error Classification & Targeted Recovery** |
| **Root-Cause Analysis**| ❌ Qualitative guess | ⚠️ Manual scripting | ⚠️ Manual scripting | ⚠️ Scripted without decomposition | **✅ Automated Pareto + Volume-Price Decomposition** |
| **Auditability** | ❌ Zero provenance | ⚠️ Ephemeral stdout | ⚠️ Log strings | ⚠️ Step logs | **✅ Cryptographic Provenance DAG + Confidence Score ($ECS$)** |

---

## 📚 Academic Documentation & Reports

The project repository includes complete academic deliverables:
- **[FEATURE_IMPLEMENTATION_SPEC.md](FEATURE_IMPLEMENTATION_SPEC.md)**: Exhaustive technical specification across all 15 core architectural features.
- **[LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md)**: Hardware requirements, quantization settings, and setup guide for running local LLMs via Ollama.
- **`Major Synopsis DRAFT - Google Docs.pdf`**: Approved academic project synopsis.
- **`Major_MidTermReport.pdf`**: Detailed mid-term technical report with theoretical formulations.
- **`Self_Validating_Analytics_Agent.pptx`** & **`Self_Validating_Analytics_Agent_pdf.pdf`**: Presentation slides with architectural diagrams and ablation findings.

---

## 👥 Authors & Academic Credits

- **Atul Sharma** (Enrolment No: **9923103263**) — Department of Computer Science & Engineering, JIIT-128
- **Parth Tiwari** (Enrolment No: **9923103258**) — Department of Computer Science & Engineering, JIIT-128
- **Supervisor:** **Mr. Noor Mohammad** (Assistant Professor, Department of Computer Science & Engineering, JIIT Sector 128, Noida)

---

## 📄 License

This project is developed as an academic Major Project at Jaypee Institute of Information Technology and is licensed under the [MIT License](LICENSE).
