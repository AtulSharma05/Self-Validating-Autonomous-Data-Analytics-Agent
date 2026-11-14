# Local Model Setup Guide

> This project uses **Ollama** to run free, local LLMs — no API keys, no cost, runs fully offline.

---

## Models Required

| Role | Model | Size | Why This Model |
|:---|:---|:---|:---|
| **Primary LLM** (Planning, Code Gen, NL→Objective, RCA narration) | `qwen2.5-coder:7b` | ~4.7 GB | Best free model for SQL + Python code generation; strong structured output adherence |
| **Structured Output / Fallback** | `llama3.1:8b` | ~4.7 GB | Excellent instruction following; used as fallback if Qwen fails structured JSON extraction |

> Both models fit comfortably on a machine with **8 GB RAM** (one at a time).  
> If you have only 8 GB total RAM, use `qwen2.5-coder:7b` only — it handles both roles.

---

## Step 1 — Install Ollama

### macOS
```bash
brew install ollama
```
Or download the installer from https://ollama.com/download

Verify:
```bash
ollama --version
```

---

## Step 2 — Start the Ollama Server

```bash
ollama serve
```

This starts the local API server at `http://localhost:11434`. Keep this terminal open (or run it as a background service).

To run as a background service on macOS:
```bash
brew services start ollama
```

---

## Step 3 — Pull the Required Models

```bash
ollama pull qwen2.5-coder:7b
```

```bash
ollama pull llama3.1:8b
```

Each download is ~4–5 GB. Verify they are available:
```bash
ollama list
```

Expected output:
```
NAME                    ID              SIZE    MODIFIED
qwen2.5-coder:7b        ...             4.7 GB  ...
llama3.1:8b             ...             4.7 GB  ...
```

---

## Step 4 — Configure the Project

Copy the example env file:
```bash
cp .env.example .env
```

The `.env` file is already pre-configured for Ollama:
```env
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5-coder:7b
```

Change `OLLAMA_MODEL_NAME` to `llama3.1:8b` if you want to switch models.

---

## Step 5 — Verify Connection

Run this quick check from the project root:
```bash
curl http://localhost:11434/api/tags
```

You should see a JSON list of your pulled models.

---

## Why These Models (Not Others)

| Model | Reason Skipped |
|:---|:---|
| `mistral:7b` | Weaker at structured JSON output (Pydantic schema filling) |
| `codellama:7b` | Outdated; Qwen2.5-Coder is strictly better for SQL/Python |
| `gemma2:9b` | Requires more RAM; overkill for this task |
| `phi3:mini` | Too small; struggles with multi-step planning prompts |

---

## RAM & Performance Guide

| Available RAM | Recommended Model | Expected Speed |
|:---|:---|:---|
| 8 GB | `qwen2.5-coder:7b` | ~15–25 tokens/sec |
| 16 GB | `qwen2.5-coder:7b` + `llama3.1:8b` (sequential) | ~20–35 tokens/sec |
| 16 GB + GPU | Either model with GPU layers | ~50–80 tokens/sec |

---

## Switching to Cloud (Optional)

If you want faster responses or are preparing the demo, switch to Gemini Flash:
```env
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

The codebase is provider-agnostic — `src/config.py` controls the switch.

