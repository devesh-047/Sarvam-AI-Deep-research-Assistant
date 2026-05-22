# Stage 4 Status

## What Stage 4 Implements

Stage 4 completes the deep research assistant requirements by making the research process transparent, introducing a formal evaluation harness, and hardening the entire execution pipeline. A subsequent enhancement layer adds ReAct-style workflow UX without changing the underlying deterministic architecture.

The key features added in Stage 4 are:
1. **Explicit Research Planner** — LLM-based planner that generates a visible Thought/Planned Actions plan before searching, with deterministic fallback.
2. **ReAct-Style Workflow UX** — Each pipeline step emits a typed event (`thought`, `action`, `observation`, `system`) that renders as a styled workflow timeline in the UI.
3. **Streaming Final Answer** — The answer is streamed token-by-token from Gemini/OpenAI and displayed incrementally with a cursor.
4. **Conflict-Aware Observations** — When evidence comes from diverse or conflicting domains, the system emits a structural conflict observation (no hallucination).
5. **Evaluation Harness** — Automated 20-question dataset, metrics engine, and report generator.
6. **Hardening & Reliability** — Retry logic, URL deduplication, FAISS fallback.

---

## Final Architecture Overview

```
User Query
    ↓
Research Planner              (app/research/planner.py)        ← LLM-first, deterministic fallback
    Thought: what evidence is needed
    ↓
Tavily Web Search             (app/research/search.py)         ← Retry + multi-query
    Action: searching the web
    ↓
Async Page Fetch              (app/research/fetcher.py)
    Action: fetching N pages concurrently
    ↓
Text Extraction               (app/research/extractor.py)
    Action: extracting readable content
    Observation: extracted M/N pages
    ↓
Persist Turn + Sources        (app/memory/repositories.py)
    ↓
Token Chunker                 (app/research/chunker.py)
    ↓
Embedding Service             (app/research/embeddings.py)
    ↓
FAISS Vector Store            (app/research/vector_store.py)   ← Graceful fallback
    ↓
Dense Retriever               (app/research/retriever.py)
    Action: dense similarity retrieval
    ↓
Select Evidence               (app/research/agent.py)          ← Observation: N passages from M sources
    ↓
Conflict Detection            (app/research/agent.py)          ← Observation: structural disagreements
    ↓
Context Builder               (app/research/context_builder.py)
    ↓
Streaming LLM Generation      (app/research/generator.py)      ← Token-by-token streaming
    ↓
Persist Final Answer          (app/memory/repositories.py)
    ↓
Rolling Summary               (app/research/summarizer.py)
    ↓
ReAct-Style UI                (app/ui/streamlit_app.py)        ← Collapsible Agent Workflow panel
```

---

## ReAct-Style Workflow Visibility

### What It Is

The UI renders each pipeline step as a typed workflow event inside a collapsible **Agent Workflow** panel. The events follow the ReAct (Reason + Act) format:

```
💭 Thought:   Analysing question and planning research strategy...
⚡ Action:    Searching the web for: "quantum computing breakthroughs 2025"
🔍 Observation: Found 5 source(s): arxiv.org · nature.com · ibm.com
⚡ Action:    Fetching content from 5 webpage(s)...
⚡ Action:    Extracting readable text from fetched pages...
🔍 Observation: Extracted content from 5/5 page(s).
⚡ Action:    Retrieving the most relevant evidence passages via dense search...
🔍 Observation: Selected 6 evidence passages from 3 unique sources.
⚠️ Observation: Evidence from 3 sources (arxiv.org, nature.com, ibm.com). Multiple perspectives present.
💭 Thought:   Synthesising a grounded answer from retrieved evidence...
[Answer streams token-by-token...]
```

### Why This Is NOT Chain-of-Thought Exposure

These events are **pre-scripted operational summaries** of real pipeline steps. They describe what the agent **did** (fetched pages, retrieved passages) — not what it **thought internally**. No raw model reasoning, no hidden prompts, no token-level thinking traces are exposed.

### Event Type Schema

| Event Type | Icon | Meaning |
|---|---|---|
| `thought` | 💭 | Reasoning summary about what evidence is needed |
| `action` | ⚡ | A concrete tool call (search, fetch, embed, retrieve) |
| `observation` | 🔍 | Result of an action (N sources found, M pages extracted) |
| `system` | ⚙️ | Infrastructure step (persist, index, chunk) |
| `token` | — | One chunk of the streaming final answer |
| `final_answer` | — | Complete event carrying the full answer + citations |

---

## Streaming Answer Generation

### Architecture

`generator.py` now provides two entry points:
- `generate_answer(prompt)` — non-streaming (used by evaluation harness)
- `generate_answer_stream(prompt)` → async generator of string chunks

**Gemini**: uses `client.aio.models.generate_content_stream()`.
**OpenAI**: uses `stream=True` + delta chunks.
**Fallback**: if streaming fails, yields the whole answer as one chunk.

The agent's `stream_run()` consumes `generate_answer_stream()` and emits one `token` event per chunk. The UI accumulates these into a growing answer display with a blinking cursor (`▌`).

The `run()` non-streaming wrapper reassembles token events to produce the `ResearchResult`, keeping the evaluation harness fully compatible.

---

## Conflict-Aware Observations

When retrieved evidence comes from 3+ distinct domains, the agent emits a structural observation:

> *"Evidence was gathered from 3 different sources (arxiv.org, nature.com, ibm.com). Multiple perspectives may be present — the answer will synthesise all viewpoints."*

When official (.gov/.edu) and secondary sources both appear:

> *"Official sources (cdc.gov) and secondary sources (healthblog.com) were both retrieved. Conflicting claims will be noted in the answer."*

**These observations are purely structural** — no LLM is involved, no content is analysed, and no hallucinated conflicts are generated.

---

## LLM-Based Planner (Default)

The planner (`app/research/planner.py`) now runs LLM planning by default (`enable_llm_planning = True`).

It prompts the LLM with a constrained output format:

```
PLAN:
Thought:
<one sentence: what kind of evidence is needed>

Planned Actions:
1. Search official documentation
2. Compare multiple recent sources
3. Retrieve supporting evidence
4. Generate grounded answer with citations

QUERIES:
- <targeted search query 1>
- <targeted search query 2>
- <targeted search query 3>
```

On any failure (API error, parse error), it silently falls back to the deterministic planner.

---

## Collapsible Agent Workflow UI

The Streamlit interface now renders:

1. **🧠 Agent Workflow** *(collapsible, expanded by default during query)*
   - Thought / Action / Observation events with colour-coded styles
   - Evidence snippets with score badges inside the panel

2. **📋 Research Plan** *(collapsible, collapsed by default)*
   - Full plan text from the LLM planner
   - List of search queries

3. **Answer** *(main area, streams token-by-token)*

4. **📚 Sources** *(citation pills below the answer)*

5. **🎯 Retrieved Evidence** *(collapsible expander at bottom)*

### Event Style Guide

| Event Type | CSS Class | Visual |
|---|---|---|
| `thought` | `react-thought` | Purple-left-border, italic, grey-blue text |
| `action` | `react-action` | Blue-left-border, bold, blue text |
| `observation` | `react-observation` | Green-left-border, green text |
| `system` | `react-system` | Small grey monospace |

---

## Explicit Agentic Orchestration

The agent executes 14 explicit deterministic steps with no recursion or autonomous loops:

| Step | Stage | Event Type | Description |
|---|---|---|---|
| 0 | `memory` | system | Load conversation context |
| 1 | `plan` → `plan_complete` | thought | LLM research plan |
| 2 | `search` → `search_complete` | action → observation | Tavily search |
| 3 | `fetch` | action | Async page fetch |
| 4 | `extract` → `extract_complete` | action → observation | Text extraction |
| 5 | `persist_turn` | system | DB write |
| 6 | `chunk` → `chunk_complete` | action → system | Token chunking |
| 7 | `embed` | action | Embedding |
| 8 | `faiss` | system | FAISS indexing |
| 9 | `retrieve` | action | Dense retrieval |
| 10 | `select_evidence` | observation | Evidence selection |
| 10b | `conflict` | observation | Conflict detection |
| 11 | `context` | system | Prompt assembly |
| 12 | `llm` + `token`×N | action + token | Streaming generation |
| 13 | `summary` → `complete` | system → final_answer | Persist + complete |

---

## What Currently Works

1. **ReAct-Style Workflow Panel** — Visible Thought/Action/Observation stream in the UI.
2. **Streaming Answer** — Answer renders token-by-token with blinking cursor.
3. **LLM Planner** — Generates a custom, user-visible research plan per query.
4. **Conflict Observations** — Structural source diversity warnings.
5. **Evidence Snippets in Workflow** — Selected passages visible in the workflow panel.
6. **Citation Rendering** — Citation pills with labels, titles, URLs.
7. **Evaluation Harness** — Fully compatible (uses non-streaming `run()`).
8. **All Tests Pass** — 85/85 unit tests.

---

## Setup & Execution Commands

### Run the Streamlit Application
```bash
streamlit run app/ui/streamlit_app.py
```

### Run All Tests
```bash
venv/bin/pytest tests/ -v
```

---

## Evaluation Commands

```bash
# Dry run (no API calls)
python -m evaluation.runner \
    --dataset evaluation/dataset/eval_questions.json \
    --output-dir evaluation/results \
    --dry-run

# Full evaluation (requires API keys)
python -m evaluation.runner \
    --dataset evaluation/dataset/eval_questions.json \
    --output-dir evaluation/results

# Generate report
python -m evaluation.report_generator \
    --results evaluation/results/results.jsonl \
    --output evaluation/results/report.md
```

---

## Remaining Limitations

- **Global FAISS Index** — Chunks from all sessions share one index (by design; prevents cold-start).
- **Deterministic Planner Fallback** — If LLM planning API key is unavailable, falls back to keyword extraction silently.
