# Stage 3 Status

## What Stage 3 Implements

Stage 3 transforms the single-turn Stage 2 RAG demo into a **full conversational deep research assistant**. Every prior capability is preserved — retrieval quality, FAISS, citations — and the following is added on top:

1. **Persistent conversation state** — every turn's Q&A is stored in SQLite and survives app restarts.
2. **Multi-turn memory injection** — the last N turns are injected verbatim into the prompt context.
3. **Rolling conversation summaries** — after every N turns, an LLM-generated summary replaces growing verbatim history to keep the prompt bounded.
4. **Streaming operational events** — the agent yields `PipelineEvent` objects at every stage; the UI renders them in real time inside `st.status()`.
5. **Session picker** — the sidebar lists all past sessions; selecting one restores the full conversation history.
6. **Chat-style UI** — conversation history renders as `st.chat_message()` bubbles; new queries use `st.chat_input()`.
7. **Context builder upgrade** — `build_context()` now accepts `conversation_summary` and `recent_turns`, with a layered token budget that always prioritises evidence over memory.

---

## Updated Architecture

```
User Query
    ↓
Tavily Search             (app/research/search.py)
    ↓
Async Page Fetch          (app/research/fetcher.py)
    ↓
Text Extraction           (app/research/extractor.py)
    ↓
Persist Turn + Sources    (app/memory/repositories.py)
    ↓
Token Chunker             (app/research/chunker.py)
    ↓
Embedding Service         (app/research/embeddings.py)
    ↓
FAISS Vector Store        (app/research/vector_store.py)
    ↓
Dense Retriever           (app/research/retriever.py)
    ↓
Load Conversation Memory  (app/memory/conversation_memory.py)  ← NEW
    ↓
Context Builder           (app/research/context_builder.py)   ← UPGRADED
    ↓
LLM Generator             (app/research/generator.py)
    ↓
Persist Final Answer      (app/memory/repositories.py)        ← NEW
    ↓
Rolling Summary (if due)  (app/research/summarizer.py)        ← NEW
    ↓
ResearchResult + PipelineEvents
    ↓
Streamlit UI              (app/ui/streamlit_app.py)           ← UPGRADED
    ↓
SQLite Persistence        (app/memory/)
```

---

## Conversational Memory Architecture

Memory is **simple, stable, and grounded** — no vector retrieval of past turns.

| Layer | What it stores | Where |
|---|---|---|
| Verbatim recent turns | Last N completed Q&A pairs | `research_turns` table |
| Rolling summary | LLM-generated factual summary of older history | `sessions.rolling_summary` column |

**Memory injection flow:**
1. `get_recent_memory(session_id, limit=N)` fetches the N most recent completed turns from SQLite, returned in chronological order.
2. `format_memory_block(turns, summary)` formats them into a token-bounded string block.
3. `build_context()` receives both `conversation_summary` and `recent_turns`; it allocates part of the token budget to them.

**Memory never grows unbounded:** once the summary is triggered, older verbatim turns no longer appear in the prompt — only the rolling summary carries their essence forward.

---

## Rolling Summary Architecture

The rolling summary is a compact factual prose (≤350 words) that covers:
*   Main research topics discussed
*   Key conclusions and facts established
*   Important entities (names, technologies, dates)
*   Unresolved questions the user raised

**Trigger:** The agent calls `update_rolling_summary()` whenever `completed_turns % summary_after_turns == 0`. Default: every 4 turns.

**Implementation:** `app/research/summarizer.py` calls the existing `generate_answer()` with a special summary prompt. It reuses the same Gemini/OpenAI provider routing. The result is persisted via `SessionRepository.update_summary()`.

**Important:** Summaries cover **conversation history only** — not retrieved webpage evidence.

---

## Streaming Event Architecture

The agent's `stream_run()` method is an `AsyncGenerator` that yields:
*   `PipelineEvent(stage, message)` — one per pipeline stage (12 total)
*   `ResearchResult` — the final result object (always last)

The UI runs `collect_stream()` which drains the generator into a list, then replays events into `st.status()`. This gives real-time operational feedback without requiring true async streaming inside Streamlit.

**Stage events emitted:**

| Stage key | Message shown in UI |
|---|---|
| `search` | 🔍 Searching the web… |
| `fetch` | 📥 Fetching N pages… |
| `extract` | 📄 Extracting readable content… |
| `persist` | 💾 Saving research turn… |
| `chunk` | ✂️ Chunking documents… |
| `embed` | 🧠 Embedding N chunks… |
| `index` | 🗄️ Indexing vectors in FAISS… |
| `retrieve` | 🎯 Retrieving relevant evidence… |
| `memory` | 🧩 Loading conversation memory… |
| `context` | 🏗️ Building grounded context… |
| `generate` | 💬 Generating grounded answer… |
| `summarize` | 📝 Updating conversation summary… *(only every N turns)* |
| `done` | ✅ Research complete. |

---

## What Should Currently Work

1. **End-to-end research pipeline** — query → grounded cited answer.
2. **Multi-turn conversation** — follow-up questions can reference prior answers.
3. **Session persistence** — conversation history survives app restarts.
4. **Session picker** — restore any previous session from the sidebar.
5. **Rolling summary** — auto-generated every 4 turns, displayed in UI.
6. **Live pipeline events** — 12 stage messages appear one-by-one during research.
7. **All Stage 1 + 2 tests** still pass (54 total).
8. **Backward-compatible `run()` method** on `ResearchAgent` for existing tests.
9. **Existing DB compatibility** — `ALTER TABLE` migration safely adds `rolling_summary` to pre-existing databases.

---

## What Is NOT Implemented Yet

- BM25 lexical retrieval
- Hybrid retrieval (dense + BM25 rank fusion)
- Token-streaming of LLM answer (partial word-by-word output)
- FastAPI backend
- Evaluation harness
- Sarvam AI integration
- Per-session FAISS index isolation (currently global)

---

## Setup Instructions

### 1. Activate virtual environment
```bash
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required:
```env
TAVILY_API_KEY=tvly-...
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
LLM_MODEL=gemini-2.5-flash
```

Optional Stage 3 tuning:
```env
MAX_RECENT_TURNS=3          # verbatim turns injected per prompt
MAX_SUMMARY_TOKENS=400      # token budget for rolling summary
SUMMARY_AFTER_TURNS=4       # regenerate summary every N turns
```

### 4. Run the application
```bash
streamlit run app/ui/streamlit_app.py
```

---

## Exact Commands to Test the Project

```bash
# Run all tests (Stage 1 + 2 + 3)
venv/bin/pytest tests/ -v

# Run only Stage 3 tests
venv/bin/pytest tests/test_conversation_memory.py \
                tests/test_summarizer.py \
                tests/test_context_builder_s3.py \
                tests/test_agent_s3.py -v
```

---

## Example Multi-Turn Workflow

**Turn 1:**
> User: *What are the latest breakthroughs in quantum computing?*  
> Assistant: [grounded answer with `[S1]`, `[S2]`, `[S3]` citations]

**Turn 2 (follow-up):**
> User: *How does that compare to the state of photonic quantum computing specifically?*  
> — The agent now injects Turn 1's Q&A into the prompt as `[Recent Conversation]`.  
> Assistant: [answer referencing the prior context + new evidence]

**Turn 3:**
> User: *Which companies are leading in this space?*  
> — Turn 1 + Turn 2 are both injected as recent conversation.

**Turn 4 (summary trigger):**
> After this turn completes, `completed_count % 4 == 0` → rolling summary is generated and saved. A "📋 Conversation Summary has been updated" notice appears in the UI.

**Turn 5+:**
> Older verbatim turns are no longer injected; only the rolling summary carries their context. The most recent 3 turns are still injected verbatim.

---

## Context Budget Layout

For a 6000-token budget (`MAX_CONTEXT_TOKENS=6000`):

| Section | Approx. tokens | Priority |
|---|---|---|
| System prompt + question | ~150 (fixed) | Always included |
| Rolling summary | ≤ 400 | 2nd |
| Recent turns | ≤ 900 (3 × 300) | 3rd |
| Evidence chunks | ~4550 (remainder) | Highest — fills last |

Evidence always wins: if memory blocks consume too much, they are truncated first.

---

## Known Limitations

*   **Global FAISS index** — all sessions share the same vector index. Per-session isolation planned for Stage 4.
*   **No token-streaming** — the full answer is generated before display; `st.status()` shows operational events only.
*   **Summary quality** — depends on LLM; if generation fails, the previous summary is retained gracefully.
*   **No deduplication across sessions** — the same URL may be fetched and chunked multiple times across different sessions.
*   **Sequential embedding** — the embedding model runs on CPU; large document sets will be slow.

---

## Next Planned Stage

**Stage 4:** BM25 lexical retrieval + hybrid dense+BM25 rank fusion, per-session FAISS isolation, and optionally FastAPI backend.