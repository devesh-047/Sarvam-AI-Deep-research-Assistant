# Stage 2 Status

## What Stage 2 Implements

Stage 2 builds the first complete RAG research agent loop on top of the Stage 1 pipeline. Given a user query it now:

1. Searches the web via Tavily
2. Fetches pages concurrently (async httpx)
3. Extracts readable text (trafilatura + BeautifulSoup fallback)
4. **Chunks** documents into overlapping token-aware pieces (tiktoken)
5. **Embeds** chunks locally (sentence-transformers)
6. **Stores vectors** in FAISS (cosine similarity, persisted to disk)
7. **Persists chunks** (with embedding_ids) in SQLite
8. **Retrieves** relevant chunks by dense similarity search
9. **Builds** a grounded context prompt with citation labels and token budget
10. **Generates** a cited answer via a modular LLM provider (defaulting to Google Gemini API)
11. **Returns** answer + citations + retrieved evidence

---

## Updated Architecture

```
User Query
    ↓
Tavily Search          (app/research/search.py)
    ↓
Async Page Fetch       (app/research/fetcher.py)
    ↓
Text Extraction        (app/research/extractor.py)
    ↓
Token Chunker          (app/research/chunker.py)   ← NEW
    ↓
Embedding Service      (app/research/embeddings.py) ← NEW
    ↓
FAISS Vector Store     (app/research/vector_store.py) ← NEW
    ↓
Dense Retriever        (app/research/retriever.py) ← NEW
    ↓
Context Builder        (app/research/context_builder.py) ← NEW
    ↓
LLM Generator          (app/research/generator.py) ← UPGRADED (Modular: Gemini & OpenAI)
    ↓
ResearchAgent          (app/research/agent.py)     ← NEW
    ↓
Streamlit UI           (app/ui/streamlit_app.py)   ← UPGRADED
    ↓
SQLite Persistence     (app/memory/)
```

---

## What Should Currently Work

1. **End-to-end research pipeline** — one query returns a grounded cited answer.
2. **FAISS index persists** across restarts (saved to `data/faiss/`).
3. **Chunk metadata survives** the full pipeline and is retrievable from SQLite.
4. **Citations** are labelled `[S1]`, `[S2]`, … and displayed in the UI.
5. **Retrieved evidence** is shown with source + score in expandable UI panels.
6. **Graceful fallback** if `GEMINI_API_KEY` (or `LLM_API_KEY` for OpenAI) is missing (no crash, placeholder message shown).
7. All **Stage 1 tests** still pass.

---

## What Is NOT Implemented Yet

- BM25 lexical retrieval
- Hybrid retrieval (dense + BM25 rank fusion)
- Conversational memory / multi-turn context
- Rolling conversation summaries
- Streaming answer generation
- FastAPI backend
- Evaluation harness
- Sarvam AI integration

---

## Setup Instructions

### 1. Activate virtual environment
```bash
source venv/bin/activate
```

### 2. Install Stage 2 dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Required keys:
```env
TAVILY_API_KEY=tvly-...          # from https://tavily.com
LLM_PROVIDER=gemini              # gemini or openai
GEMINI_API_KEY=AIzaSy...         # Google Gemini API Key
LLM_MODEL=gemini-2.5-flash       # any Gemini model (or OpenAI model if provider=openai)
```

Optional tuning:
```env
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=100
RETRIEVAL_TOP_K=6
MAX_CONTEXT_TOKENS=6000
```

### 4. Run the application
```bash
streamlit run app/ui/streamlit_app.py
```

---

## Exact Commands to Test the Project

```bash
# Run all tests
pytest tests/ -v

# Run only Stage 2 tests
pytest tests/test_chunker.py tests/test_embeddings.py tests/test_vector_store.py \
       tests/test_chunk_repo.py tests/test_context_builder.py \
       tests/test_generator.py tests/test_retriever.py -v
```

> **Note:** Embedding tests download the `all-MiniLM-L6-v2` model (~90 MB) on first run.

---

## Example Research Workflow

1. Open `http://localhost:8501` in your browser.
2. Enter a question: *"What are the latest breakthroughs in quantum computing?"*
3. Click **Run Research**.
4. The status panel shows each pipeline step completing in sequence.
5. When complete, you see:
   - A grounded **answer** with inline `[S1]`, `[S2]` citations.
   - A **Citations** section linking to the original sources.
   - An expandable **Retrieved Evidence** section with the actual chunks used.

---

## Retrieval Architecture Explanation

| Component | Technology | Role |
|-----------|-----------|------|
| Chunker | tiktoken (cl100k_base) | Splits docs into 512-token overlapping windows |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Produces 384-dim L2-normalised vectors |
| Index | FAISS `IndexFlatIP` | Cosine similarity via inner product on normalised vectors |
| ID Map | JSON file + SQLite | Maps FAISS row position → SQLite chunk_id |
| Retriever | DenseRetriever | Embeds query → FAISS search → SQLite metadata lookup |

**Why IndexFlatIP + normalised vectors?**  
After L2-normalising both the query and stored vectors, inner product equals cosine similarity. `IndexFlatIP` is exact (no approximation) and straightforward to reason about for a Stage 2 baseline.

---

## Citation Architecture Explanation

1. The **context builder** assigns a unique `[S1]`, `[S2]`, … label per distinct source URL.
2. Multiple chunks from the same URL share the same label.
3. The label appears in the evidence block shown to the LLM.
4. The LLM is instructed to cite claims using these labels.
5. The UI renders each citation as a clickable link to the original source.

---

## Known Limitations

- FAISS index is **global** (all sessions share it). Per-session isolation will come in Stage 3.
- The embedding model is loaded into CPU memory; inference is sequential.
- No retry logic on LLM calls.
- No streaming; the entire answer is generated before display.
- Context is limited to `MAX_CONTEXT_TOKENS` (default 6000); very long documents lose their tail chunks.

---

## Next Planned Stage

**Stage 3:** BM25 lexical retrieval + hybrid rank fusion + conversational memory.
