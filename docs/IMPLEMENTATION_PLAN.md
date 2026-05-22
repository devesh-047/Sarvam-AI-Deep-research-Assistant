# Deep Research Agent Implementation Plan

## Project Understanding

This project is not a simple chatbot. The goal is to build a Deep Research Agent similar to Perplexity or ChatGPT Deep Research style systems.

The system must:

- research information from the live web
- retrieve and analyze multiple sources
- synthesize grounded answers
- maintain conversational memory across sessions
- support multi-turn research workflows
- provide transparent citations
- stream visible operational progress updates

The system should behave like a research assistant rather than a generic LLM wrapper.

The focus is on:

- retrieval quality
- context construction
- grounding
- orchestration
- memory management
- uncertainty handling
- evaluation methodology

The assignment explicitly discourages agent frameworks because the orchestration logic itself is part of the evaluation.

## Constraints

Do not use:

- LangChain
- LangGraph
- CrewAI
- LlamaIndex
- Haystack
- any agent orchestration framework

Build all orchestration manually in Python.

## Final Recommended Architecture

```text
User Query
    ↓
Session Manager
    ↓
Planner
    ↓
Search Layer (Tavily)
    ↓
Page Fetcher
    ↓
HTML → Clean Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Cosine Retrieval + BM25 Lexical Retrieval
    ↓
Hybrid Rank Fusion
    ↓
Conversation Memory Retrieval
    ↓
Context Builder
    ↓
LLM Answer Generation
    ↓
Citation Formatter
    ↓
Streaming Response
    ↓
Persistent Storage
```

## Tech Stack

### Language

- Python 3.11+

### UI

- Streamlit

### Backend/API

- FastAPI

### Web Search

- Tavily API

### HTTP Fetching

- httpx, async

### Content Extraction

- trafilatura
- BeautifulSoup4 as fallback/manual parser

### RAG/Retrieval

- sentence-transformers
- FAISS for dense cosine similarity
- BM25 for lexical retrieval, using `rank-bm25` or an equivalent lightweight in-project implementation

### Database/Persistence

- SQLite

### Token/Context Management

- tiktoken

### Async Orchestration

- asyncio

### Evaluation

- pandas
- numpy
- matplotlib

### Config Management

- python-dotenv

### Testing

- pytest

## RAG Design

1. Fetch webpages from Tavily search results.
2. Extract readable article text.
3. Chunk documents into smaller overlapping chunks.
4. Generate embeddings for chunks.
5. Store normalized embeddings in FAISS for dense cosine similarity retrieval.
6. Build a BM25 lexical index over the same chunk texts.
7. Store metadata in SQLite:
   - chunk_id
   - source_url
   - title
   - domain
   - chunk_text
   - BM25 tokenized text or enough raw text to rebuild the BM25 index
   - timestamps
   - session_id
8. On each query:
   - embed query
   - retrieve candidate chunks using FAISS cosine similarity
   - retrieve candidate chunks using BM25 lexical scoring
   - combine both candidate sets using reciprocal rank fusion or weighted score fusion
   - retrieve relevant previous conversation turns
   - build final context
   - generate grounded answer with citations

## Session And Memory Requirements

Persist:

- session_id
- user messages
- assistant messages
- timestamps

Per-turn storage:

- user query
- search queries issued
- URLs opened
- retrieved snippets
- final answer
- citations
- timestamp

Long conversation support:

- rolling summaries
- retrieval of relevant prior turns
- max context enforcement
- summarization fallback

## Citation Requirements

Every factual answer should cite sources like:

```text
[Title — domain]
```

or:

```text
(domain, URL)
```

If sources conflict:

- explicitly mention disagreement
- cite both sources

## Evaluation Harness Requirements

Create an evaluation dataset containing:

- factual questions
- comparison questions
- multi-hop questions
- conflicting-source questions
- insufficient-evidence questions
- multi-turn/session questions

The evaluation runner should:

- run the agent on the dataset
- store outputs
- store citations
- log intermediate artifacts
- produce summary metrics

Metrics should evaluate:

- grounding quality
- citation quality
- answer usefulness
- uncertainty handling
- retrieval relevance for dense, BM25, and fused retrieval
- robustness across sessions

## Phase 1: Project Foundation

### Goal

Create a clean module structure that can support research orchestration without becoming a tangled chatbot wrapper.

Suggested structure:

```text
app/
  api/
    routes.py
    schemas.py
  core/
    config.py
    logging.py
  research/
    agent.py
    planner.py
    search.py
    fetcher.py
    extractor.py
    chunker.py
    embeddings.py
    vector_store.py
    bm25_store.py
    retriever.py
    context_builder.py
    generator.py
    citations.py
  memory/
    db.py
    sessions.py
    history.py
    summaries.py
  streaming/
    events.py
  evaluation/
    dataset.py
    runner.py
    metrics.py
  ui/
    streamlit_app.py
tests/
data/
  research.db
  faiss/
```

Install core dependencies:

```text
fastapi
uvicorn
streamlit
httpx
trafilatura
beautifulsoup4
sentence-transformers
faiss-cpu
rank-bm25
pydantic
python-dotenv
tiktoken
pytest
pandas
numpy
matplotlib
```

SQLite can be implemented using the Python standard library `sqlite3`. A helper package such as `sqlite-utils` may be added only if it clearly simplifies development without hiding important persistence behavior.

### Why This Comes First

The project needs clear ownership boundaries. If orchestration, retrieval, persistence, and UI logic are mixed early, later evaluation and memory support become painful.

### Milestone

- App structure exists.
- Config loads from `.env`.
- SQLite connection works.
- Basic FastAPI health route works.

### Pitfalls

- Starting with Streamlit before the backend/research pipeline exists.
- Mixing API, retrieval, and UI concerns in one file.
- Adding framework abstractions that hide the manual orchestration being evaluated.

## Phase 2: Configuration And Data Models

### Goal

Define the contracts before implementing behavior.

Create config for:

```text
TAVILY_API_KEY
LLM_API_KEY
LLM_PROVIDER
EMBEDDING_MODEL_NAME
DATABASE_PATH
FAISS_INDEX_PATH
BM25_INDEX_PATH
HYBRID_DENSE_TOP_K
HYBRID_BM25_TOP_K
HYBRID_RRF_K
MAX_FETCHED_PAGES
MAX_CHUNKS_PER_QUERY
MAX_CONTEXT_TOKENS
FETCH_TIMEOUT_SECONDS
MAX_CONCURRENT_FETCHES
```

Define shared Pydantic/domain models:

- SearchResult
- FetchedPage
- ExtractedDocument
- DocumentChunk
- RetrievedChunk
- Citation
- ResearchTurn
- ResearchEvent
- AgentAnswer

### Why This Comes Before Search And Fetch

Every later module should pass structured objects, not loose dictionaries. This minimizes refactoring when adding metadata, citations, scores, or evaluation logs.

### Milestone

- Shared schemas exist.
- Unit tests validate required fields and basic serialization.

### Pitfalls

- Under-modeling metadata early.
- Failing to track source rank, fetch status, extraction status, retrieval score, and citation identifiers.

## Phase 3: SQLite Persistence Layer

### Goal

Store sessions, messages, turns, sources, chunks, citations, and evaluation artifacts.

Recommended tables:

```text
sessions
  id
  title
  created_at
  updated_at
  rolling_summary

messages
  id
  session_id
  role
  content
  created_at

research_turns
  id
  session_id
  user_query
  search_queries_json
  opened_urls_json
  final_answer
  created_at

sources
  id
  session_id
  turn_id
  url
  title
  domain
  fetched_at
  extraction_status

chunks
  id
  source_id
  session_id
  turn_id
  chunk_text
  bm25_text
  chunk_index
  token_count
  embedding_id

citations
  id
  turn_id
  source_id
  chunk_id
  label
  url
  title
  domain

retrieval_logs
  id
  turn_id
  query
  chunk_id
  retrieval_method
  score
  dense_score
  bm25_score
  fused_score
  rank
```

### Why This Comes Early

Persistence is not an add-on. Memory, citations, evaluation, and debugging all depend on records being saved consistently.

### Milestone

- Can create a session.
- Can save and retrieve messages.
- Can save sources, chunks, citations, and retrieval logs.
- Tests cover basic database operations.

### Pitfalls

- Treating FAISS as the only source of truth.
- Treating BM25 as an in-memory-only artifact that cannot be rebuilt from stored chunks.
- Not storing failed fetches/extractions.
- Making it impossible to reconstruct why an answer was generated.

## Phase 4: Search Layer

### Goal

Implement Tavily search as an isolated service.

Responsibilities:

- Accept a user query or planned search query.
- Return normalized `SearchResult` objects.
- Deduplicate URLs.
- Track search query text and original result rank.
- Handle API errors gracefully.

Suggested behavior:

- Start with 1 to 3 search queries per user question.
- Retrieve 5 to 10 results total.
- Filter obvious low-quality URLs only when the heuristic is clear and explainable.

### Why This Comes Before Fetching

The fetcher depends on normalized URLs and titles from search results.

### Milestone

- Given a query, returns structured search results.
- Search queries are logged into the turn record.

### Pitfalls

- Over-searching too early.
- Not preserving original Tavily rank.
- Failing the whole turn if one search request fails.

## Phase 5: Async Page Fetcher

### Goal

Fetch webpages concurrently using `httpx`.

Responsibilities:

- Async fetch multiple URLs.
- Set timeouts.
- Respect max page count.
- Capture status code, final URL, content type, and raw HTML.
- Fail per page, not per turn.

Recommended defaults:

```text
timeout: 10-15 seconds
max concurrent fetches: 5
skip non-HTML content initially
```

### Why This Comes Here

Extraction needs actual HTML. Also, live web fetching is failure-prone, so this module must be stable before RAG work.

### Milestone

- Can fetch 5 pages concurrently.
- Failed URLs are recorded without crashing the full research run.

### Pitfalls

- Hanging on slow sites.
- Treating PDFs, JavaScript-heavy sites, and blocked pages as normal pages.
- Losing redirected final URLs.

## Phase 6: Content Extraction

### Goal

Convert HTML into readable text.

Primary extractor:

- `trafilatura.extract`

Fallback extractor:

- BeautifulSoup4
- remove scripts, styles, nav, footer, and obvious boilerplate
- extract headings, paragraphs, and list items
- normalize whitespace

Store:

- extracted text
- title
- domain
- extraction status
- word/token count

### Why This Comes After Fetching

This is the first quality gate. Bad extraction leads directly to bad retrieval and bad answers.

### Milestone

- Extract readable text from typical articles.
- Reject pages with too little content.
- Store successful sources in SQLite.

### Pitfalls

- Keeping boilerplate navigation text.
- Allowing tiny snippets as evidence.
- Not recording extraction failures, which hurts evaluation and debugging.

## Phase 7: Chunking

### Goal

Split extracted documents into retrievable evidence units.

Recommended approach:

- Token-aware chunking with `tiktoken`.
- Chunk size: 400 to 800 tokens.
- Overlap: 80 to 150 tokens.
- Preserve source metadata on every chunk.

Each chunk should know:

```text
chunk_id
source_url
title
domain
chunk_index
text
token_count
```

### Why This Comes Before Embeddings

Embedding quality depends heavily on chunk size. Too large gives vague retrieval; too small loses context.

### Milestone

- Documents become stable chunks.
- Chunks are stored in SQLite.
- Tests cover overlap and token limits.

### Pitfalls

- Character-only chunking.
- Losing source/title metadata during chunking.
- Producing chunks that start and end mid-sentence too often.

## Phase 8: Hybrid Retrieval Store

### Goal

Index chunks for both dense semantic retrieval and lexical retrieval.

Use:

- `sentence-transformers`
- FAISS `IndexFlatIP` with normalized vectors for cosine similarity
- BM25 over normalized chunk text for exact-term and keyword-sensitive retrieval
- reciprocal rank fusion as the default way to combine dense and BM25 candidates

Responsibilities:

- Load embedding model once.
- Embed chunks in batches.
- Store vector IDs mapped to SQLite chunk IDs.
- Save/load FAISS index from disk.
- Build, save, and rebuild the BM25 index from stored chunk text.
- Retrieve dense candidates with cosine similarity.
- Retrieve lexical candidates with BM25.
- Fuse both candidate lists into a single ranked set.
- Support per-session or global retrieval filtering.

Important design decision:

- Start with a single FAISS index plus SQLite metadata.
- Start with a single BM25 index over the same chunk corpus.
- Store `embedding_id -> chunk_id`.
- Store `bm25_doc_id -> chunk_id`.
- Filter retrieved chunks by session, turn, or source after FAISS and BM25 lookup.
- Use reciprocal rank fusion first because FAISS cosine scores and BM25 scores are not naturally comparable. Add weighted score fusion later only after score normalization is validated.

### Why This Comes Now

Retrieval is the core of the system. Do not build final generation until evidence selection works.

### Milestone

- Can embed chunks.
- Can retrieve top-k chunks using dense cosine similarity.
- Can retrieve top-k chunks using BM25.
- Can produce a fused top-k result set with source metadata and per-method scores.
- Can map retrieved dense and BM25 candidates back to text and source metadata.

### Pitfalls

- FAISS index and SQLite metadata getting out of sync.
- BM25 document IDs and SQLite chunk IDs getting out of sync.
- Rebuilding the full index every query.
- Not normalizing vectors consistently.
- Comparing raw cosine and BM25 scores directly without rank fusion or score normalization.

## Phase 9: Memory System

### Goal

Support persistent multi-turn conversations.

Implement:

1. Session history storage.
2. Relevant memory retrieval.
3. Rolling summaries when conversation history exceeds a token threshold.

Memory retrieval can start simple:

- keyword retrieval over previous turns
- embedding-based retrieval over previous turns once the vector store path is stable

### Why This Comes After Core Hybrid Retrieval

Source evidence should remain primary. Memory should help continuity, not replace web grounding.

### Milestone

- Sessions persist across app restarts.
- Agent can refer to relevant prior turns.
- Old history gets summarized when needed.

### Pitfalls

- Letting memory override fresh web evidence.
- Injecting too much conversation history into context.
- Failing to distinguish user preferences/history from factual evidence.

## Phase 10: Context Builder

### Goal

Construct the final prompt context from retrieved evidence and memory.

Responsibilities:

- Select top chunks from fused dense and BM25 retrieval results.
- Enforce source diversity.
- Enforce token budget.
- Include citation labels.
- Include relevant prior turns separately from evidence.
- Preserve source metadata.

Suggested prompt structure:

```text
System instruction:
  You are a deep research assistant. Answer only from evidence.

User question:
  ...

Relevant memory:
  ...

Evidence:
  [S1] Title, domain, URL
  chunk text...

  [S2] Title, domain, URL
  chunk text...

Answer requirements:
  cite claims using [S1], [S2]
  mention uncertainty
  do not invent sources
```

### Why This Comes Before Generation

Grounded answering depends more on context construction than model cleverness.

### Milestone

- Given retrieved chunks and memory, produces a bounded prompt.
- Drops lower-value chunks when token budget is exceeded.
- Maintains citation mapping.

### Pitfalls

- Stuffing too many chunks.
- Duplicating chunks from the same source.
- Mixing source text and memory without labels.

## Phase 11: Answer Generation

### Goal

Generate grounded answers with citations.

Responsibilities:

- Use selected evidence only.
- Cite factual claims.
- Mention insufficient evidence when needed.
- Mention conflicting evidence when detected.
- Return structured output:
  - final answer
  - citations
  - uncertainty notes
  - used source IDs

The generator should not fetch, search, or retrieve. It should only consume context.

### Why This Comes Here

Generation should be the last part of the research pipeline, not the first.

### Milestone

- Produces cited answers from retrieved evidence.
- Refuses or qualifies answers when evidence is weak.
- Saves answer and citations to SQLite.

### Pitfalls

- Prompt allowing uncited factual claims.
- Model citing unused sources.
- Hiding uncertainty.

## Phase 12: Manual Orchestrator

### Goal

Implement the actual research agent flow without frameworks.

`ResearchAgent.run(query, session_id)` should orchestrate:

```text
1. emit "Planning research..."
2. create visible plan
3. emit "Searching web..."
4. run Tavily searches
5. emit "Fetching sources..."
6. fetch pages
7. emit "Extracting readable content..."
8. extract text
9. emit "Indexing evidence..."
10. chunk + embed + store
11. emit "Selecting relevant evidence..."
12. retrieve chunks using hybrid dense/BM25 retrieval + memory
13. emit "Generating grounded answer..."
14. generate answer
15. emit "Saving research turn..."
16. persist everything
17. stream final answer
```

Use async generators for streaming events:

```python
async def run_research(query, session_id):
    yield ResearchEvent(type="status", message="Planning research...")
    ...
    yield ResearchEvent(type="answer_delta", content=chunk)
    yield ResearchEvent(type="done", data=final)
```

### Why This Comes After Individual Modules

Orchestration should compose tested parts. This is also the assignment's main evaluation surface.

### Milestone

- One query runs end-to-end.
- Intermediate progress events stream.
- Turn artifacts are saved.

### Pitfalls

- Putting business logic directly in FastAPI or Streamlit.
- Making the orchestrator too model-dependent.
- Continuing after zero usable sources without telling the user.

## Phase 13: FastAPI Backend

### Goal

Expose sessions and research runs.

Recommended routes:

```text
POST /sessions
GET /sessions
GET /sessions/{session_id}
POST /sessions/{session_id}/messages
POST /research/stream
GET /sessions/{session_id}/turns
GET /turns/{turn_id}/citations
```

For streaming:

- Use Server-Sent Events.
- Stream status events, answer deltas, citations, and done events.

### Why This Comes After The Orchestrator

The API should wrap the agent, not define the agent.

### Milestone

- Can create a session.
- Can send a research query.
- Can stream progress and answer.
- Can reload previous sessions.

### Pitfalls

- Blocking the async event loop with embedding/model calls.
- Not handling client disconnects.
- Returning only final answer and losing operational transparency.

## Phase 14: Streamlit UI

### Goal

Provide a usable research assistant interface.

Core UI:

- Session selector/sidebar.
- Chat-style research conversation.
- Visible progress updates.
- Final answer with citations.
- Expandable source list.
- Prior session loading.

Recommended layout:

```text
Left: sessions
Main: conversation + progress + answer
Bottom: query input
Optional: source/citation panel
```

### Why This Comes After Backend

The UI should consume the real API flow. Avoid building a fake local-only demo that later needs rewriting.

### Milestone

- User can start/resume a session.
- User sees intermediate progress.
- User sees cited final answers.

### Pitfalls

- UI calling internal modules directly.
- Not displaying failed or ignored sources.
- Hiding citations in raw JSON.

## Phase 15: Evaluation Harness

### Goal

Prove the research system works beyond demos.

Create a dataset with:

```text
question
category
expected_behavior
reference_answer_optional
requires_current_web
requires_conflict_handling
requires_insufficient_evidence
session_setup_optional
```

Categories:

- factual
- comparison
- multi-hop
- conflicting-source
- insufficient-evidence
- multi-turn/session
- recent/current information

Runner should:

- create or reuse sessions
- run the agent on each item
- store final answer, citations, retrieved chunks, search queries, and opened URLs
- export CSV/JSON results

Metrics:

- citation presence
- citation relevance
- answer groundedness
- dense retrieval relevance
- BM25 retrieval relevance
- fused retrieval relevance
- uncertainty handling
- source diversity
- latency
- fetch success rate

### Why This Comes Near The End

Evaluation needs the real pipeline and stored artifacts.

### Milestone

- `python -m app.evaluation.runner` runs the dataset.
- Produces result files and summary metrics.
- Logs enough artifacts for manual inspection.

### Pitfalls

- Evaluating only final answers.
- Ignoring retrieval quality.
- No tests for insufficient evidence cases.

## Phase 16: Testing Strategy

### Goal

Add focused tests for the core research pipeline.

Priority tests:

- chunking token limits
- extraction fallback
- database persistence
- FAISS cosine mapping
- BM25 indexing and chunk mapping
- hybrid rank fusion behavior
- citation formatting
- context budget enforcement
- memory retrieval
- orchestrator behavior with failed fetches
- insufficient evidence path

Mock external APIs:

- Tavily
- LLM
- web fetches

### Why This Comes After Core Flow

At this point the real integration boundaries are known. Earlier unit tests still help, but integration tests become meaningful here.

### Milestone

- `pytest` passes.
- Core pipeline works with mocked external services.
- One optional live integration test can run when API keys exist.

## Phase 17: Hardening And Quality Improvements

### Goal

Improve retrieval quality, reliability, and operational behavior after the baseline system works.

Add:

- URL deduplication by canonical URL.
- Domain diversity constraints.
- Better query planning.
- Source quality heuristics.
- BM25 tokenizer tuning for acronyms, names, dates, and technical terms.
- Hybrid retrieval candidate-count and RRF tuning once evaluation data exists.
- Cached fetched pages.
- Retry logic.
- Rate-limit handling.
- Better conflict detection.
- More robust summarization.
- Per-session index cleanup.

### Why This Comes After Baseline

These are quality improvements. They should not delay the first working end-to-end research agent.

### Pitfalls

- Over-engineering before baseline works.
- Adding heuristics that silently remove useful sources.
- Making caching hide freshness problems.

## Phase 18: Optional Sarvam AI Integration

### Goal

Add Sarvam support modularly, mainly for multilingual use.

Create a model provider abstraction:

```python
class LLMProvider:
    async def generate(...)
    async def stream_generate(...)
```

Providers:

```text
OpenAIProvider or default provider
SarvamProvider
```

Use Sarvam optionally for:

- multilingual answer generation
- translation
- Indian language support

Do not tie retrieval, memory, or orchestration to Sarvam.

### Why This Comes Last

Sarvam is an enhancement, not the architecture. The assignment values research-agent design more than vendor coupling.

### Milestone

- Sarvam can be enabled via config.
- Existing pipeline still works without Sarvam.

## Recommended Development Order Summary

1. Project structure and config.
2. Shared data models.
3. SQLite persistence.
4. Tavily search.
5. Async page fetcher.
6. Text extraction.
7. Chunking.
8. Hybrid retrieval with embeddings, FAISS cosine similarity, BM25, and rank fusion.
9. Session memory.
10. Context builder.
11. Grounded generator.
12. Manual research orchestrator.
13. FastAPI streaming API.
14. Streamlit interface.
15. Evaluation harness.
16. Tests and integration checks.
17. Retrieval and memory hardening.
18. Optional Sarvam integration.

This order keeps the system honest: first make evidence acquisition reliable, then retrieval, then memory, then grounded generation, then UI and evaluation. That gives the project a real Deep Research Agent instead of a chat UI with search sprinkled around it.
