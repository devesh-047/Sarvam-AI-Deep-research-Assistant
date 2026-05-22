# Final System Engineering Review

## 1. Architectural Strengths

The system is structured as a deterministic AI workflow rather than a generic chatbot wrapper. The core research path is explicit: planning, Tavily search, async fetching, extraction, chunking, retrieval, context construction, answer generation, citation formatting, memory update, and streaming workflow events.

Key strengths:

- Orchestration lives in `ResearchAgent` and does not depend on agent frameworks.
- Retrieval metadata remains in SQLite while FAISS stores only dense vectors.
- Streaming events expose operational progress without leaking hidden chain-of-thought.
- Conversation memory is separated from source evidence and injected as context, not as proof.
- Multilingual support is layered around the English research pipeline instead of changing retrieval semantics.
- Evaluation is reproducible and does not require judge-model calls for baseline metrics.

## 2. Evaluation Criteria Alignment

The evaluation harness now better reflects the assignment goals:

- **Grounding:** measured through citation presence and sentence-level grounding ratio.
- **Citation integrity:** measured through citation label integrity and citation source traceability.
- **Retrieval quality:** represented by hit count, source diversity, and retrieval method mix.
- **Uncertainty handling:** measured with explicit uncertainty language detection.
- **Conflict handling:** measured with conflict/disagreement language detection.
- **Operational reliability:** represented through error rate, latency, and per-category summaries.

These metrics are intentionally simple and defensible. They are not a substitute for human review, but they make regressions visible and force the system to report evidence behavior rather than only answer fluency.

## 3. Citation Integrity Analysis

Citations are assigned by the context builder from retrieved chunks. This is the right boundary: the model can cite labels only after the system has created a fixed evidence map.

The review hardened three important citation risks:

- Citations are now emitted only for evidence blocks that fit inside the actual prompt budget.
- Evaluation now checks whether answer labels such as `[S1]` exist in the citations payload.
- Evaluation now checks whether citation URLs came from retrieved chunks.

The multilingual layer also protects inline `[S#]` labels during localization and falls back to the original English answer if translation fails. This keeps localized answers accessible without severing citation traceability.

Remaining limitation: citation metrics verify structural integrity, not whether every cited sentence semantically supports the claim. That deeper check would require human review or a separate groundedness evaluator.

## 4. Retrieval And Context-Selection Analysis

Retrieval is now hybrid:

- FAISS with normalized vectors provides cosine-similarity semantic retrieval.
- BM25 lexical scoring protects exact terms, names, acronyms, dates, and rare entities.
- Reciprocal rank fusion combines both candidate lists without comparing raw score scales.

This improves robustness without replacing the existing dense retrieval architecture. The retriever still exposes the same `retrieve(query)` interface to the orchestrator, preserving the deterministic flow.

The context builder enforces a token budget and deduplicates citation labels by source URL. Evidence selection remains explainable through workflow events that include source metadata, fused score, retrieval method, dense score, lexical score, and preview text.

Conflict handling is currently lightweight and structural. It detects source diversity and official-vs-secondary source mixtures, then instructs the answer generator to surface disagreement. A future production version should add claim-level contradiction detection.

## 5. Session And Memory Analysis

Session state is persisted through SQLite research turns and rolling summaries. Recent turns are injected separately from source evidence, which is important: memory supports continuity, but retrieved web evidence remains the grounding authority.

Strengths:

- Sessions survive restarts through SQLite.
- Recent turns and rolling summaries are bounded by configuration.
- The evaluation runner separates conversational follow-ups into a shared session.

Remaining limitation: memory retrieval is mostly recency-based. Semantic retrieval over prior turns would improve long-running sessions, but should remain secondary to fresh source evidence.

## 6. Code Quality Analysis

The system has good modular boundaries:

- `research/` owns search, fetch, extraction, retrieval, context, generation, and orchestration.
- `memory/` owns SQLite persistence.
- `multilingual/` owns language detection, normalization, Sarvam calls, and localization.
- `evaluation/` owns repeatable measurement and reporting.
- `ui/` owns Streamlit presentation.

The hardening pass preserved this structure and avoided broad refactors. Improvements focused on local boundaries:

- Hybrid retrieval was added inside the existing retriever.
- Citation pruning was fixed inside the context builder.
- Localization citation preservation was fixed inside the multilingual layer.
- Evaluation metrics and reports were made more interpretable.

Remaining limitation: `ResearchAgent.stream_run` is still large. It is readable because the pipeline is linear, but a future cleanup could extract private step helpers without changing behavior.

## 7. Reliability And Error-Handling Analysis

The system already degrades gracefully in several places:

- Tavily search has retry behavior.
- Fetching fails per URL rather than per turn.
- Extraction failures are counted and reported.
- FAISS load failure falls back to an empty index.
- Streaming generation falls back to non-streaming generation.
- Sarvam failures fall back to English output or raw query input.

The hybrid retriever adds another reliability layer: if FAISS is empty or unavailable, BM25 can still return lexical evidence from stored chunks.

Remaining limitation: live API behavior still depends on external credentials, network availability, rate limits, and model availability. The dry-run evaluation path is appropriate for CI, while live evaluation should be run before final demos.

## 8. Remaining Limitations

- Claim-level conflict detection is heuristic rather than semantic.
- Evaluation metrics are deterministic proxies and should be supplemented with manual review.
- Memory retrieval is recency-based rather than deeply semantic.
- The Streamlit app is suitable for demo and assignment use, but a production deployment would need authentication, observability, and rate limiting.
- BM25 currently rebuilds from SQLite chunks at retrieval time. This is simple and reliable for the assignment scale; larger deployments should persist an optimized lexical index.

## 9. Final Engineering Rationale

The system is strongest where it matters for a deep research assignment: clear orchestration, source acquisition, retrieval, bounded context construction, grounded synthesis, visible workflow events, memory persistence, and measurable evaluation artifacts.

The hardening changes were intentionally incremental. They improve evidence robustness, citation integrity, evaluation credibility, and multilingual safety without redesigning the architecture or introducing framework dependencies.

## 10. Why Deterministic Orchestration Was Chosen

Deterministic orchestration was chosen because the evaluation target is the research workflow itself, not autonomous agent improvisation. Explicit steps make the system:

- reproducible
- debuggable
- inspectable
- easier to evaluate
- easier to recover from partial failures
- safer for citation-grounded synthesis

This is the right tradeoff for a production-style research assistant.

## 11. Why Multilingual Retrieval Was Intentionally Avoided

The system normalizes multilingual queries into English for retrieval and localizes final answers afterward. This is intentional.

Reasons:

- Most Tavily/web source coverage and extracted evidence will be stronger in English for broad research topics.
- A single retrieval language keeps embeddings, BM25 tokenization, context construction, and evaluation more stable.
- Citation labels and source URLs remain unchanged across localization.
- The multilingual layer becomes an accessibility layer rather than a second, harder-to-evaluate retrieval pipeline.

Future multilingual retrieval can be added later for region-specific questions, but it should be evaluated separately.

## 12. Production-Readiness Discussion

The project is assignment-ready and demonstrates production-style discipline. It has explicit module boundaries, graceful fallback behavior, persistent sessions, visible workflow events, citation-aware generation, hybrid retrieval, and an evaluation harness with interpretable metrics.

Before true production deployment, add:

- structured request tracing and observability
- rate limiting and authentication
- migration-managed SQLite or a production database
- persisted lexical index for large corpora
- stronger live monitoring for API failures
- human-reviewed evaluation sets
- claim-level support verification

Within the project constraints, the system is polished, maintainable, evaluation-driven, and architecturally sound.
