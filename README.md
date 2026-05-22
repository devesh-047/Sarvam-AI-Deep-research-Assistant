# Deep Research Assistant

A grounded, deterministic web research assistant that answers complex queries using live internet evidence, hybrid retrieval, and multi-step reasoning. Built from scratch in Python, this system explicitly avoids black-box recursive frameworks (like LangChain or AutoGPT) in favor of a strictly observable, citation-backed orchestration pipeline.

It natively understands Romanized Indic languages (Hinglish/Benglish) and provides localized responses via Sarvam AI, while maintaining a robust English-centric retrieval core for maximum evidence quality.

**Repository:** [github.com/devesh-047/Sarvam-AI-Deep-research-Assistant](https://github.com/devesh-047/Sarvam-AI-Deep-research-Assistant)

---

## 2. Video Demo

🎥 **Watch the Demo:** [Google Drive Link](https://drive.google.com/file/d/1Z8TgTxgzwa-NJiqhtAQIzcYak8-SKSLh/view?usp=sharing)

The demo showcases:
- **Planning:** The system formulating a multi-query search strategy.
- **Research Workflow:** Real-time ReAct-style visibility into searching, fetching, extracting, and retrieving.
- **Citations:** Grounded synthesis with clickable source URLs and domain attribution.
- **Multilingual Flow:** Querying in Hinglish and receiving localized outputs with preserved English citations.
- **Conversational Memory:** The system successfully resolving pronouns (e.g., "it") from prior turns.
- **Streaming Responses:** Token-by-token answer generation for a responsive UX.

---

## 3. Features

- **Grounded Web Research:** Uses Tavily to fetch live internet sources dynamically.
- **Citation-Aware Answers:** Forces the LLM to synthesize only from retrieved chunks, appending explicit inline citations (e.g., `[S1]`).
- **ReAct-Style Workflow Visibility:** Exposes the agent's internal state (Thought, Action, Observation) securely to the UI.
- **Conversational Memory:** Preserves multi-turn context using SQLite and rolling summaries.
- **Multilingual Accessibility:** Detects Romanized Indic queries and provides translated/transliterated flows via Sarvam AI.
- **Streaming Responses:** Smooth, ChatGPT-inspired UI streaming directly into the Streamlit frontend.
- **Evaluation Harness:** A standalone, repeatable offline test suite measuring citation integrity, retrieval diversity, and conflict handling.
- **Conflict-Aware Synthesis:** Detects divergent domains (e.g., official `.gov` vs. commercial `.com`) and prompts the LLM to expose the disagreement.

---

## 4. Design Note (Part 1 Requirement)

### 4A. Target Users & Problem
**Users:** Analysts, students, developers, and researchers who need verified, up-to-date information without the risk of LLM hallucination.
**Problem:** Standard LLMs hallucinate facts, lack current events knowledge, and fail to cite verifiable sources. Conversely, autonomous AI agents (like AutoGPT) often get trapped in infinite search loops and fail unpredictably. 
**Solution:** A grounded research assistant that strictly couples every generated claim to a dynamically fetched, user-verifiable source document, orchestrated deterministically.

### 4B. What "Deep Research" Means In This Project
"Deep Research" is defined as a multi-step evidence acquisition pipeline:
1. Deconstructing a query into a structured web-search plan.
2. Acquiring raw HTML from multiple sources.
3. Extracting, chunking, and embedding the text.
4. Selecting the most relevant passages via hybrid scoring (Dense Cosine + BM25 Lexical).
5. Synthesizing a final answer that structurally resolves conflicting evidence and explicitly cites its origins.

### 4C. Success Metrics
- **Citation Integrity:** Every claim must map to a retrieved chunk. Verified by checking if the LLM output contains `[S#]` tags that match the retrieval payload.
- **Grounding Quality:** Evaluated by ensuring the LLM doesn't introduce external knowledge outside the provided context window.
- **Retrieval Relevance:** Measured by source diversity (unique domains) and the presence of high-scoring lexical/dense hits.
- **Conversational Continuity:** The ability to answer follow-up questions accurately using memory blocks.
- **Conflict & Uncertainty Handling:** The system's ability to explicitly state "Sources disagree" or "Insufficient evidence found" instead of guessing.

### 4D. Data Flow & Architecture

1. **User Query:** Received via Streamlit.
2. **Planner:** LLM generates a visible "Thought" and 1–3 targeted Tavily search queries.
3. **Search:** Tavily API returns candidate URLs.
4. **Fetch:** Async HTTP fetches raw HTML for new URLs.
5. **Extract:** Boilerplate removal to extract clean readable text.
6. **Chunk:** Documents are split into overlapping passages.
7. **Embed:** Passages are converted to vectors (Sentence-Transformers).
8. **Retrieve:** Hybrid search (FAISS + BM25) selects the top evidence chunks.
9. **Context Build:** Evidence, past memory, and system prompts are assembled.
10. **Grounded Generation:** LLM streams the answer using *only* the provided context.
11. **Citations:** Source metadata is appended to the UI.
12. **Memory Persistence:** SQLite stores the turn, sources, chunks, and rolling summary.

### 4E. Why Deterministic Orchestration Was Chosen
**Per the assignment instructions, framework dependencies like LangChain, CrewAI, or LlamaIndex were explicitly avoided.**
Furthermore, building uncontrolled ReAct recursive loops was intentionally avoided. Recursive autonomous agents are notoriously difficult to evaluate, prone to infinite loops, and opaque to debug. By enforcing a **deterministic orchestration pipeline**, the system guarantees:
- **Grounding:** The LLM cannot bypass the retrieval step.
- **Reproducibility:** The pipeline steps always execute in the same order.
- **Evaluation:** Every intermediate artifact (chunks, embeddings, retrieved lists) can be isolated, logged, and scored.
- **Maintainability:** Clear Python module boundaries instead of opaque framework abstractions.

*Note: While the backend execution avoids uncontrolled recursive looping, **ReAct-style workflow visibility** is still provided in the UI (exposing "Thought", "Action", and "Observation" milestones) to maintain a highly transparent and engaging user experience.*

### 4F. Risks & Limitations
*Honest assessment of the current system constraints:*
1. **Multilingual Limitations:** While Hinglish understanding is good, Sarvam AI translation quality can occasionally alter nuanced technical terminology or translation fails altogether.
2. **English-Centric Retrieval:** The system normalizes queries to English for search and retrieval. Pure native-script queries may suffer if highly local cultural context is lost in translation.
3. **Latency:** Sequential fetching, embedding, and LLM API calls result in ~10-20 second wait times before streaming begins.
4. **Tavily Dependency:** Web search quality is entirely bottlenecked by Tavily's indexing and scraping permissions.
5. **Context Window & LLM Token Limits:** Extremely dense research topics or long-running conversations may exceed LLM context window capacities and prompt token budgets, forcing truncation of relevant evidence.
6. **Lexical Limits:** BM25 is currently recalculated in-memory per turn; it does not persist a global inverted index.
7. **Hallucinated Citations:** While rare, the LLM can occasionally misattribute a fact to the wrong `[S#]` label if multiple chunks contain similar keywords.

### 4G. Future Improvements
- **Streamlit Scalability:** Streamlit is used for the frontend. While highly effective for prototyping, it suffers from state-management rigidity. A future migration to React/Next.js and FastAPI is recommended for scaling to thousands of users.
- **Database Scalability:** SQLite and FAISS are currently used locally. For distributed clustering, migration to a dedicated vector database (e.g., Qdrant/Milvus) and a relational database (e.g., PostgreSQL) is required.
- **Prompt Management:** Several system prompts reside directly within `planner.py` and `query_normalizer.py`. Migrating these to structured configuration files or a CMS would improve codebase maintainability.
- **True Multilingual Retrieval:** Indexing chunks in their native languages and using cross-lingual embedding models (e.g., multilingual-e5) instead of translating queries.
- **Adaptive Reranking:** Implementing a Cross-Encoder step after the hybrid retrieval to heavily penalize irrelevant chunks.
- **Source Trust Scoring:** Automatically down-weighting blogs/commercial sites in favor of `.edu`/`.gov` domains during the retrieval fusion step.
- **Async Distributed Retrieval:** Offloading the Fetch/Extract/Embed steps to Celery workers for massively parallel execution.

---

## 5. Architecture Overview

```text
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│                 │       │                  │       │  Sarvam AI      │
│  Streamlit UI   │ ◄───► │  Multilingual    │ ◄───► │  Translation/   │
│                 │       │  Normalizer      │       │  Localization   │
└────────┬────────┘       └────────┬─────────┘       └─────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Orchestrator (agent.py)                       │
└────┬─────────┬─────────┬─────────┬──────────┬──────────┬─────────┬───┘
     │         │         │         │          │          │         │    
     ▼         ▼         ▼         ▼          ▼          ▼         ▼    
 ┌──────┐  ┌──────┐  ┌───────┐ ┌───────┐ ┌────────┐  ┌───────┐ ┌───────┐
 │ Plan │  │Search│  │ Fetch │ │Extract│ │ Chunk/ │  │ Hybrid│ │ LLM   │
 │(LLM) │  │(Web) │  │(Async)│ │(Text) │ │ Embed  │  │ Search│ │ Gen   │
 └──────┘  └──────┘  └───────┘ └───────┘ └────────┘  └───────┘ └───────┘
                                             │           │              
                                             ▼           ▼              
                                         ┌───────┐   ┌───────┐          
                                         │ FAISS │   │ SQLite│          
                                         └───────┘   └───────┘          
```

---

## 6. Agentic Workflow Explanation

The system is **agentic** because it actively plans its own tool usage. When a user asks a question, the LLM formulates independent web queries (e.g., transforming "Who won the 2024 elections?" into "2024 election official results"). 

However, the orchestration of those tools is **deterministic**. Instead of giving the LLM an open-ended `while` loop to execute arbitrary tools until it "feels" done, the Python backend rigorously enforces the sequence: *Plan → Search → Fetch → Extract → Retrieve → Synthesize*. 

This internal workflow is exposed securely to the user via **ReAct-style visibility**. The UI renders safe, pre-scripted milestone summaries (e.g., "Thought: Analyzing question...", "Action: Searching the web for...") to provide transparency without exposing raw, brittle chain-of-thought tokens.

---

## 7. Multilingual Sarvam Layer

The system seamlessly handles Romanized Indic text (e.g., Hinglish, Benglish) and native scripts.

**Workflow:**
1. **Detection:** Heuristics detect if a query is non-English.
2. **Normalization:** "bharat me AI startups ka future kya hai" is sent to Sarvam API, which transliterates and translates it to: "What is the future of AI startups in India?"
3. **English-Centric Retrieval:** The translated English query is used to search the web, fetch English documents, and run hybrid retrieval. **Why?** Because the highest quality, most abundant technical web evidence exists in English.
4. **Localization:** The LLM generates the grounded answer in English. The text is then sent back to Sarvam API to localize it to the user's requested target language.
5. **Citation Preservation:** Critically, inline citations (`[S1]`) and URLs are computationally masked before localization to ensure translations don't destroy traceability.

---

## 8. Setup & Run Instructions

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/devesh-047/Sarvam-AI-Deep-research-Assistant.git
cd Sarvam-AI-Deep-research-Assistant

# 2. Set up Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Environment Variables
# Create a .env file based on .env.example
cp .env.example .env
```

**Required API Keys in `.env`:**
```ini
TAVILY_API_KEY=tvly-...
GEMINI_API_KEY=AIza...      # Or OPENAI_API_KEY
SARVAM_API_KEY=...          # For multilingual support
LLM_PROVIDER=gemini         # or openai
```

**Run the Application:**
```bash
streamlit run app/ui/streamlit_app.py
```

---

## 9. Project Structure

```text
.
├── app/
│   ├── core/           # Config, logging
│   ├── memory/         # SQLite persistence, conversational history, rolling summaries
│   ├── models/         # Pydantic schemas, data models
│   ├── multilingual/   # Sarvam translation wrappers, detection, normalization
│   ├── research/       # THE CORE: Agent orchestrator, planner, fetcher, chunker, hybrid retriever
│   └── ui/             # Streamlit frontend styling and streaming logic
├── data/               # Local SQLite db and FAISS indexes (ignored in git)
├── docs/               # Architecture reviews and cleanup documentation
├── evaluation/         # Offline evaluation harness and generated reports
└── tests/              # Pytest suite
```

---

## 10. Example Conversations

**Example 1: Standard Grounded Research**
- *User:* "What are the key architectural differences between LLaMA 3 and GPT-4?"
- *Agent Workflow:* Plans queries → Searches Tavily → Fetches technical blogs → Retrieves chunks → Generates a comparative answer citing specific parameters and design choices.

**Example 2: Follow-up (Conversational Memory)**
- *User:* "How much does it cost to train it?"
- *Agent Workflow:* Resolves "it" to "LLaMA 3" using SQLite memory → Searches for LLaMA 3 training compute costs → Synthesizes new evidence with previous context.

**Example 3: Romanized Indic (Hinglish)**
- *User:* "India me renewable energy goals 2030 ke kya hain?"
- *Agent Workflow:* Normalizes to "What are India's renewable energy goals for 2030?" → Retrieves English `.gov.in` reports → Localizes the final generated answer back into Hindi.

---

## 11. Evaluation Methodology

The evaluation harness (`evaluation/runner.py`) uses a curated offline dataset to measure the system without human intervention. 

**Categories Evaluated:**
- **Factual:** Can it find explicit facts and cite them?
- **Comparison:** Can it synthesize multiple sources?
- **Multi-hop / Conversational:** Can it use session memory to resolve ambiguous follow-up questions?
- **Conflicting Sources:** Can it detect and acknowledge when sources disagree?
- **Insufficient Evidence:** Does it confidently state when an answer cannot be found, avoiding hallucination?

These categories matter because they measure *behavioral safety* and *grounding capability*, not just fluency.

---

## 12. Evaluation Findings

*Based on internal offline reporting:*

- **Strengths:** Excellent citation integrity. The system rarely hallucinates claims and is highly reliable at injecting `[S#]` tags perfectly mapped to the retrieval payload. The deterministic orchestration prevents infinite loops.
- **Weaknesses:** Deep comparative queries (e.g., comparing 5 different frameworks) sometimes hit the strict token limit during context building, causing the LLM to miss nuanced differences.
- **Multilingual Limitations:** Translation round-tripping works well for general knowledge, but highly technical jargon is sometimes transliterated awkwardly by the Sarvam API.
- **Conflict Handling:** The system effectively detects structural domain diversity (e.g., a `.gov` and `.com` source) and adds warning caveats, but relies on the LLM to identify semantic contradictions.

---

## 13. How to Run Evaluation

The evaluation suite runs entirely offline against a curated JSON dataset.

```bash
# 1. Run the evaluation harness
python -m evaluation.runner

# 2. View the results
# The harness will output a Markdown report to:
cat evaluation/results/report.md
```
The report includes aggregated metrics (success rates, average citations) and detailed per-question logs tracing the exact retrieved chunks and scores.

---

## 14. Tech Stack

- **Python 3.10+**: Core language.
- **Streamlit**: Fast, reactive frontend UI.
- **Tavily API**: Dedicated web search API optimized for LLMs.
- **FAISS**: Local, high-performance dense vector similarity search.
- **Sentence-Transformers**: `all-MiniLM-L6-v2` for local, fast text embedding.
- **Gemini / OpenAI**: LLM generation via `google-genai` or standard `openai` SDK.
- **Sarvam AI**: State-of-the-art Indic language transliteration and translation.
- **SQLite**: Zero-dependency local persistence for conversation memory and chunk metadata.

Local SQLite and FAISS were chosen to ensure the project remains portable, simple to install, and entirely contained without requiring heavy Dockerized infrastructure.

---

## 15. Final Engineering Notes

Building an AI agent is easy; building a reliable one is hard. 

This project demonstrates that **deterministic orchestration** is vastly superior to unbounded autonomous loops (like early AutoGPT) for production-grade research tasks. By explicitly separating the planning, evidence acquisition, and generation phases, absolute control over the prompt context window is maintained. 

The biggest lesson learned is the importance of **hybrid retrieval**. Pure semantic search (FAISS) often smoothed over critical exact-match keywords (like specific years or acronyms). Adding a traditional BM25 scoring layer and fusing the ranks dramatically improved the evidence quality fed to the LLM. 

Ultimately, this architecture proves that high-quality, grounded, multilingual deep research can be achieved natively in Python, free from the bloat and opacity of heavy agentic frameworks.
