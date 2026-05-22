# Stage 1 Status

## What Stage 1 implements
Stage 1 implements the foundational project structure and the research ingestion pipeline. It includes:
- Clean modular structure
- SQLite database initialization and simple repositories
- Tavily search wrapper
- Async webpage fetcher
- HTML content extractor (trafilatura with BeautifulSoup fallback)
- Basic Streamlit UI to run the pipeline

## Current architecture
```
app/
  core/
  models/
  research/
  memory/
  ui/
tests/
```

## What should currently work
1. Querying via the Streamlit UI.
2. Searching Tavily (if API key is set).
3. Concurrently fetching URLs.
4. Extracting readable text.
5. Persisting session, turn, and source data in SQLite.
6. Displaying results in the UI.

## What is NOT implemented yet
- Retrieval (FAISS, BM25)
- Chunking and Embeddings
- LLM Answer Generation
- Memory Retrieval
- Streaming responses
- Advanced orchestrator logic

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
Once the virtual environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys and Environment Variables
The project uses environment variables for configuration. You need to create a `.env` file from the provided example:
```bash
cp .env.example .env
```

Next, open the newly created `.env` file in your text editor. It looks like this:
```env
TAVILY_API_KEY=your_tavily_api_key_here
DATABASE_PATH=data/research.db
FETCH_TIMEOUT_SECONDS=10
MAX_CONCURRENT_FETCHES=5
```

**Setting the Tavily API Key:**
1. Go to [Tavily](https://tavily.com/) and create an account or log in.
2. Navigate to the API dashboard to generate or copy your API key.
3. Replace `your_tavily_api_key_here` in the `.env` file with your actual key. Make sure there are no spaces around the `=`.

*Note: The SQLite database will be automatically created at `DATABASE_PATH` when the app runs for the first time.*

### 4. Run the Application
Finally, start the Streamlit UI:
```bash
streamlit run app/ui/streamlit_app.py
```

## Testing Stage 1

### 1. Automated Testing (pytest)
We use `pytest` with `pytest-asyncio` to test asynchronous page fetching, database initialization, and content extraction.

To run the full suite of automated tests:
```bash
pytest tests/ -v
```

This will run:
*   `test_db_initialization`: Verifies that `init_db()` correctly creates `sessions`, `messages`, `research_turns`, and `sources` tables in SQLite.
*   `test_session_repository`: Validates that sessions are inserted and retrieved properly.
*   `test_search_no_api_key`: Tests that the Tavily search module fails gracefully when no API key is set.
*   `test_fetch_error_handling`: Confirms the async page fetcher catches connection errors properly.
*   `test_extractor_fallback`: Validates content extraction logic and BeautifulSoup fallback.

### 2. Manual Testing & Verification
You can manually run and inspect the end-to-end flow to verify SQLite persistence.

1.  Run the Streamlit application:
    ```bash
    streamlit run app/ui/streamlit_app.py
    ```
2.  Open the Streamlit UI, enter a query (e.g., `"Recent space missions in 2026"`), and click **Run Research**.
3.  Ensure the search results, fetch status, and extracted text previews appear correctly.
4.  Open a terminal to inspect the SQLite database file (`data/research.db` by default) to verify the data was saved:
    ```bash
    sqlite3 data/research.db "SELECT * FROM sessions;"
    sqlite3 data/research.db "SELECT * FROM research_turns;"
    sqlite3 data/research.db "SELECT * FROM sources;"
    ```
    *If you don't have `sqlite3` installed, you can use a quick Python snippet to print the data:*
    ```bash
    python -c "import sqlite3; conn = sqlite3.connect('data/research.db'); cursor = conn.cursor(); print(cursor.execute('SELECT * FROM research_turns').fetchall())"
    ```


## Example workflow
1. Open Streamlit UI.
2. Enter a query like "Latest advancements in AI agents".
3. Click "Run Research".
4. Pipeline searches, fetches, extracts, stores, and displays text previews.

## Known limitations
- Fails gracefully but silently if Tavily API key is missing (returns no results).
- Very complex sites may resist extraction.
- Does not yet answer the user's question, only fetches the context.

## Next planned stage
Stage 2: Chunking, Embeddings, and Hybrid Retrieval (FAISS + BM25).
