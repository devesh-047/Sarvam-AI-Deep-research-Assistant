import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Tavily
    tavily_api_key: str = ""

    # Database
    database_path: str = "data/research.db"

    # Fetcher
    fetch_timeout_seconds: int = 10
    max_concurrent_fetches: int = 5

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 100

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # FAISS
    faiss_index_path: str = "data/faiss/index.faiss"
    faiss_map_path: str = "data/faiss/id_map.json"

    # Retrieval
    retrieval_top_k: int = 6

    # LLM
    llm_provider: str = "gemini"          # "gemini" or "openai"
    llm_api_key: str = ""                 # OpenAI key
    gemini_api_key: str = ""              # Gemini key
    llm_model: str = "gemini-2.5-flash"
    llm_base_url: str = ""                # leave empty to use default endpoint
    max_context_tokens: int = 6000
    max_output_tokens: int = 4000
    gemini_thinking_budget: int = 0       # 0 to disable reasoning/thinking, or set a positive integer

    # Conversational Memory (Stage 3)
    max_recent_turns: int = 3          # number of verbatim turns to keep in context window
    max_summary_tokens: int = 400      # token budget for the rolling summary block
    summary_after_turns: int = 4       # threshold of turns after which we start summarizing (not strictly used but config ready)


    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
