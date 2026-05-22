"""
Conversation memory helpers (Stage 3).

Formats recent research turns from SQLite as a token-bounded string block.
"""
from typing import List
from app.models.schema import ResearchTurn
from app.core.logging import get_logger
import tiktoken

logger = get_logger(__name__)

def format_memory_block(recent_turns: List[ResearchTurn], max_tokens: int) -> str:
    """
    Formats turns into:
      User: ...
      Assistant: ...
    Enforces the token budget (max_tokens) using tiktoken, keeping the most recent turns.
    """
    if not recent_turns:
        return ""
        
    encoding = tiktoken.get_encoding("cl100k_base")
    formatted_turns = []
    
    # Process from newest to oldest to respect the token budget for the most recent context
    for turn in reversed(recent_turns):
        turn_str = f"User: {turn.user_query}\nAssistant: {turn.final_answer or ''}\n\n"
        candidate_block = "".join(reversed(formatted_turns)) + turn_str
        if len(encoding.encode(candidate_block)) <= max_tokens:
            formatted_turns.append(turn_str)
        else:
            logger.info(f"[Memory] Exceeded budget of {max_tokens} tokens. Truncating older turns.")
            break
            
    if not formatted_turns:
        return ""
        
    return "".join(reversed(formatted_turns)).strip()
