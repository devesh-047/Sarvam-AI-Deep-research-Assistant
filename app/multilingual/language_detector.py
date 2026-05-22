"""
Language Detector — Phase 5 Multilingual Layer.

Classifies the probable language / script of a user query using
lightweight heuristics. No external APIs. No ML model.
Fast, deterministic, graceful.

Detected categories:
  "english"          — standard English
  "hinglish"         — Romanized Hindi / Hindi-English mix
  "benglish"         — Romanized Bengali / Bengali-English mix
  "devanagari"       — Native Devanagari script (Hindi, Marathi, etc.)
  "bengali_script"   — Native Bengali script
  "south_indic"      — Tamil / Telugu / Kannada / Malayalam script
  "transliterated"   — Romanized Indic (generic, not definitively hindi/bengali)
  "unknown"          — fallback
"""
import re
from typing import Tuple

# ── Unicode block ranges ───────────────────────────────────────────────────────
_DEVANAGARI   = (0x0900, 0x097F)
_BENGALI      = (0x0980, 0x09FF)
_TAMIL        = (0x0B80, 0x0BFF)
_TELUGU       = (0x0C00, 0x0C7F)
_KANNADA      = (0x0C80, 0x0CFF)
_MALAYALAM    = (0x0D00, 0x0D7F)
_GUJARATI     = (0x0A80, 0x0AFF)
_GURMUKHI     = (0x0A00, 0x0A7F)

_INDIC_RANGES = [
    (_DEVANAGARI, "devanagari"),
    (_BENGALI,    "bengali_script"),
    (_TAMIL,      "south_indic"),
    (_TELUGU,     "south_indic"),
    (_KANNADA,    "south_indic"),
    (_MALAYALAM,  "south_indic"),
    (_GUJARATI,   "devanagari"),
    (_GURMUKHI,   "devanagari"),
]

# ── Romanized Indic keyword sets ───────────────────────────────────────────────
# Common Hinglish / Romanized Hindi words
_HINGLISH_WORDS = {
    "kya", "hai", "hain", "ka", "ke", "ki", "mein", "se", "ko", "ne",
    "ho", "tha", "thi", "nahi", "aur", "bhi", "yeh", "woh",
    "ap", "aap", "hum", "tum", "unka", "unke", "iska", "uska", "inke",
    "bharat", "desh", "sarkar", "log", "kuch", "sab", "bahut", "jab",
    "kyun", "kyunki", "isliye", "lekin", "magar", "phir", "abhi", "aaj",
    "kal", "agar", "toh", "sirf", "teri", "meri", "uski", "unki",
    "lagta", "lagti", "chahiye", "milega", "milegi", "padh", "likha",
    "accha", "theek", "sahi", "galat", "mujhe", "mera", "mere", "roz",
    "kitni", "kitna", "kitne", "baar", "din", "kaise", "kahan", "kab",
    "karna", "karo", "kare", "karti", "karta", "hoti", "hota", "hote",
    "kaun", "kisko", "wala", "wali", "wale", "jyada", "kam", "khana", "peena",
    "jana", "aana", "dena", "lena", "diya", "liya", "kaha", "suna", "dekh",
    "dekha", "karo", "kijiye", "raha", "rahi", "rahe", "hoga", "hogi", "hoge",
}

# Common Romanized Bengali words
_BENGLISH_WORDS = {
    "amar", "tomar", "amader", "tomader", "ache", "nei", "hobe", "korbo",
    "korbe", "kemon", "keno", "kothay", "ki", "ebar", "onek", "kintu",
    "tobe", "jodi", "tahole", "amra", "tara", "ekta", "duita", "notun",
    "purana", "desh", "manush", "bhalo", "kharap", "kolkata", "bangla",
    "bangladesh", "porichoy", "bazaar", "bazar", "kore", "kora", "niye",
    "nibe", "neta", "hoche", "hocche", "geche", "jacche", "khabe", "khabo",
    "kotha", "bolchi", "bolbo", "bolbe", "dibe", "dibo", "taka", "pabe", "pabo",
}

# ── Helper functions ──────────────────────────────────────────────────────────

def _count_indic_chars(text: str) -> Tuple[int, str]:
    """Return (count, script_name) for the most common Indic script in text."""
    counts: dict = {}
    for ch in text:
        cp = ord(ch)
        for (lo, hi), name in _INDIC_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return 0, "unknown"
    best = max(counts, key=counts.get)
    return counts[best], best


def _romanized_word_overlap(text_lower: str, word_set: set) -> int:
    """Count how many words from word_set appear in text."""
    words = re.findall(r"[a-z]+", text_lower)
    return sum(1 for w in words if w in word_set)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_language(query: str) -> str:
    """
    Detect the probable language/script of the input query.

    Returns one of:
      "english", "hinglish", "benglish", "devanagari",
      "bengali_script", "south_indic", "transliterated", "unknown"
    """
    if not query or not query.strip():
        return "unknown"

    # ── Native script detection (fastest path) ─────────────────────────────
    indic_count, script = _count_indic_chars(query)
    total_alpha = sum(1 for c in query if c.isalpha())
    if total_alpha == 0:
        return "unknown"
    indic_ratio = indic_count / total_alpha

    if indic_ratio > 0.3:
        return script  # "devanagari", "bengali_script", "south_indic"

    # ── Romanized Indic detection ─────────────────────────────────────────
    query_lower = query.lower()
    hinglish_hits = _romanized_word_overlap(query_lower, _HINGLISH_WORDS)
    benglish_hits  = _romanized_word_overlap(query_lower, _BENGLISH_WORDS)

    word_count = len(re.findall(r"[a-zA-Z]+", query))
    if word_count == 0:
        return "unknown"

    hinglish_ratio = hinglish_hits / word_count
    benglish_ratio  = benglish_hits  / word_count

    # Thresholds are intentionally low to catch short Romanized Indic phrases
    if hinglish_hits >= 2 and hinglish_ratio >= 0.15 and hinglish_hits > benglish_hits:
        return "hinglish"
    if benglish_hits >= 2 and benglish_ratio >= 0.15 and benglish_hits > hinglish_hits:
        return "benglish"
    if hinglish_hits >= 2 and hinglish_ratio >= 0.15:
        return "hinglish"
    if benglish_hits >= 2 and benglish_ratio >= 0.15:
        return "benglish"

    # Weak Romanized signal (1 hit out of few words)
    if (hinglish_hits >= 1 or benglish_hits >= 1) and word_count <= 8:
        return "transliterated"

    return "english"


def is_non_english(detected_lang: str) -> bool:
    """Return True if the detected language requires normalization."""
    return detected_lang not in ("english", "unknown")


# Language code mapping for Sarvam API
LANG_CODE_MAP = {
    "en":     "en-IN",
    "hi":     "hi-IN",
    "bn":     "bn-IN",
    "ta":     "ta-IN",
    "te":     "te-IN",
    "mr":     "mr-IN",
    "gu":     "gu-IN",
    "kn":     "kn-IN",
    "ml":     "ml-IN",
    "pa":     "pa-IN",
    "or":     "or-IN",
    "as":     "as-IN",
}

SUPPORTED_RESPONSE_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}

DETECTED_TO_SARVAM_SOURCE = {
    "hinglish":      "hi-IN",
    "benglish":      "bn-IN",
    "devanagari":    "hi-IN",
    "bengali_script": "bn-IN",
    "south_indic":   "ta-IN",   # conservative default
    "transliterated": "hi-IN",  # default to Hindi for generic Romanized Indic
    "english":       "en-IN",
    "unknown":       "en-IN",
}
