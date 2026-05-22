import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
import json

import streamlit as st

from app.memory.db import init_db
from app.memory.repositories import SessionRepository, ResearchTurnRepository
from app.research.agent import ResearchAgent
from app.multilingual.language_detector import SUPPORTED_RESPONSE_LANGUAGES

# ── Boot ──────────────────────────────────────────────────────────────────────
init_db()
session_repo = SessionRepository()
turn_repo = ResearchTurnRepository()

st.set_page_config(
    page_title="Deep Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Establish focused conversational layout */
[data-testid="stAppViewBlockContainer"] {
    max-width: 840px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 3.5rem !important;
    margin: 0 auto !important;
}

/* Global Font Override & Softer Surfaces */
html, body, [class*="css"], [data-testid="stMarkdownContainer"] p {
    font-family: 'Outfit', 'Source Sans Pro', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #e3e6ed !important;
}

/* Chat Message Redesign */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    margin-bottom: 1.5rem !important;
    padding: 0 !important;
}

[data-testid="stChatMessageContent"] {
    background-color: #161821 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 18px !important;
    padding: 18px 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* Make User Messages compact and visually aligned right */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    display: flex;
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background-color: #212433 !important;
    border: 1px solid rgba(129, 140, 248, 0.15) !important;
    border-radius: 18px 18px 4px 18px !important;
    max-width: 75% !important;
    margin-right: 12px;
}

/* Sidebar Aesthetic Modernization */
[data-testid="stSidebar"] {
    background-color: #08090d !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
}
[data-testid="stSidebar"] [data-testid="stSubheader"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #8f9aa8 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.5rem;
}

/* Premium Floating Input Bar */
[data-testid="stChatInput"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background-color: #12131a !important;
    border-radius: 24px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    padding: 2px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.12) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e3e6ed !important;
    font-size: 0.95rem !important;
}

/* Custom Clickable Prompt Grid styling */
.prompt-grid-container div[data-testid="stButton"] button {
    width: 100% !important;
    background-color: #14161f !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    text-align: left !important;
    color: #e3e6ed !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    margin-bottom: 8px;
    height: 110px !important;
}
.prompt-grid-container div[data-testid="stButton"] button:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    background-color: #1a1c29 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15) !important;
}

/* Timelined AI Reasoning timeline styles */
.react-thought, .react-action, .react-observation, .react-system {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    animation: timelineFadeIn 0.3s ease-out;
}
@keyframes timelineFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.react-thought {
    background: rgba(168, 85, 247, 0.04) !important;
    border-color: rgba(168, 85, 247, 0.12) !important;
    color: #d8b4fe !important;
}
.react-action {
    background: rgba(59, 130, 246, 0.04) !important;
    border-color: rgba(59, 130, 246, 0.12) !important;
    color: #93c5fd !important;
}
.react-observation {
    background: rgba(16, 185, 129, 0.04) !important;
    border-color: rgba(16, 185, 129, 0.12) !important;
    color: #6ee7b7 !important;
}
.react-system {
    background: rgba(107, 114, 128, 0.03) !important;
    color: #9ca3af !important;
}
.event-icon {
    font-size: 0.95rem;
    margin-right: 8px;
    display: inline-flex;
    vertical-align: middle;
}

/* Citation Pill UI */
.citation-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.citation-chip {
    display: inline-flex;
    align-items: center;
    background: #181922;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.8rem;
    color: #a5b4fc !important;
    text-decoration: none !important;
    transition: all 0.2s ease;
}
.citation-chip:hover {
    background: #202230;
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
}
.chip-label {
    font-weight: 600;
    margin-right: 6px;
    color: #818cf8;
}
.chip-domain {
    color: #9ca3af;
    font-size: 0.76rem;
}

/* Evidence Cards */
.evidence-card {
    background: #111219 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

/* Streaming Blinking Cursor */
@keyframes cursorBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}
.cursor-blink {
    display: inline-block;
    color: #818cf8;
    font-weight: bold;
    animation: cursorBlink 0.8s infinite;
    margin-left: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("**Stage 4 + Phase 5** · Deep Research · ReAct UX · Multilingual")
    st.divider()

    sessions = session_repo.list_all()

    # Automatically start a new session on app launch/refresh
    if "session_initialized" not in st.session_state:
        new_s = session_repo.create(title=f"Chat History {len(sessions) + 1}")
        st.session_state.session_id = new_s.id
        st.session_state.session_initialized = True
        st.rerun()

    # Load sessions and display them as Chat History
    session_titles = {
        s.id: f"{s.title.replace('Research Session', 'Chat History')}"
        for s in sessions
    }
    session_ids = [s.id for s in sessions]

    if "session_id" not in st.session_state or st.session_state.session_id not in session_ids:
        st.session_state.session_id = session_ids[-1] if session_ids else None

    selected_session_id = st.selectbox(
        "Chat History",
        options=session_ids,
        format_func=lambda x: session_titles.get(x, f"Chat History {x}"),
        index=session_ids.index(st.session_state.session_id) if st.session_state.session_id in session_ids else 0
    )
    st.session_state.session_id = selected_session_id

    if st.button("🆕 New Session", use_container_width=True):
        new_s = session_repo.create(title=f"Chat History {len(sessions) + 1}")
        st.session_state.session_id = new_s.id
        st.rerun()

    st.divider()

    # ── Language Selector ─────────────────────────────────────────
    st.subheader("🌐 Response Language")
    lang_options = list(SUPPORTED_RESPONSE_LANGUAGES.keys())
    lang_labels  = list(SUPPORTED_RESPONSE_LANGUAGES.values())
    if "target_language" not in st.session_state:
        st.session_state.target_language = "en"
    lang_idx = lang_options.index(st.session_state.target_language) if st.session_state.target_language in lang_options else 0
    selected_lang = st.selectbox(
        "Response in:",
        options=lang_options,
        format_func=lambda x: SUPPORTED_RESPONSE_LANGUAGES.get(x, x),
        index=lang_idx,
        key="lang_selector",
    )
    st.session_state.target_language = selected_lang
    if selected_lang != "en":
        st.caption("💡 Answers will be translated after grounded retrieval. Citations remain in English.")

    st.divider()

    summary = session_repo.get_summary(st.session_state.session_id)
    if summary:
        st.subheader("📝 Summary")
        st.info(summary)
    else:
        st.caption("No conversation summary yet.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_event_html(event_type: str, icon: str, text: str) -> str:
    """Return styled HTML for a workflow event row."""
    return f'<div class="react-{event_type}"><span class="event-icon">{icon}</span> {text}</div>'


def _render_workflow_event(container, event_type: str, icon: str, text: str):
    """Write a single ReAct event into the workflow container."""
    container.markdown(
        _render_event_html(event_type, icon, text),
        unsafe_allow_html=True,
    )


def _render_historical_turn(turn):
    """Render a past turn from DB."""
    with st.chat_message("user"):
        st.markdown(turn.user_query)
        try:
            sq = json.loads(turn.search_queries_json or "[]")
            if sq:
                st.caption("🔍 Searched: " + " · ".join(f"`{q}`" for q in sq))
        except Exception:
            pass

    with st.chat_message("assistant"):
        if not turn.final_answer:
            st.warning("This turn did not produce an answer.")
            return

        st.markdown(turn.final_answer)

        # Citations
        if turn.citations_json:
            try:
                cits = json.loads(turn.citations_json)
                if cits:
                    st.markdown("---")
                    st.markdown("**📚 Sources**")
                    chips_html = '<div class="citation-container">'
                    for c in cits:
                        chips_html += (
                            f'<a href="{c["url"]}" target="_blank" class="citation-chip">'
                            f'<span class="chip-label">{c["label"]}</span>'
                            f'<span class="chip-domain">{c["domain"]}</span>'
                            f'</a>'
                        )
                    chips_html += '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)
            except Exception:
                pass

        # Evidence expander
        if turn.retrieved_chunks_json:
            try:
                chunks = json.loads(turn.retrieved_chunks_json)
                if chunks:
                    with st.expander("🎯 Retrieved Evidence", expanded=False):
                        for chunk in chunks:
                            lbl = chunk.get("citation_label") or ""
                            st.markdown(
                                f'<div class="evidence-card">'
                                f'<strong>{lbl} {chunk["title"]}</strong> — <code>{chunk["domain"]}</code><br/>'
                                f'<small><a href="{chunk["source_url"]}">{chunk["source_url"]}</a></small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            preview = chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else "")
                            st.caption(preview)
            except Exception:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────
_LANG_ICONS = {
    "english":        "",
    "hinglish":       "🇮🇳 Hinglish",
    "benglish":       "🇮🇳 Benglish",
    "devanagari":     "🇮🇳 Devanagari",
    "bengali_script": "🇮🇳 Bengali Script",
    "south_indic":    "🇮🇳 South Indic Script",
    "transliterated": "🇮🇳 Transliterated Indic",
    "unknown":        "",
}

turns = turn_repo.get_all_for_session(st.session_state.session_id)

if not turns:
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem; margin-bottom: 3rem;">
        <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none;">🔬 Deep Research Agent</h1>
        <p style="font-size: 1.15rem; color: #9ca3af; max-width: 600px; margin: 0 auto;">
            Grounded Retrieval-Augmented Generation & Multilingual Research Orchestration
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='color: #e5e7eb; margin-bottom: 1.5rem; font-weight: 500;'>💡 Ask a research question...</h4>", unsafe_allow_html=True)

    st.markdown('<div class="prompt-grid-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🇮🇳 Hinglish\n\nbharat me AI startups ka future kya hai"):
            st.session_state.query_to_run = "bharat me AI startups ka future kya hai"
            st.rerun()
    with col2:
        if st.button("🇺🇸 English\n\nCompare OpenAI and Anthropic pricing structure"):
            st.session_state.query_to_run = "Compare OpenAI and Anthropic pricing structure"
            st.rerun()
    with col3:
        if st.button("🇧🇩 Benglish\n\nবাংলাদেশে EV market growth kemon"):
            st.session_state.query_to_run = "বাংলাদেশে EV market growth kemon"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("<h3 style='margin-top: 0.5rem; font-weight: 600; margin-bottom: 2rem;'>🔬 Conversational Deep Research</h3>", unsafe_allow_html=True)
    for turn in turns:
        _render_historical_turn(turn)

# ── New Query Execution ───────────────────────────────────────────────────────
query = None
if "query_to_run" in st.session_state:
    query = st.session_state.pop("query_to_run")

if not query:
    query = st.chat_input("🔬 Ask anything — English, Hinglish, हिन्दी, বাংলা ...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    target_language = st.session_state.get("target_language", "en")
    agent = ResearchAgent()

    with st.chat_message("assistant"):

        # ── Workflow panel (collapsible) ──────────────────────────────────────
        workflow_expander = st.expander("🧠 Agent Workflow", expanded=True)
        workflow_container = workflow_expander.container()

        lang_info_slot = st.empty()   # fills with language badge for Indic queries

        # ── Plan panel (fills after plan_complete) ────────────────────────────
        plan_expander_slot = st.empty()

        # ── Streaming answer area ─────────────────────────────────────────────
        answer_area = st.empty()
        citations_area = st.empty()
        evidence_area = st.empty()

        async def run_stream():
            answer_buffer = []
            plan_shown = False

            async for event in agent.stream_run(
                query, st.session_state.session_id, target_language=target_language
            ):

                stage = event.stage
                msg   = event.message
                data  = event.data or {}

                # ── Phase 5: Multilingual events ────────────────────────────
                if stage == "lang_detect":
                    dl = data.get("detected_lang", "")
                    label = _LANG_ICONS.get(dl, dl)
                    if dl and dl != "english":
                        _render_workflow_event(workflow_container, "observation", "🌐", f"Detected: {label}")
                        lang_info_slot.info(
                            f"🌐 **Detected language:** {label}  \n"
                            f"🔄 **Normalizing to English for grounded research**"
                        )

                elif stage == "normalize":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "normalize_complete":
                    nq = data.get("normalized_query", "")
                    _render_workflow_event(workflow_container, "observation", "🔍", f'Normalized query: "{nq}"')

                elif stage == "localize":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "localize_complete":
                    _render_workflow_event(workflow_container, "observation", "🔍", msg)

                # ── Existing workflow panel events ───────────────────────────
                elif stage == "start":
                    _render_workflow_event(workflow_container, "system", "⚙️", msg)

                elif stage == "memory":
                    _render_workflow_event(workflow_container, "system", "🗃️", msg)

                elif stage == "plan":
                    _render_workflow_event(workflow_container, "thought", "💭", msg)

                elif stage == "plan_complete":
                    plan_text = data.get("plan_text", "")
                    search_queries = data.get("search_queries", [])
                    _render_workflow_event(
                        workflow_container, "thought", "💭",
                        "Research plan formulated. Starting evidence acquisition."
                    )
                    # Render plan in its own expander below workflow
                    if plan_text and not plan_shown:
                        plan_shown = True
                        with plan_expander_slot.expander("📋 Research Plan", expanded=False):
                            st.markdown(plan_text)
                            if search_queries:
                                st.markdown("**Search queries:**")
                                for sq in search_queries:
                                    st.markdown(f"- `{sq}`")

                elif stage == "search":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "search_complete":
                    urls = data.get("urls", [])
                    _render_workflow_event(
                        workflow_container, "observation", "🔍",
                        f"Found {len(urls)} source(s): " + ", ".join(f"`{u.split('/')[2]}`" for u in urls[:3])
                        + ("..." if len(urls) > 3 else "")
                    )

                elif stage == "fetch":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "extract":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "extract_complete":
                    ok = data.get("successful_count", "?")
                    total = ok + data.get("failed_count", 0)
                    _render_workflow_event(
                        workflow_container, "observation", "🔍",
                        f"Extracted readable content from {ok}/{total} page(s)."
                    )

                elif stage == "persist_turn":
                    _render_workflow_event(workflow_container, "system", "💾", msg)

                elif stage == "chunk":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "chunk_complete":
                    _render_workflow_event(workflow_container, "system", "⚙️", msg)

                elif stage == "embed":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "faiss":
                    _render_workflow_event(workflow_container, "system", "⚙️", msg)

                elif stage == "retrieve":
                    _render_workflow_event(workflow_container, "action", "⚡", msg)

                elif stage == "select_evidence":
                    chunk_count = data.get("chunk_count", 0)
                    top_sources = data.get("top_sources", [])
                    domains = " · ".join(f"`{s['domain']}`" for s in top_sources[:4])
                    _render_workflow_event(
                        workflow_container, "observation", "🔍",
                        f"Selected {chunk_count} evidence passage(s) from: {domains}"
                    )

                    # Show evidence snippets inside workflow panel
                    chunks = data.get("chunks", [])
                    if chunks:
                        workflow_container.markdown("**Relevant Evidence Snippets:**")
                        for chunk in chunks[:4]:
                            workflow_container.markdown(
                                f'<div class="evidence-card">'
                                f'<strong>{chunk["title"]}</strong> <code>{chunk["domain"]}</code> '
                                f'<span style="color:#6b7280;font-size:0.75rem;">score: {chunk["score"]}</span><br/>'
                                f'<span style="color:#9ca3af;font-size:0.82rem;">{chunk["preview"]}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                elif stage == "conflict":
                    _render_workflow_event(
                        workflow_container, "observation", "⚠️", msg
                    )

                elif stage == "context":
                    _render_workflow_event(workflow_container, "system", "⚙️", msg)

                elif stage == "llm":
                    _render_workflow_event(
                        workflow_container, "thought", "💭",
                        "Synthesising a grounded answer from retrieved evidence..."
                    )

                elif stage == "token":
                    # Stream token into answer area with animated cursor
                    answer_buffer.append(msg)
                    answer_area.markdown("".join(answer_buffer) + '<span class="cursor-blink">▌</span>', unsafe_allow_html=True)

                elif stage == "summary":
                    _render_workflow_event(workflow_container, "system", "📝", msg)

                elif stage == "complete":
                    # Final answer (remove cursor)
                    answer_area.markdown(data.get("answer", ""))

                    # Citations
                    cits = data.get("citations", [])
                    if cits:
                        citations_area.markdown("---")
                        citations_area.markdown("**📚 Sources**")
                        chips_html = '<div class="citation-container">'
                        for c in cits:
                            chips_html += (
                                f'<a href="{c["url"]}" target="_blank" class="citation-chip">'
                                f'<span class="chip-label">{c["label"]}</span>'
                                f'<span class="chip-domain">{c["domain"]}</span>'
                                f'</a>'
                            )
                        chips_html += '</div>'
                        citations_area.markdown(chips_html, unsafe_allow_html=True)

                    # Evidence expander at bottom
                    retrieved = data.get("retrieved_chunks", [])
                    if retrieved:
                        with evidence_area.expander("🎯 Retrieved Evidence", expanded=False):
                            for chunk in retrieved:
                                lbl = chunk.get("citation_label") or ""
                                st.markdown(
                                    f'<div class="evidence-card">'
                                    f'<strong>{lbl} {chunk["title"]}</strong> — <code>{chunk["domain"]}</code><br/>'
                                    f'<small><a href="{chunk["source_url"]}">{chunk["source_url"]}</a></small>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                                preview = chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else "")
                                st.caption(preview)

                    _render_workflow_event(
                        workflow_container, "system", "✅",
                        "Research complete."
                    )

                elif stage == "error":
                    workflow_container.error(f"❌ {msg}")
                    answer_area.error(f"Research failed: {msg}")

        asyncio.run(run_stream())

    st.rerun()
