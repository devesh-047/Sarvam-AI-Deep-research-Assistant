import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
import json

import streamlit as st

from app.memory.db import init_db
from app.memory.repositories import SessionRepository, ResearchTurnRepository
from app.research.agent import ResearchAgent

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Thought events */
.react-thought {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    color: #a5b4fc;
    font-style: italic;
    font-size: 0.88rem;
}

/* Action events */
.react-action {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    color: #93c5fd;
    font-size: 0.88rem;
    font-weight: 500;
}

/* Observation events */
.react-observation {
    background: rgba(16,185,129,0.08);
    border-left: 3px solid #10b981;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    color: #6ee7b7;
    font-size: 0.88rem;
}

/* System events */
.react-system {
    color: #6b7280;
    font-size: 0.80rem;
    padding: 2px 8px;
    margin: 1px 0;
}

/* Evidence card */
.evidence-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}

/* Citation pill */
.citation-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 0.80rem;
    color: #c7d2fe;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("**Stage 4** · Deep Research · ReAct UX")
    st.divider()

    sessions = session_repo.list_all()
    if not sessions:
        new_session = session_repo.create(title="Research Session 1")
        sessions = [new_session]

    session_titles = {s.id: f"{s.title}" for s in sessions}
    session_ids = [s.id for s in sessions]

    if "session_id" not in st.session_state or st.session_state.session_id not in session_ids:
        st.session_state.session_id = session_ids[0]

    selected_session_id = st.selectbox(
        "Session",
        options=session_ids,
        format_func=lambda x: session_titles.get(x, f"Session {x}"),
        index=session_ids.index(st.session_state.session_id)
    )
    st.session_state.session_id = selected_session_id

    if st.button("🆕 New Session", use_container_width=True):
        new_s = session_repo.create(title=f"Research Session {len(sessions) + 1}")
        st.session_state.session_id = new_s.id
        st.rerun()

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
    return f'<div class="react-{event_type}">{icon} {text}</div>'


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
                    for c in cits:
                        st.markdown(
                            f'<span class="citation-pill">{c["label"]}</span> '
                            f'[{c["title"]}]({c["url"]}) — `{c["domain"]}`',
                            unsafe_allow_html=True,
                        )
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
st.title("🔬 Conversational Deep Research")
st.caption("Grounded answers · Dense RAG · Multi-turn memory · Visible agentic workflow")

turns = turn_repo.get_all_for_session(st.session_state.session_id)
for turn in turns:
    _render_historical_turn(turn)

# ── New Query ─────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a research question..."):
    with st.chat_message("user"):
        st.markdown(query)

    agent = ResearchAgent()

    with st.chat_message("assistant"):

        # ── Workflow panel (collapsible) ──────────────────────────────────────
        workflow_expander = st.expander("🧠 Agent Workflow", expanded=True)
        workflow_container = workflow_expander.container()

        # ── Plan panel (fills after plan_complete) ────────────────────────────
        plan_expander_slot = st.empty()

        # ── Streaming answer area ─────────────────────────────────────────────
        answer_area = st.empty()
        citations_area = st.empty()
        evidence_area = st.empty()

        async def run_stream():
            answer_buffer = []
            plan_shown = False

            async for event in agent.stream_run(query, st.session_state.session_id):
                etype = event.event_type
                stage = event.stage
                msg   = event.message
                data  = event.data or {}

                # ── Workflow panel events ─────────────────────────────────────
                if stage == "start":
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
                    # Stream token into answer area
                    answer_buffer.append(msg)
                    answer_area.markdown("".join(answer_buffer) + "▌")

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
                        for c in cits:
                            citations_area.markdown(
                                f'<span class="citation-pill">{c["label"]}</span> '
                                f'[{c["title"]}]({c["url"]}) — `{c["domain"]}`',
                                unsafe_allow_html=True,
                            )

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
