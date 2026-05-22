import sys
from pathlib import Path
# Ensure the project root is on sys.path regardless of how streamlit is invoked
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from typing import List, Union

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

# Premium Dark Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .sidebar .sidebar-content {
        background: #1E293B;
    }
    h1, h2, h3 {
        color: #F8FAFC;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("**Stage 3** · Conversational Research")
    st.divider()

    # Session Selector
    sessions = session_repo.list_all()
    if not sessions:
        new_session = session_repo.create(title="Research Session")
        sessions = [new_session]
        
    session_titles = {s.id: f"{s.title} (ID: {s.id})" for s in sessions}
    session_ids = [s.id for s in sessions]
    
    # Manage session ID in session state
    if "session_id" not in st.session_state or st.session_state.session_id not in session_ids:
        st.session_state.session_id = session_ids[0]
        
    selected_session_id = st.selectbox(
        "Select Session",
        options=session_ids,
        format_func=lambda x: session_titles.get(x, f"Session {x}"),
        index=session_ids.index(st.session_state.session_id)
    )
    st.session_state.session_id = selected_session_id

    # New Session Button
    if st.button("🆕 New Session", use_container_width=True):
        new_s = session_repo.create(title=f"Research Session {len(sessions) + 1}")
        st.session_state.session_id = new_s.id
        st.rerun()

    st.divider()
    
    # Display current rolling summary
    summary = session_repo.get_summary(st.session_state.session_id)
    if summary:
        st.subheader("📝 Rolling Summary")
        st.info(summary)
    else:
        st.caption("No conversation summary yet. Start asking research questions to build one.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Conversational Deep Research")
st.caption("Grounded answers with citations using dense RAG & multi-turn memory.")

# Get all turns for the active session
turns = turn_repo.get_all_for_session(st.session_state.session_id)

# Display historical messages in chat format
for turn in turns:
    with st.chat_message("user"):
        st.markdown(turn.user_query)
    with st.chat_message("assistant"):
        if turn.final_answer:
            st.markdown(turn.final_answer)
            
            # Display persisted Citations
            if turn.citations_json:
                try:
                    import json
                    citations = json.loads(turn.citations_json)
                    if citations:
                        cit_str = "\n".join([f"**{c['label']}** [{c['title']}]({c['url']}) — `{c['domain']}`" for c in citations])
                        st.markdown(f"### 📚 Citations\n{cit_str}")
                except Exception as e:
                    pass
                    
            # Display persisted Evidence
            if turn.retrieved_chunks_json:
                try:
                    import json
                    retrieved = json.loads(turn.retrieved_chunks_json)
                    if retrieved:
                        with st.expander("🎯 Retrieved Evidence"):
                            for chunk in retrieved:
                                lbl = chunk.get("citation_label") or ""
                                st.markdown(f"**{lbl} {chunk['title']}** ({chunk['domain']})")
                                st.caption(chunk['source_url'])
                                preview = chunk['text'][:600] + ("…" if len(chunk['text']) > 600 else "")
                                st.text(preview)
                                st.divider()
                except Exception as e:
                    pass
        else:
            st.warning("This research turn failed or did not output an answer.")

# Chat input for new query
if query := st.chat_input("Ask a follow-up or a new research question..."):
    # Render user query immediately in UI
    with st.chat_message("user"):
        st.markdown(query)
        
    # Instantiating agent and streaming response
    agent = ResearchAgent()
    
    with st.chat_message("assistant"):
        # Setting up streaming pipeline UI
        with st.status("Initializing research pipeline...", expanded=True) as status_box:
            status_text = st.empty()
            
            async def run_stream():
                ans_placeholder = st.empty()
                cit_placeholder = st.empty()
                ev_placeholder = st.empty()
                
                async for event in agent.stream_run(query, st.session_state.session_id):
                    # Update status text based on active pipeline stage
                    if event.stage == "start":
                        status_box.update(label="Starting pipeline...", state="running")
                        status_text.text(f"🚀 {event.message}")
                    elif event.stage == "memory":
                        status_text.text(f"🧠 {event.message}")
                    elif event.stage == "search":
                        status_text.text(f"🔍 {event.message}")
                    elif event.stage == "search_complete":
                        status_text.text(f"✅ {event.message}")
                    elif event.stage == "fetch":
                        status_text.text(f"📥 {event.message}")
                    elif event.stage == "extract":
                        status_text.text(f"📄 {event.message}")
                    elif event.stage == "extract_complete":
                        status_text.text(f"✅ {event.message}")
                    elif event.stage == "chunk":
                        status_text.text(f"✂️ {event.message}")
                    elif event.stage == "chunk_complete":
                        status_text.text(f"✅ {event.message}")
                    elif event.stage == "embed":
                        status_text.text(f"🧠 {event.message}")
                    elif event.stage == "faiss":
                        status_text.text(f"🗄️ {event.message}")
                    elif event.stage == "retrieve":
                        status_text.text(f"🎯 {event.message}")
                    elif event.stage == "context":
                        status_text.text(f"🏗️ {event.message}")
                    elif event.stage == "llm":
                        status_text.text(f"💬 {event.message}")
                    elif event.stage == "summary":
                        status_text.text(f"📝 {event.message}")
                    elif event.stage == "complete":
                        status_box.update(label="Research complete ✅", state="complete")
                        
                        data = event.data
                        ans_placeholder.markdown(data["answer"])
                        
                        # Display Citations
                        citations = data["citations"]
                        if citations:
                            cit_str = "\n".join([f"**{c['label']}** [{c['title']}]({c['url']}) — `{c['domain']}`" for c in citations])
                            cit_placeholder.markdown(f"### 📚 Citations\n{cit_str}")
                            
                        # Display Evidence
                        retrieved = data["retrieved_chunks"]
                        if retrieved:
                            with ev_placeholder.expander("🎯 Retrieved Evidence"):
                                for chunk in retrieved:
                                    lbl = chunk.get("citation_label") or ""
                                    st.markdown(f"**{lbl} {chunk['title']}** ({chunk['domain']})")
                                    st.caption(chunk['source_url'])
                                    preview = chunk['text'][:600] + ("…" if len(chunk['text']) > 600 else "")
                                    st.text(preview)
                                    st.divider()
                                    
                    elif event.stage == "error":
                        status_box.update(label=f"Pipeline error: {event.message}", state="error")
                        st.error(event.message)
                        
            asyncio.run(run_stream())
            
    # Force a rerun to reload all chat messages and updated summary sidebar
    st.rerun()
