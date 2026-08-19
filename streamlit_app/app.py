"""The chat UI. Run from the project root:  streamlit run app.py"""

import hashlib

import streamlit as st
from app import app_config as cfg
from app.app_engine import get_chat_engine, get_table_of_contents
from app.app_model_loader import get_llm

# Must be the first Streamlit call - sets the browser tab's title/icon,
# separate from st.title() below which sets the on-page heading.
st.set_page_config(page_title=cfg.PAGE_TITLE)

st.title(cfg.PAGE_TITLE)

uploaded_files = st.sidebar.file_uploader(
    "Chat with your own document instead",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Stable id for this exact set of files: re-uploading the same one(s)
    # reuses its index instead of rebuilding it.
    digest = hashlib.sha256()
    for f in sorted(uploaded_files, key=lambda f: f.name):
        digest.update(f.name.encode())
        digest.update(f.getvalue())
    cache_key = digest.hexdigest()[:16]

    data_path = cfg.UPLOADS_PATH / cache_key / "source"
    persist_dir = cfg.UPLOADS_PATH / cache_key / "index"
    if not data_path.exists():
        data_path.mkdir(parents=True)
        for f in uploaded_files:
            (data_path / f.name).write_bytes(f.getvalue())

    doc_label = ", ".join(f.name for f in uploaded_files)
else:
    cache_key = "default"
    data_path = cfg.DATA_PATH
    persist_dir = cfg.VECTOR_STORE_PATH
    doc_label = "AI Engineering (default book)"

toc = get_table_of_contents(cache_key, data_path)
if toc:
    # One toggle for the whole list rather than one expander per chapter -
    # Streamlit doesn't allow nesting expanders inside each other anyway, and
    # ~20 separate chapter toggles would dominate the sidebar on their own.
    with st.sidebar.expander("📑 Table of Contents", expanded=False):
        chapters: list[dict] = []
        for entry in toc:
            if entry["depth"] == 0:
                chapters.append({**entry, "children": []})
            elif chapters:
                chapters[-1]["children"].append(entry)
        for chapter in chapters:
            page_suffix = f" (p.{chapter['page']})" if chapter["page"] else ""
            st.markdown(f"**{chapter['title']}**{page_suffix}")
            for child in chapter["children"]:
                indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * child["depth"]
                page_suffix = f" · p.{child['page']}" if child["page"] else ""
                st.markdown(
                    f"{indent}- {child['title']}{page_suffix}", unsafe_allow_html=True
                )

st.caption(f"📄 Currently chatting with: **{doc_label}**")

# Cheap after the first build for a given cache_key (cache_resource lookup),
# and switching back to a previous document restores its own chat history.
st.session_state.engine = get_chat_engine(cache_key, data_path, persist_dir)

# The engine's own memory only keeps role/content, not which chunks backed
# each answer, so citations are tracked separately here: one list per
# assistant turn, keyed by document so switching docs doesn't mix them up.
citations_by_doc = st.session_state.setdefault("citations_by_doc", {})
citations = citations_by_doc.setdefault(cache_key, [])

if st.sidebar.button("🧹 Clear conversation"):
    # The chat engine is cached (@st.cache_resource) and shared across
    # reruns, so clearing history means resetting its memory in place
    # rather than dropping the object - only this document's chat is
    # affected, other cached documents keep their own history.
    st.session_state.engine.reset()
    citations.clear()
    st.rerun()


def _describe_source(node_with_score) -> dict:
    meta = node_with_score.node.metadata
    name = meta.get("file_name", "document")
    page = meta.get("page_label")
    label = f"{name} — page {page}" if page else name
    snippet = node_with_score.node.get_content().strip()
    return {
        "label": label,
        "score": node_with_score.score or 0.0,
        "snippet": snippet[:400] + ("…" if len(snippet) > 400 else ""),
    }


def _render_citations(turn_citations: list[dict]) -> None:
    if not turn_citations:
        return
    with st.expander(f"📚 Sources ({len(turn_citations)})"):
        for c in turn_citations:
            st.markdown(f"**{c['label']}** · relevance {c['score']:.2f}")
            st.caption(c["snippet"])
            st.divider()


# Changes the LLM in place, so the conversation survives moving the slider
#get_llm().temperature = st.sidebar.slider(
#    "Temperature", 0.0, 1.0, cfg.LLM_TEMPERATURE, 0.05
#)

# Directly read LlamaIndex's internal memory to display past messages
assistant_turn = 0
for msg in st.session_state.engine.chat_history:
    st.chat_message(msg.role.value).markdown(msg.content)
    if msg.role.value == "assistant":
        if assistant_turn < len(citations):
            _render_citations(citations[assistant_turn])
        assistant_turn += 1

# Handle new input and stream the response

if question := st.chat_input(cfg.CHAT_PLACEHOLDER):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking ... "):
            response = st.session_state.engine.stream_chat(question)
        if response.source_nodes:
            st.write_stream(response.response_gen)
        else:
            # No chunk cleared MIN_RETRIEVAL_SCORE, so the question is off-topic
            # for this document - refuse without ever involving the LLM, instead
            # of streaming its generic, contextless "Empty Response" filler.
            st.markdown(cfg.REFUSAL_MESSAGE)
        turn_citations = [_describe_source(ns) for ns in response.source_nodes]
        citations.append(turn_citations)
        _render_citations(turn_citations)