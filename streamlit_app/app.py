"""The chat UI. Run from the project root:  streamlit run app.py"""

import hashlib

import streamlit as st
from app import app_config as cfg
from app.app_engine import get_chat_engine
from app.app_model_loader import get_llm

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

st.caption(f"📄 Currently chatting with: **{doc_label}**")

# Cheap after the first build for a given cache_key (cache_resource lookup),
# and switching back to a previous document restores its own chat history.
st.session_state.engine = get_chat_engine(cache_key, data_path, persist_dir)

# Changes the LLM in place, so the conversation survives moving the slider
#get_llm().temperature = st.sidebar.slider(
#    "Temperature", 0.0, 1.0, cfg.LLM_TEMPERATURE, 0.05
#)

# Directly read LlamaIndex's internal memory to display past messages
for msg in st.session_state.engine.chat_history:
    st.chat_message(msg.role.value).markdown(msg.content)

# Handle new input and stream the response

if question := st.chat_input(cfg.CHAT_PLACEHOLDER):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking ... "):
            response = st.session_state.engine.stream_chat(question)
        st.write_stream(response.response_gen)