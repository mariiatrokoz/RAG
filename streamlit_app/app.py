"""The chat UI. Run from the project root:  streamlit run app.py"""

import streamlit as st
from app import app_config as cfg
from app.app_engine import get_chat_engine
from app.app_model_loader import get_llm

st.title(cfg.PAGE_TITLE)

# Initialize and cache the engine in session state
if "engine" not in st.session_state:
    st.session_state.engine = get_chat_engine()

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