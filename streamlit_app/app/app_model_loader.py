"""The two models: the one that embeds, and the one that answers."""

import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from app import app_config as cfg

load_dotenv(cfg.ROOT_PATH / ".env")   # GROQ_API_KEY


def get_embedding_model() -> HuggingFaceEmbedding:
    """Turns text into vectors. Downloaded once, then read from the cache."""

    return HuggingFaceEmbedding(
        model_name=cfg.EMBEDDING_MODEL,
        cache_folder=cfg.EMBEDDING_CACHE_PATH.as_posix(),
    )


@st.cache_resource
def get_llm() -> Groq:
    """The LLM that writes the answers. Cached, so the slider in app.py can
    change its temperature on the object the chat engine already holds."""

    return Groq(
        api_key=os.environ["GROQ_API_KEY"],
        model=cfg.LLM_MODEL,
        temperature=cfg.LLM_TEMPERATURE,
        max_tokens=cfg.LLM_MAX_TOKENS,
        additional_kwargs={"reasoning_effort": cfg.LLM_REASONING_EFFORT},
    )
