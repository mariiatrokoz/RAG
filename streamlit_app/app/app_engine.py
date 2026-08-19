"""The RAG pipeline: semantic chunks -> vector search -> reranking -> LLM."""

import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.postprocessor import (
    PrevNextNodePostprocessor,
    SentenceTransformerRerank,
)

from app import app_config as cfg
from app.app_model_loader import get_embedding_model, get_llm


def get_vector_store(embed_model) -> VectorStoreIndex:
    """Loads the vector store from disk, or builds it on the first run."""

    if any(cfg.VECTOR_STORE_PATH.glob("*")):
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=cfg.VECTOR_STORE_PATH.as_posix()),
            embed_model=embed_model,
        )

    # Cuts where the meaning changes instead of at a fixed length.
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=cfg.BREAKPOINT_PERCENTILE,
    )
    documents = SimpleDirectoryReader(cfg.DATA_PATH.as_posix()).load_data()
    nodes = splitter.get_nodes_from_documents(documents)

    # Semantic chunks have no size limit, so cut the long ones down.
    nodes = SentenceSplitter(
        chunk_size=cfg.MAX_CHUNK_TOKENS, chunk_overlap=0
    ).get_nodes_from_documents(nodes)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=cfg.VECTOR_STORE_PATH.as_posix())
    return index


@st.cache_resource(show_spinner="Loading models and documents…")
def get_chat_engine() -> CondensePlusContextChatEngine:
    """Assembles the chatbot once, then reuses it on every rerun.

    ponytail: one engine for all browser tabs, so they share the chat memory -
    fine for a local single-user app; build it per session if that changes.
    """

    index = get_vector_store(get_embedding_model())

    return CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=cfg.SIMILARITY_TOP_K),
        llm=get_llm(),
        memory=ChatMemoryBuffer.from_defaults(token_limit=cfg.CHAT_MEMORY_TOKEN_LIMIT),
        system_prompt=cfg.SYSTEM_PROMPT,
        # Run in order: rerank down to the best few, then glue the chunks that
        # sat next to them in the book back on, so the LLM reads whole scenes.
        node_postprocessors=[
            SentenceTransformerRerank(top_n=cfg.RERANKER_TOP_N, model=cfg.RERANKER_MODEL),
            PrevNextNodePostprocessor(
                docstore=index.docstore, num_nodes=cfg.NEIGHBOUR_CHUNKS, mode="both"
            ),
        ],
    )
