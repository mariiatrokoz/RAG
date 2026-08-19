"""The RAG pipeline: semantic chunks -> vector search -> reranking -> LLM."""

from pathlib import Path

import streamlit as st
from pypdf import PdfReader
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
    SimilarityPostprocessor,
)

from app import app_config as cfg
from app.app_model_loader import get_embedding_model, get_llm


def get_vector_store(embed_model, data_path: Path, persist_dir: Path) -> VectorStoreIndex:
    """Loads the vector store from disk, or builds it on the first run."""

    if any(persist_dir.glob("*")):
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=persist_dir.as_posix()),
            embed_model=embed_model,
        )

    # Cuts where the meaning changes instead of at a fixed length.
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=cfg.BREAKPOINT_PERCENTILE,
    )
    documents = SimpleDirectoryReader(data_path.as_posix()).load_data()
    nodes = splitter.get_nodes_from_documents(documents)

    # Semantic chunks have no size limit, so cut the long ones down.
    nodes = SentenceSplitter(
        chunk_size=cfg.MAX_CHUNK_TOKENS, chunk_overlap=0
    ).get_nodes_from_documents(nodes)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=persist_dir.as_posix())
    return index


def _flatten_outline(reader: PdfReader, items, depth: int = 0) -> list[dict]:
    entries = []
    for item in items:
        if isinstance(item, list):
            entries.extend(_flatten_outline(reader, item, depth + 1))
            continue
        try:
            page = reader.get_destination_page_number(item) + 1
        except Exception:
            page = None
        entries.append({"title": item.title, "page": page, "depth": depth})
    return entries


@st.cache_data(show_spinner=False)
def get_table_of_contents(cache_key: str, _data_path: Path) -> list[dict]:
    """Reads the embedded bookmarks out of every PDF in the document.

    Not every PDF (and no txt/docx/md file) has these, so an empty list is a
    normal result, not an error - the sidebar just shows nothing then.
    """

    toc = []
    for pdf_path in sorted(_data_path.glob("*.pdf")):
        reader = PdfReader(pdf_path.as_posix())
        if reader.outline:
            toc.extend(_flatten_outline(reader, reader.outline))
    return toc


@st.cache_resource(show_spinner="Reading the document and warming up the model…")
def get_chat_engine(
    cache_key: str, _data_path: Path, _persist_dir: Path
) -> CondensePlusContextChatEngine:
    """Assembles the chatbot once per document, then reuses it on every rerun.

    Cached on `cache_key` (a stable id for the active document), so switching
    documents rebuilds only the first time and keeps each document's own chat
    history when you switch back. The leading underscore on the path args
    tells Streamlit to leave them out of the cache key, since Path objects
    aren't hashable the way we'd want.
    """

    index = get_vector_store(get_embedding_model(), _data_path, _persist_dir)

    return CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=cfg.SIMILARITY_TOP_K),
        llm=get_llm(),
        memory=ChatMemoryBuffer.from_defaults(token_limit=cfg.CHAT_MEMORY_TOKEN_LIMIT),
        system_prompt=cfg.SYSTEM_PROMPT,
        # Run in order: drop chunks that were never really a match (so an
        # off-topic question doesn't survive to be answered at all - see
        # MIN_RETRIEVAL_SCORE), rerank what's left down to the best few, then
        # glue the chunks that sat next to the survivors back on, so the LLM
        # reads whole scenes.
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=cfg.MIN_RETRIEVAL_SCORE),
            SentenceTransformerRerank(top_n=cfg.RERANKER_TOP_N, model=cfg.RERANKER_MODEL),
            PrevNextNodePostprocessor(
                docstore=index.docstore, num_nodes=cfg.NEIGHBOUR_CHUNKS, mode="both"
            ),
        ],
    )
