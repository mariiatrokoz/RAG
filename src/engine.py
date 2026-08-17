from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer

from src.config import LLM_SYSTEM_PROMPT, CHAT_MEMORY_TOKEN_LIMIT
from src.model_loader import initialise_llm

from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm
)


def main_chat_loop() -> None:
    """Main chat loop."""

    llm: LLM = initialise_llm()

    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[],
        llm=llm,
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    conversation: SimpleChatEngine = SimpleChatEngine.from_defaults(
        llm=llm,
        system_prompt=LLM_SYSTEM_PROMPT
    )

    conversation.chat_repl()


def _create_new_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""

    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

    # This reads all the text files in the specified directory.
    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # This breaks the long document into smaller, more manageable chunks.
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # This is the core of the vector store. It takes the text chunks,
    # uses the embedding model to convert them to vectors, and stores them.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )

    # This saves the newly created index to disk for future use.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index


def loading_index(embed_model:HuggingFaceEmbedding) -> VectorStoreIndex:

    ''' loads index if it exists or uses helper function to create one, it it doesn not. '''

    # creating a parent directory or loading an existing one 
    VECTOR_STORE_PATH.mkdir(parent=True, exist_ok=True)

    # checking if directory contains any files
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk.")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir = VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )

    else:
        return _create_new_vector_store(embed_model)


def get_chat_engine(
        llm: Groq,
        embed_model: HuggingFaceEmbedding
) -> BaseChatEngine:
    """Initialises and returns the main conversational RAG chat engine."""

    vector_index: VectorStoreIndex = get_vector_store(embed_model)
    memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    # Assemble the RAG chat engine
    chat_engine: BaseChatEngine = vector_index.as_chat_engine(
        memory=memory,
        llm=llm,
        system_prompt=LLM_SYSTEM_PROMPT,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    return chat_engine