from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_SYSTEM_PROMPT: str = ("Scientific tech language.")
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03

CHAT_MEMORY_TOKEN_LIMIT = 4000

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 2
# The size of each text chunk in tokens
CHUNK_SIZE: int = 512
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 50


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"
