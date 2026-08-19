"""Every setting of the chatbot, in one place."""

from pathlib import Path

ROOT_PATH: Path = Path(__file__).resolve().parent.parent.parent
DATA_PATH: Path = ROOT_PATH / "data"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage" / "semantic_vector_store"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage" / "embedding_model"

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
BREAKPOINT_PERCENTILE: int = 90   # lower = the splitter cuts more often
# The semantic splitter has no size limit, so its chunks are cut down to this
# many tokens afterwards. Small chunks embed sharply (the model only reads the
# first ~256 tokens of one) and the neighbours below give back the context.
MAX_CHUNK_TOKENS: int = 100

SIMILARITY_TOP_K: int = 30        # chunks the vector store hands over
RERANKER_TOP_N: int = 5           # chunks kept after reranking
NEIGHBOUR_CHUNKS: int = 1         # chunks added on each side of a kept one
RERANKER_MODEL: str = "BAAI/bge-reranker-base"

LLM_MODEL: str = "openai/gpt-oss-20b"
LLM_TEMPERATURE: float = 0.01
LLM_MAX_TOKENS: int = 2048        # covers the hidden thinking AND the answer
LLM_REASONING_EFFORT: str = "low"
SYSTEM_PROMPT: str = (
    "You are a helpful chatbot. "
    "You only answer if the relevant information is in your context!"
)
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

PAGE_TITLE: str = "AI Engineering expert 🤓"

CHAT_PLACEHOLDER: str = "Ask something ... "
