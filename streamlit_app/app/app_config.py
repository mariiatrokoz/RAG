"""Every setting of the chatbot, in one place."""

from pathlib import Path

ROOT_PATH: Path = Path(__file__).resolve().parent.parent.parent
DATA_PATH: Path = ROOT_PATH / "data"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage" / "semantic_vector_store"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage" / "embedding_model"
UPLOADS_PATH: Path = ROOT_PATH / "local_storage" / "uploads"

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
# Gate on the *raw* embedding similarity, before reranking: on-topic questions
# score 0.3-0.6+ here, off-topic ones (tested against several unrelated
# questions) topped out at 0.29. The cross-encoder reranker's own score
# looked like a better signal but wasn't - it occasionally scored a genuinely
# on-topic chunk near 0 for a paraphrased question, which caused false
# refusals. Filtering here, before reranking runs, avoided that.
MIN_RETRIEVAL_SCORE: float = 0.3

LLM_MODEL: str = "openai/gpt-oss-20b"
LLM_TEMPERATURE: float = 0.01
LLM_MAX_TOKENS: int = 2048        # covers the hidden thinking AND the answer
LLM_REASONING_EFFORT: str = "low"
# Refusing off-topic questions is now mostly handled upstream (MIN_RETRIEVAL_SCORE
# filters them down to zero chunks, which short-circuits to REFUSAL_MESSAGE
# without ever calling the LLM - see app.py). This prompt only has to stop the
# model blending in outside facts when real chunks *are* present; a stronger,
# more absolute wording here was tested and caused false refusals on legitimate
# questions instead.
REFUSAL_MESSAGE: str = "I don't have information about that in this document."
SYSTEM_PROMPT: str = (
    "You are a helpful assistant that answers questions about a specific "
    "document, using the context excerpts provided below. Base your answer on "
    "that context - do not bring in outside facts or general knowledge to fill "
    f'gaps. If the context does not address the question, say: "{REFUSAL_MESSAGE}"'
)
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

PAGE_TITLE: str = "Your document expert 🤓"

CHAT_PLACEHOLDER: str = "Ask something ... "
