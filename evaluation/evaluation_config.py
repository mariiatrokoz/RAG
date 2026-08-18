from pathlib import Path

from ragas.metrics.base import Metric
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,

)

# --- LLM Model Configuration ---
EVALUATION_LLM_MODEL: str = "moonshotai/kimi-k2-instruct"

# --- Embedding Model Configuration ---
EVALUATION_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# --- Paths for Evaluation ---
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results/"
EXPERIMENTAL_VECTOR_STORES_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_vector_stores/"
)
EVALUATION_EMBEDDING_CACHE_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_embedding_models/"
)

# --- Ragas Evaluation Metrics ---
EVALUATION_METRICS: list[Metric] = [
    Faithfulness(),
    AnswerCorrectness(),
    ContextPrecision(),
    ContextRecall(),
]

# --- Sleep Timers for API Limits ---
SLEEP_PER_EVALUATION: int = 60
SLEEP_PER_QUESTION: int = 6