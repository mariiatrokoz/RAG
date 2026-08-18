from datasets import Dataset

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.query_engine import (
    BaseQueryEngine,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pandas as pd
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_helper_functions import (
    generate_qa_dataset,
    get_evaluation_data,
    get_or_build_index,
    save_results,
    evaluate_without_rate_limit,
    #evaluate_with_rate_limit,
)
from evaluation.evaluation_model_loader import load_ragas_models
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SIMILARITY_TOP_K,
)
from src.model_loader import get_embedding_model, initialise_llm

from evaluation.evaluation_helper_functions import (

    #evaluate_without_rate_limit
    evaluate_with_rate_limit
)

def evaluate_baseline() -> None:
    """
    Evaluates the RAG system using only the settings from config.py.
    """

    print("--- 🚀 Stage 1: Evaluating Baseline Configuration ---")

    llm_to_test: Groq = initialise_llm()

    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions: list[str]
    ground_truths: list[str]
    questions, ground_truths = get_evaluation_data()

    index: VectorStoreIndex = get_or_build_index(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model=embed_model_to_test
    )

    query_engine: BaseQueryEngine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        llm=llm_to_test
    )

    qa_dataset: Dataset = generate_qa_dataset(
        query_engine,
        questions,
        ground_truths
    )

    print("--- Running Ragas evaluation for baseline... ---")

    ragas_llm: LlamaIndexLLMWrapper
    ragas_embeddings: HuggingFaceEmbeddings
    ragas_llm, ragas_embeddings = load_ragas_models()

    # results_df: pd.DataFrame = evaluate_without_rate_limit(
    #     qa_dataset,
    #     ragas_llm,
    #     ragas_embeddings,
    # )

    results_df: pd.DataFrame = evaluate_with_rate_limit(
    qa_dataset,
    ragas_llm,
    ragas_embeddings,
)

    # Add Chunk Size and Chunk Overlap to DataFrame to help tracking
    results_df['chunk_size'] = CHUNK_SIZE
    results_df['chunk_overlap'] = CHUNK_OVERLAP

    save_results(results_df, "baseline_evaluation")

    print("--- ✅ Baseline Evaluation Complete ---")
