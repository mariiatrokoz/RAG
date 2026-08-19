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

from evaluation.evaluation_config import (
    CHUNKING_STRATEGY_CONFIGS,
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

def evaluate_chunking_strategies() -> None:
    """ Evaluates different chunk sizes and overlaps. """
    print("\n--- 🚀 Stage 2: Evaluating Chunking Strategies ---")

    llm_to_test: Groq = initialise_llm()

    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()

    ragas_llm: LlamaIndexLLMWrapper
    ragas_embeddings: HuggingFaceEmbeddings
    ragas_llm, ragas_embeddings = load_ragas_models()

    all_results: list[pd.DataFrame] = []

    for config in CHUNKING_STRATEGY_CONFIGS:

        chunk_size, chunk_overlap = config['size'], config['overlap']

        print(f"--- Testing Chunk Config: size={chunk_size}, "
              f"overlap={chunk_overlap} ---")

        index: VectorStoreIndex = get_or_build_index(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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

        print("--- Running Ragas evaluation for chunking... ---")

        # --- If you don't have a Rate per Minute limit on your API ---
        # results_df: pd.DataFrame = evaluate_without_rate_limit(
        #     qa_dataset,
        #     ragas_llm,
        #     ragas_embeddings,
        # )

        # --- If you do have a Rate per Minute API limit ---
        results_df: pd.DataFrame = evaluate_with_rate_limit(
            qa_dataset,
            ragas_llm,
            ragas_embeddings,
        )

        # Add Chunk Size and Chunk Overlap to DataFrame to help tracking
        results_df['chunk_size'] = chunk_size
        results_df['chunk_overlap'] = chunk_overlap

        all_results.append(results_df)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)

    save_results(final_df, "chunking_evaluation")

    print("--- ✅ Chunking Strategy Evaluation Complete ---")