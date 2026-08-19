from evaluation.evaluation_engine import (

    #evaluate_baseline,
    #evaluate_chunking_strategies,
    evaluate_reranker_strategies
)


if __name__ == "__main__":

    # Run Stage 1: Baseline Evaluation
    #evaluate_baseline()

    # Run Stage 2: Chunking Strategy Evaluation
    #evaluate_chunking_strategies()

    # Run Stage 3: Reranker Strategy Evaluation
    evaluate_reranker_strategies()