from llama_index.core.llms import CompletionResponse, LLM

from src.config import LLM_QUESTION
from src.model_loader import initialise_llm


def main_chat_loop() -> None:
    """Main chat loop to ask a question to the LLM and print the answer."""

    llm: LLM = initialise_llm()

    answer: CompletionResponse = llm.complete(LLM_QUESTION)

    print(f"Question: {LLM_QUESTION}")
    print("---")
    print(f"Answer: {answer}")