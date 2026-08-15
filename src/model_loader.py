import os
from dotenv import load_dotenv

from llama_index.llms.groq import Groq

from src.config import (
    LLM_MODEL,
    # LLM_MAX_NEW_TOKENS,
    # LLM_TEMPERATURE,
    # LLM_TOP_P,
    # LLM_REPETITION_PENALTY
)


# Load environment variables from the .env file
load_dotenv()


def initialise_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    api_key: str | None = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file."
        )

    return Groq(
        api_key=api_key,
        model=LLM_MODEL,
        # The following parameters are optional
        # and will default to the model's defaults if not set
        # max_tokens=LLM_MAX_NEW_TOKENS,
        # temperature=LLM_TEMPERATURE,
        # top_p=LLM_TOP_P,
    )