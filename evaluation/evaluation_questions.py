EVALUATION_DATA: list[dict[str, str]] = [
    {
        'question': "How many tokens does GPT-4 break the phrase "
                "'I can't wait to build AI applications' into, and "
                "what is GPT-4's vocabulary size?",
        'ground_truth': "GPT-4 breaks the phrase into nine tokens, splitting "
                    "'can't' into two tokens ('can' and 't'). GPT-4's "
                    "vocabulary size is 100,256 tokens."
    }
    ]