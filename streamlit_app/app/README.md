# Streamlit RAG chatbot

`playground_rerank.ipynb` as a web app: semantic chunking, vector search,
cross-encoder reranking, Groq LLM.

```bash
conda activate rag-project-env      # streamlit is installed here
streamlit run app.py                # from the project root
```

Run it in `rag-project-env`, the same environment as the notebooks. Other
environments have older llama-index versions that split the book into much
bigger chunks, and the embedding model only reads the first ~256 tokens of a
chunk, so search gets noticeably worse.

Needs `GROQ_API_KEY` in the project's `.env`. The first start downloads the
reranker (~1.1 GB); if `local_storage/semantic_vector_store/` is missing it
also chunks and embeds `data/` once and saves it there - delete that folder
to force a rebuild.

`app.py` (project root) the UI · `app/app_config.py` every setting ·
`app/app_model_loader.py` the two models · `app/app_engine.py` the RAG
pipeline. Same split as `src/`.
