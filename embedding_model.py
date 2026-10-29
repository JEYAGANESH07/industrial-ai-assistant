"""Shared, lazy-loaded sentence embedding model.

Both ingestion and query retrieval import this module, so the model is loaded
once per Python process rather than once in each module.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    print(f"Loading shared embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print("Shared embedding model loaded successfully.")
    return model
