# embed.py

import numpy as np
from sentence_transformers import SentenceTransformer

# Global model instance
_model = None

def get_batch_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts using
    the tiny paraphrase-albert-small-v2 model.
    """
    global _model
    if _model is None:
        print("Loading paraphrase-albert-small-v2…")
        _model = SentenceTransformer('paraphrase-albert-small-v2')

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings
