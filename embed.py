# embed.py
import numpy as np
from sentence_transformers import SentenceTransformer

# Global model instance
_model = None

def get_batch_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts using
    the intfloat/multilingual-e5-large model.
    """
    global _model
    if _model is None:
        print("Loading intfloat/multilingual-e5-large…")
        _model = SentenceTransformer('intfloat/multilingual-e5-large')

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings

