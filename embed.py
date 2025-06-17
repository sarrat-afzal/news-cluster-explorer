# embed.py

import numpy as np
from sentence_transformers import SentenceTransformer

# Global model instance
_model: SentenceTransformer | None = None

def get_batch_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts using
    the much smaller paraphrase-MiniLM-L3-v2 model.
    """
    global _model
    if _model is None:
        print("Loading sentence-transformers/paraphrase-MiniLM-L3-v2...")
        _model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # Encode all texts, returning a NumPy array
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings
