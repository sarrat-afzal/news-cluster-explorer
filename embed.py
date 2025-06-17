import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Union, List

class TextEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _average_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        token_embeddings = hidden_states.masked_fill(~mask, 0.0)
        summed = token_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).unsqueeze(-1)
        return summed / counts

    def embed(self, texts: Union[str, List[str]], max_length: int = 512) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        inputs = [f"query: {t}" for t in texts]
        enc = self.tokenizer(
            inputs, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            pooled = self._average_pool(out.last_hidden_state, enc["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

_embedder: TextEmbedder = None

def get_batch_embeddings(texts: List[str], max_length: int = 512, batch_size: int = 32) -> np.ndarray:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs.append(_embedder.embed(batch, max_length))
    return np.vstack(embs)
