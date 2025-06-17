import faiss
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class FAISSClusterer:
    def __init__(self, embeddings: np.ndarray, texts: list, article_ids: list):
        self.embeddings = embeddings.astype('float32')
        self.texts = texts
        self.article_ids = article_ids
        self.n, self.dim = self.embeddings.shape
        self.index = None
        self.cluster_labels = None

    def build_faiss_index(self, index_type: str = 'flat'):
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError("Only 'flat' supported here")
        self.index.add(self.embeddings)

    def perform_clustering(self, n_clusters: int = None, method: str = 'kmeans'):
        if n_clusters is None:
            n_clusters = max(2, min(20, int(np.sqrt(self.n / 2))))
        if method == 'kmeans':
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = km.fit_predict(self.embeddings)
            self.centroids = km.cluster_centers_
        else:
            raise ValueError("Only 'kmeans' supported here")
        return self.cluster_labels

    def get_cluster_summaries(self, max_articles_per_cluster: int = 5):
        if self.cluster_labels is None:
            raise RuntimeError("No clustering done yet")
        total = len(self.cluster_labels)
        summaries = {}
        for cid in np.unique(self.cluster_labels):
            idxs = np.where(self.cluster_labels == cid)[0]
            size = len(idxs)
            pct = size / total * 100
            samples = []
            for i in idxs[:max_articles_per_cluster]:
                title = self.texts[i].split('\n')[0]
                samples.append({'id': self.article_ids[i], 'title': title})
            summaries[int(cid)] = {'size': size, 'percentage': pct, 'samples': samples}
        return summaries
