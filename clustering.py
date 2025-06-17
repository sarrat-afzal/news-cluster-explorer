from fetch import fetch_all_news_by_date
from embed import get_batch_embeddings
from faiss_clusterer import FAISSClusterer

def get_clusters_for_date(date: str, n_clusters: int = None):
    data = fetch_all_news_by_date(date, output_file=None)
    articles = data.get('articles', [])
    texts, ids, meta = [], [], []
    for art in articles:
        title = art.get('title', '')
        subtitle = art.get('subtitle', '')
        text = f"{title}. {subtitle}".strip() or art.get('content', '')
        texts.append(text)
        ids.append(art.get('_id'))
        meta.append({
            'title': title,
            'url': art.get('url'),
            'source': art.get('sourceName'),
            'published': art.get('publishedAt')
        })

    embeddings = get_batch_embeddings(texts, batch_size=32)
    clusterer = FAISSClusterer(embeddings, texts, ids)
    clusterer.build_faiss_index('flat')
    labels = clusterer.perform_clustering(n_clusters=n_clusters)
    raw = clusterer.get_cluster_summaries(max_articles_per_cluster=0)

    clusters = []
    for cid, info in raw.items():
        arts = [meta[i] for i, lbl in enumerate(labels) if lbl == cid]
        clusters.append({
            'id': cid,
            'size': info['size'],
            'percentage': round(info['percentage'], 1),
            'articles': arts
        })
    clusters.sort(key=lambda x: x['size'], reverse=True)
    return clusters
