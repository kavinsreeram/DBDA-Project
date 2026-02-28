import faiss
import numpy as np


# ======================================================
# Build FAISS Index (In-Memory)
# ======================================================
def build_faiss_index(article_embs):
    """
    Builds an in-memory FAISS index from article embeddings.
    """

    if article_embs is None or len(article_embs) == 0:
        return None

    # Ensure 2D shape
    if len(article_embs.shape) == 1:
        article_embs = article_embs.reshape(1, -1)

    dim = article_embs.shape[1]

    # L2 similarity index (stable + version safe)
    index = faiss.IndexFlatL2(dim)
    index.add(article_embs.astype("float32"))

    return index


# ======================================================
# Retrieve Top-K Similar Articles
# ======================================================
def faiss_search(index, query_emb, articles, top_k=3):
    """
    Returns top_k most relevant structured articles.
    """

    if index is None or index.ntotal == 0 or query_emb is None:
        return []

    # Convert query to correct shape
    query_vec = np.array([query_emb]).astype("float32")

    # Ensure we don't exceed available articles
    k = min(top_k, index.ntotal)

    distances, ids = index.search(query_vec, k)

    top_results = []

    for rank, idx in enumerate(ids[0]):
        if 0 <= idx < len(articles):
            article = articles[idx].copy()
            article["score"] = float(distances[0][rank])
            top_results.append(article)

    return top_results
