from text_embedding import get_embeddings_for_claim
from vector_db import build_faiss_index, faiss_search


# ======================================================
# COMPLETE RAG RETRIEVAL PIPELINE (TOP 3 ARTICLES)
# ======================================================
def run_rag_pipeline(claim):
    """
    Full Retrieval Stage:
    1. Search + Extract
    2. Embed
    3. Build FAISS
    4. Retrieve Top 3 Articles
    """

    # Step 1: Get structured articles + embeddings
    articles, query_emb, article_embs = get_embeddings_for_claim(claim)

    # Safety check
    if not articles or article_embs is None or query_emb is None:
        return []

    # Step 2: Build FAISS index
    index = build_faiss_index(article_embs)

    if index is None:
        return []

    # Step 3: Retrieve Top 3 relevant articles
    top_articles = faiss_search(
        index=index,
        query_emb=query_emb,
        articles=articles,
        top_k=3
    )

    return top_articles


# ======================================================
# DEBUG TEST
# ======================================================
if __name__ == "__main__":
    test_claim = "Government announced a new pension scheme"

    results = run_rag_pipeline(test_claim)

    print("\nTop Retrieved Articles:\n")

    for i, article in enumerate(results, 1):
        print(f"{i}. {article['title']}")
        print(f"URL: {article['url']}")
        print("-" * 80)
