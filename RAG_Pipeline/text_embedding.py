from sentence_transformers import SentenceTransformer
from web_reterival import search_fact_check_sites, extract_article_content


# ======================================================
# Load embedding model (only once)
# ======================================================
model = SentenceTransformer("all-MiniLM-L6-v2")


# ======================================================
# Embedding helpers
# ======================================================
def embed_text(text):
    return model.encode(text, convert_to_numpy=True)


def embed_multiple(text_list):
    return model.encode(text_list, convert_to_numpy=True)


# ======================================================
# MAIN FUNCTION: Retrieve + Embed
# ======================================================
def get_embeddings_for_claim(claim):
    """
    Pipeline:
    1. Search trusted websites
    2. Extract structured article content
    3. Generate embeddings
    """

    print("\n🔍 Searching trusted websites...")
    urls = search_fact_check_sites(claim)

    if not urls:
        print("⚠️ No URLs found.")
        return [], None, None

    print("\n📄 Extracting article content...")
    articles = []

    for url in urls:
        print(f" - Extracting from: {url}")
        article = extract_article_content(url)

        if article and len(article["text"]) > 200:
            articles.append(article)

    if not articles:
        print("⚠️ No valid articles extracted.")
        return [], None, None

    print("\n🔢 Generating embeddings...")

    # Query embedding
    query_emb = embed_text(claim)

    # Article embeddings (use article text only)
    article_texts = [article["text"] for article in articles]
    article_embs = embed_multiple(article_texts)

    return articles, query_emb, article_embs
