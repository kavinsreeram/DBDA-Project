from rag_pipeline import run_rag_pipeline
from llm_summary import generate_news_summary


def fact_check(claim):
    """
    End-to-end RAG News Retrieval + Summarization
    """

    top_articles = run_rag_pipeline(claim)

    if not top_articles:
        return {
            "title": "No Articles Found",
            "claim": claim,
            "summary": "No reliable articles were found for this query.",
            "sources": []
        }

    summary = generate_news_summary(claim, top_articles)

    title = top_articles[0]["title"]
    sources = [article["url"] for article in top_articles]

    return {
        "title": title,
        "claim": claim,
        "summary": summary,
        "sources": sources
    }
