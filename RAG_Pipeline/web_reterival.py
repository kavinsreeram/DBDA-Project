import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


# ======================================================
# STEP 1: SEARCH TRUSTED NEWS / FACT SITES
# ======================================================
def search_fact_check_sites(claim):
    """
    Search trusted websites for news articles related to claim.
    Returns list of unique URLs.
    """

    trusted_sites = [
        "altnews.in",
        "boomlive.in",
        "factcheck.pib.gov.in",
        "indiatoday.in"
    ]

    STOPWORDS = {"india", "scheme", "announced"}
    words = [
        w for w in claim.lower().split()
        if len(w) > 3 and w not in STOPWORDS
    ]

    query_core = " ".join(words[:6])

    results = []

    try:
        with DDGS() as ddgs:
            for site in trusted_sites:
                query = f"{query_core} site:{site}"
                for r in ddgs.text(query, max_results=3):
                    url = r.get("href")
                    if url and url.startswith("http"):
                        results.append(url)
    except Exception as e:
        print("Search error:", e)

    # Fallback open search
    if not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query_core, max_results=5):
                    url = r.get("href")
                    if url and url.startswith("http"):
                        results.append(url)
        except Exception as e:
            print("Fallback search error:", e)

    # Remove duplicates
    return list(dict.fromkeys(results))


# ======================================================
# STEP 2: EXTRACT TITLE + CLEAN ARTICLE TEXT
# ======================================================
def extract_article_content(url):
    """
    Extract title and cleaned article text.
    Returns:
        {
            "title": str,
            "text": str,
            "url": str
        }
    """

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        # Extract title
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text().strip()
        else:
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else "Untitled Article"

        # Extract paragraph text
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text().strip() for p in paragraphs)

        # Clean short lines
        clean_lines = [line for line in text.split("\n") if len(line.strip()) > 40]
        clean_text = "\n".join(clean_lines)

        if len(clean_text) < 200:
            return None

        return {
            "title": title,
            "text": clean_text,
            "url": url
        }

    except Exception as e:
        print("Extraction error:", e)
        return None
