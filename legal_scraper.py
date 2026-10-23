"""
legal_scraper.py  (v2 — fully free, no paid APIs)
--------------------------------------------------
Sources used:
  1. Indian Kanoon website  — direct scrape (free, no API key)
  2. India Code             — direct scrape (free government site)
  3. Brave Search API       — free tier, 2000 queries/month, NO credit card
  4. Tavily Search API      — free tier, 1000 queries/month, NO credit card

Get your free keys (5 min total):
  Brave  → https://api.search.brave.com/app/keys   (sign up, copy key)
  Tavily → https://app.tavily.com                  (sign up, copy key)

Both are genuinely free with no card required.

Usage:
    docs = asyncio.run(scrape(query, brave_api_key="...", tavily_api_key="..."))
"""

import asyncio
import re
import time
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup

from relevance_filter import ScrapedDoc

# ── Constants ─────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

FETCH_TIMEOUT  = 15     # seconds per page fetch
MAX_WORDS      = 1200   # words kept per document
SCRAPE_DELAY   = 1.0    # polite delay between requests to same host

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"(cookie policy|privacy policy|terms of use|all rights reserved"
        r"|subscribe|advertisement|login|sign up)[^\n]*",
        "", text, flags=re.IGNORECASE
    )
    return text.strip()

def _truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

async def _fetch(url: str, client: httpx.AsyncClient,
                 extra_headers: dict | None = None) -> str | None:
    h = {**HEADERS, **(extra_headers or {})}
    try:
        r = await client.get(url, headers=h, timeout=FETCH_TIMEOUT,
                             follow_redirects=True)
        if r.status_code != 200:
            print(f"  [fetch] ✗ {url[:60]} → HTTP {r.status_code}")
            return None
        return r.text
    except httpx.HTTPError as e:
        print(f"  [fetch] ✗ {url[:60]} → {type(e).__name__}: {e!r}")
        return None

def _parse(html: str, selector: str | None = None) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "form", "iframe"]):
        tag.decompose()
    node = soup.select_one(selector) if selector else None
    src  = node or soup.find("body")
    return _clean(src.get_text(separator=" ")) if src else ""


# ══════════════════════════════════════════════════════════════════════════════
# Source 1 — Indian Kanoon  (FREE — direct website scrape)
# URL pattern: https://indiankanoon.org/search/?formInput=<query>
# No API key needed. Uses the public search page + judgment pages.
# ══════════════════════════════════════════════════════════════════════════════

IK_BASE   = "https://indiankanoon.org"
IK_SEARCH = IK_BASE + "/search/?formInput={query}&pagenum=0"

async def scrape_indiankanoon(
    query: str,
    client: httpx.AsyncClient,
    max_results: int = 6,
) -> list[ScrapedDoc]:
    """
    Scrapes Indian Kanoon search page to get judgment links,
    then fetches each judgment's full text.
    No API key required.
    """
    search_url = IK_SEARCH.format(query=quote_plus(query))
    html = await _fetch(search_url, client)
    if not html:
        print("  [indiankanoon] could not reach search page")
        return []

    soup  = BeautifulSoup(html, "html.parser")
    # Indian Kanoon currently renders search results as <article class="result">.
    items = soup.select("article.result")[:max_results]

    if not items:
        print("  [indiankanoon] no results found (site layout may have changed)")
        return []

    docs: list[ScrapedDoc] = []
    for item in items:
        title_link = item.select_one("a[href^='/docfragment/'], a[href^='/doc/']")
        document_link = item.select_one("a.cite_tag[href^='/doc/']")
        if title_link is None:
            continue

        title = title_link.get_text(strip=True)
        href = document_link.get("href", "") if document_link else title_link.get("href", "")
        if not href:
            continue
        full_url = IK_BASE + href if href.startswith("/") else href

        await asyncio.sleep(SCRAPE_DELAY)
        doc_html = await _fetch(full_url, client)
        if not doc_html:
            continue

        # Main judgment text is inside <div id="judgments"> on IK pages
        content = _parse(doc_html, selector="#judgments")
        if not content:
            content = _parse(doc_html)           # fallback: whole body
        if not content or len(content) < 100:
            continue

        # Try to extract court name from the result snippet
        source = "indiankanoon.org"

        docs.append(ScrapedDoc(
            doc_id  = f"ik_{abs(hash(full_url))}",
            title   = title,
            content = _truncate(content),
            url     = full_url,
            source  = source,
        ))
        print(f"  [indiankanoon] ✓ {title[:65]}")

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Source 2 — India Code  (FREE — official government site)
# https://www.indiacode.nic.in  — central acts and statutes
# ══════════════════════════════════════════════════════════════════════════════

IC_SEARCH = ("https://www.indiacode.nic.in/handle/123456789/1362/simple-search"
             "?query={query}&rpp={n}&sort_by=score&order=desc")

async def scrape_indiacode(
    query: str,
    client: httpx.AsyncClient,
    max_results: int = 3,
) -> list[ScrapedDoc]:
    """Scrape India Code for Acts and statutes — fully free."""
    url  = IC_SEARCH.format(query=quote_plus(query), n=max_results)
    html = await _fetch(url, client)
    if not html:
        return []

    soup  = BeautifulSoup(html, "html.parser")
    links = soup.select("div.artifact-title a")[:max_results]
    docs  = []

    for link in links:
        title    = link.get_text(strip=True)
        href     = link.get("href", "")
        if not href or len(title) < 5:
            continue
        full_url = urljoin("https://www.indiacode.nic.in", href)

        await asyncio.sleep(SCRAPE_DELAY)
        doc_html = await _fetch(full_url, client)
        content  = _parse(doc_html, "div.item-page") if doc_html else ""
        if not content:
            continue

        docs.append(ScrapedDoc(
            doc_id  = f"ic_{abs(hash(full_url))}",
            title   = title,
            content = _truncate(content),
            url     = full_url,
            source  = "indiacode.nic.in",
        ))
        print(f"  [indiacode]    ✓ {title[:65]}")

    return docs


# # ══════════════════════════════════════════════════════════════════════════════
# # Source 3 — Brave Search API  (FREE — 2000 queries/month, NO credit card)
# # Sign up: https://api.search.brave.com/app/keys
# # Returns web results + snippets; we then fetch each page for full content.
# # ══════════════════════════════════════════════════════════════════════════════

# BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# # Legal sites to restrict results to (passed as site: operators in query)
# LEGAL_SITE_FILTER = (
#     "site:indiankanoon.org OR site:indiacode.nic.in OR "
#     "site:legalserviceindia.com OR site:main.sci.gov.in OR "
#     "site:judis.nic.in OR site:bareactslive.com OR "
#     "site:manupatra.com OR site:scconline.com"
# )

# async def scrape_brave(
#     query: str,
#     client: httpx.AsyncClient,
#     api_key: str,
#     max_results: int = 5,
# ) -> list[ScrapedDoc]:
#     """
#     Brave Search API — completely free, 2000 queries/month.
#     Get your free key at: https://api.search.brave.com/app/keys
#     (No credit card required)
#     """
#     if not api_key:
#         print("  [brave] no API key — skipping. "
#               "Get one FREE (no card) at https://api.search.brave.com/app/keys")
#         return []

#     # Append legal site filter so results are from trusted legal sources
#     full_query = f"{query} Indian law ({LEGAL_SITE_FILTER})"

#     headers = {
#         **HEADERS,
#         "Accept": "application/json",
#         "Accept-Encoding": "gzip",
#         "X-Subscription-Token": api_key,
#     }
#     params = {
#         "q":     full_query,
#         "count": min(max_results, 20),
#         "safesearch": "off",
#         "country": "IN",
#         "search_lang": "en",
#     }

#     try:
#         r = await client.get(BRAVE_SEARCH_URL, headers=headers,
#                              params=params, timeout=FETCH_TIMEOUT)
#         r.raise_for_status()
#         web_results = r.json().get("web", {}).get("results", [])
#     except Exception as e:
#         print(f"  [brave] API error: {e}")
#         return []

#     docs = []
#     for item in web_results[:max_results]:
#         title    = item.get("title", "")
#         page_url = item.get("url", "")
#         snippet  = item.get("description", "")
#         source   = item.get("profile", {}).get("long_name", "") or \
#                    item.get("meta_url", {}).get("hostname", "")

#         if not page_url:
#             continue

#         await asyncio.sleep(SCRAPE_DELAY)
#         doc_html = await _fetch(page_url, client)
#         content  = _parse(doc_html) if doc_html else snippet
#         if not content:
#             content = snippet
#         if len(content) < 80:
#             continue

#         docs.append(ScrapedDoc(
#             doc_id  = f"brave_{abs(hash(page_url))}",
#             title   = title,
#             content = _truncate(content),
#             url     = page_url,
#             source  = source,
#         ))
#         print(f"  [brave]        ✓ {title[:65]}")

#     return docs


# ══════════════════════════════════════════════════════════════════════════════
# Source 4 — Tavily Search API  (FREE — 1000 queries/month, NO credit card)
# Sign up: https://app.tavily.com
# Designed for AI/research use — returns clean extracted content automatically.
# Best feature: returns full page content without needing a second fetch step.
# ══════════════════════════════════════════════════════════════════════════════

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

async def scrape_tavily(
    query: str,
    client: httpx.AsyncClient,
    api_key: str,
    max_results: int = 5,
) -> list[ScrapedDoc]:
    """
    Tavily Search API — free tier 1000 queries/month, no card.
    Get your key at: https://app.tavily.com

    Best for this project: Tavily returns pre-extracted clean text
    from each page, so no second fetch needed.
    """
    if not api_key:
        print("  [tavily] no API key — skipping. "
              "Get one FREE at https://app.tavily.com")
        return []

    payload = {
        "api_key":              api_key,
        "query":                f"{query} India law court judgment",
        "search_depth":         "advanced",   # deeper extraction (free)
        "include_raw_content":  True,         # get full page text
        "max_results":          max_results,
        "include_domains": [
            "indiankanoon.org",
            "indiacode.nic.in",
            "legalserviceindia.com",
            "main.sci.gov.in",
            "judis.nic.in",
            "bareactslive.com",
            "lawrato.com",
        ],
    }

    try:
        r = await client.post(TAVILY_SEARCH_URL, json=payload,
                              timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"  [tavily] API error: {e}")
        return []

    docs = []
    for item in results[:max_results]:
        title    = item.get("title", "")
        page_url = item.get("url", "")
        # Tavily returns raw_content (full page text) when include_raw_content=True
        content  = item.get("raw_content") or item.get("content", "")
        source   = re.sub(r"https?://(www\.)?", "", page_url).split("/")[0]

        if not content or not page_url:
            continue

        docs.append(ScrapedDoc(
            doc_id  = f"tavily_{abs(hash(page_url))}",
            title   = title,
            content = _truncate(_clean(content)),
            url     = page_url,
            source  = source,
        ))
        print(f"  [tavily]       ✓ {title[:65]}")

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Main scrape() function
# ══════════════════════════════════════════════════════════════════════════════

async def scrape(
    query: str,
    max_per_source: int = 5,
    brave_api_key: str  = "",
    tavily_api_key: str = "",
) -> list[ScrapedDoc]:
    """
    Scrape all sources for the query. Returns combined, deduplicated ScrapedDocs.

    Minimum setup (zero cost):
        - No keys at all → only Indian Kanoon + India Code scraped (still useful!)

    Recommended free setup (5 min, no card):
        - brave_api_key  from https://api.search.brave.com/app/keys
        - tavily_api_key from https://app.tavily.com

    Args:
        query          : the user's legal search query
        max_per_source : how many docs to pull from each source
        brave_api_key  : Brave Search free API key
        tavily_api_key : Tavily Search free API key
    """
    print(f"\n{'─'*68}")
    print(f"[scraper] Query      : {query}")
    print(f"[scraper] Sources    : IndianKanoon + IndiaCode"
          + (" + Brave" if brave_api_key else "")
          + (" + Tavily" if tavily_api_key else ""))
    print(f"{'─'*68}\n")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            scrape_indiankanoon(query, client, max_per_source),
            scrape_indiacode(query, client, max(1, max_per_source // 2)),
            # scrape_brave(query, client, brave_api_key, max_per_source),
            scrape_tavily(query, client, tavily_api_key, max_per_source),
            return_exceptions=True,
        )

    all_docs: list[ScrapedDoc] = []
    for batch in batches:
        if isinstance(batch, list):
            all_docs.extend(batch)
        elif isinstance(batch, Exception):
            print(f"[scraper] source error: {batch}")

    # Deduplicate by URL
    seen: set[str] = set()
    deduped = [d for d in all_docs
               if d.url not in seen and not seen.add(d.url)]

    print(f"\n[scraper] Total: {len(all_docs)} scraped  →  "
          f"{len(deduped)} after dedup\n")
    return deduped
