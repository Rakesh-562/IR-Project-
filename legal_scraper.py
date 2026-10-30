"""Retrieve Indian legal documents from public web sources."""

import asyncio
import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup


@dataclass
class ScrapedDoc:
    """One document retrieved by a legal source."""
    doc_id: str
    title: str
    content: str
    url: str = ""
    source: str = ""
    provider: str = ""

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
    text = re.sub(
        r"\b(cookie policy|privacy policy|terms of use|all rights reserved"
        r"|subscribe|advertisement|login|sign up)\b",
        " ", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip()

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


def _tavily_content(raw_content: str, summary: str, title: str) -> str:
    if not raw_content:
        return summary

    content = raw_content
    heading = re.search(
        rf"(?im)^\s*#+\s*{re.escape(title)}\s*$",
        content,
    ) if title else None
    if heading:
        content = content[heading.start():]

    opening = content[:1500].lower()
    link_count = len(re.findall(r"\]\s*\(", opening))
    if (
        "skip to main content" in opening
        or "main navigation" in opening
        or "mobile navigation" in opening
        or link_count >= 4
    ):
        content = summary or content

    return content


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

        content = _parse(doc_html, selector=".judgments")
        if not content or len(content) < 100:
            print(f"  [indiankanoon] no judgment text found at {full_url}")
            continue

        docs.append(ScrapedDoc(
            doc_id  = f"ik_{abs(hash(full_url))}",
            title   = title,
            content = _truncate(content),
            url     = full_url,
            source  = "indiankanoon.org",
            provider = "Indian Kanoon",
        ))
        print(f"  [indiankanoon] ✓ {title[:65]}")

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Source 2 — India Code public search API
# ══════════════════════════════════════════════════════════════════════════════

IC_BASE = "https://indiacode.gov.in"
IC_SEARCH = f"{IC_BASE}/server/api/discover/search/objects"


def _metadata_text(values: list[dict]) -> str:
    parts = [
        BeautifulSoup(str(item.get("value", "")), "html.parser").get_text(" ", strip=True)
        for item in values
    ]
    return " ".join(part for part in parts if part)

async def scrape_indiacode(
    query: str,
    client: httpx.AsyncClient,
    max_results: int = 3,
) -> list[ScrapedDoc]:
    """Search India Code's public DSpace API for Acts, sections, and statutes."""
    try:
        response = await client.get(
            IC_SEARCH,
            params={"query": query, "size": max_results, "page": 0},
            headers={"Accept": "application/json"},
            timeout=FETCH_TIMEOUT,
        )
        response.raise_for_status()
        search_data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"  [indiacode] search API failed: {type(exc).__name__}: {exc}")
        return []

    search_result = (
        search_data.get("_embedded", {})
        .get("searchResult", {})
        .get("_embedded", {})
    )
    objects = search_result.get("objects", [])
    if not objects:
        print(f"  [indiacode] no results for query: {query}")
        return []

    docs: list[ScrapedDoc] = []
    for result in objects[:max_results]:
        item = result.get("_embedded", {}).get("indexableObject", {})
        metadata = item.get("metadata", {})

        title = _metadata_text(metadata.get("dc.title", [])) or item.get("name", "")
        if not title:
            continue

        content_parts = [
            title,
            _metadata_text(metadata.get("dc.identifier.act_name", [])),
            _metadata_text(metadata.get("dc.identifier.act_year", [])),
            _metadata_text(metadata.get("dc.identifier.section_number", [])),
            _metadata_text(metadata.get("dc.identifier.section_page_note", [])),
            _metadata_text(metadata.get("dc.description", [])),
            _metadata_text(metadata.get("dc.description.abstract", [])),
        ]
        content = _clean(" ".join(part for part in content_parts if part))
        handle = item.get("handle", "")
        if not handle or not content:
            print(f"  [indiacode] incomplete search result skipped: {title[:65]}")
            continue
        full_url = urljoin(f"{IC_BASE}/", f"handle/{handle}")
        docs.append(ScrapedDoc(
            doc_id  = f"ic_{item.get('uuid') or item.get('id') or abs(hash(full_url))}",
            title   = title,
            content = _truncate(content),
            url     = full_url,
            source  = "indiacode.gov.in",
            provider = "India Code",
        ))
        print(f"  [indiacode]    ✓ {title[:65]}")

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Source 3 — Tavily Search API
# ══════════════════════════════════════════════════════════════════════════════

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

async def scrape_tavily(
    query: str,
    client: httpx.AsyncClient,
    api_key: str,
    max_results: int = 5,
) -> list[ScrapedDoc]:
    """Fetch search results and extracted page content from Tavily."""
    if not api_key:
        print("  [tavily] no API key — skipping.")
        return []

    payload = {
        "api_key":              api_key,
        "query":                f"{query} India law court judgment",
        "search_depth":         "advanced",   # deeper extraction (free)
        "include_raw_content":  True,         # get full page text
        "max_results":          max_results,
        "include_domains": [
            "indiankanoon.org",
            "indiacode.gov.in",
            "api.sci.gov.in",
            "sci.gov.in",
            "ecourts.gov.in",
            "egazette.gov.in",
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
        content = _tavily_content(
            item.get("raw_content", ""),
            item.get("content", ""),
            title,
        )
        source   = re.sub(r"https?://(www\.)?", "", page_url).split("/")[0]

        if not content or not page_url:
            continue

        docs.append(ScrapedDoc(
            doc_id  = f"tavily_{abs(hash(page_url))}",
            title   = title,
            content = _truncate(_clean(content)),
            url     = page_url,
            source  = source,
            provider = "Tavily",
        ))
        print(f"  [tavily]       ✓ {title[:65]}")

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Main scrape() function
# ══════════════════════════════════════════════════════════════════════════════

async def scrape(
    query: str,
    max_per_source: int = 5,
    tavily_api_key: str = "",
) -> list[ScrapedDoc]:
    """Scrape configured sources and return documents deduplicated by URL."""
    print(f"\n{'─'*68}")
    print(f"[scraper] Query      : {query}")
    print(f"[scraper] Sources    : Indian Kanoon + India Code"
          + (" + Tavily" if tavily_api_key else ""))
    print(f"{'─'*68}\n")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            scrape_indiankanoon(query, client, max_per_source),
            scrape_indiacode(query, client, max(1, max_per_source // 2)),
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
