"""
pipeline.py
------------
Full end-to-end pipeline:

    User Query
        → legal_scraper.scrape()  [fetch legal documents]
            → bm25_rank()         [rank all retrieved documents]
                → print results

Run:
    python pipeline.py "right to privacy Article 21 India"

Optional API keys:
    Tavily → https://app.tavily.com (additional search results)

Keys may be set in the project .env file. Only Tavily search requires a key;
the direct scrapers and BM25 ranking work without API keys.
"""

import asyncio
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from legal_scraper import ScrapedDoc, scrape

# ── Config (reads from env vars — or hardcode here for testing) ───────────────

load_dotenv(Path(__file__).with_name(".env"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")   # free at app.tavily.com

MAX_PER_SOURCE = 20   # per-source retrieval cap, not a full archive crawl
MAX_FINAL      = 10   # results shown to user

# ── Ranker ─────────────────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    doc: ScrapedDoc
    bm25_score: float
    rank: int


def bm25_rank(query: str, docs: list[ScrapedDoc]) -> list[RankedResult]:
    """Rank every retrieved document by BM25 keyword relevance."""
    if not docs:
        return []
    corpus = [(doc.title + " " + doc.content).lower().split() for doc in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [
        RankedResult(doc, float(score), index)
        for index, (doc, score) in enumerate(ranked, start=1)
    ]


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_results(query: str, results: list[RankedResult], retrieved_count: int) -> None:
    SEP = "─" * 72

    print(f"\n{'═'*72}")
    print(f"  LEGAL SEARCH RESULTS")
    print(f"  Query : {query}")
    print(f"{'═'*72}")
    print(
        f"\n  Retrieved {retrieved_count} documents; ranked all; "
        f"showing up to {MAX_FINAL}.\n"
    )

    if results:
        shown = results[:MAX_FINAL]
        providers = sorted({res.doc.provider or res.doc.source for res in results})
        print("  Source coverage (retrieved / shown / best rank):")
        for provider in providers:
            provider_results = [
                res for res in results
                if (res.doc.provider or res.doc.source) == provider
            ]
            shown_count = sum(
                1 for res in shown
                if (res.doc.provider or res.doc.source) == provider
            )
            best_rank = min(res.rank for res in provider_results)
            print(
                f"    {provider}: {len(provider_results)} / "
                f"{shown_count} / {best_rank}"
            )
        print()

    if not results:
        print("  No documents found.")
        return

    for res in results[:MAX_FINAL]:
        print(SEP)
        print(f"  [{res.rank}]  BM25: {res.bm25_score:.3f}")
        print(f"  {res.doc.title}")
        print(f"  {res.doc.url}")
        print(f"  Source : {res.doc.source}")
        words = res.doc.content.split()
        preview = " ".join(words[:35])
        if len(words) > 35:
            preview += " ..."
        print(textwrap.fill(preview, 66,
                            initial_indent="  Preview: ",
                            subsequent_indent="           "))

    print(SEP)
    print()


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(query: str) -> list[RankedResult]:
    # Step 1 — Scrape
    scraped = await scrape(
        query,
        max_per_source = MAX_PER_SOURCE,
        tavily_api_key = TAVILY_API_KEY,
    )
    if not scraped:
        print("[pipeline] No documents scraped. Check keys / network.")
        return []

    print(f"[pipeline] Ranking all {len(scraped)} retrieved documents ...")
    ranked = bm25_rank(query, scraped)
    print_results(query, ranked, len(scraped))
    return ranked


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Legal query: ").strip()
    if not q:
        sys.exit("No query provided.")
    asyncio.run(run_pipeline(q))
