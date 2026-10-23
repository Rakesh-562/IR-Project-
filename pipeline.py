"""
pipeline.py
------------
Full end-to-end pipeline:

    User Query
        → legal_scraper.scrape()               [fetch real web pages]
            → relevance_filter.filter_documents()   [LLM judges relevance]
                → bm25_rank()                  [rank passing docs]
                    → print results

Run:
    python pipeline.py "right to privacy Article 21 India"

Optional API keys:
    Tavily    → https://app.tavily.com (additional search results)
    Anthropic → https://console.anthropic.com (LLM relevance judgments)

Without an Anthropic key, relevance filtering uses local BM25 ranking.

Keys may be set in the project .env file. Only Tavily search requires a key;
the direct scrapers and local BM25 fallback work without API keys.
"""

import asyncio
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from legal_scraper import scrape
from relevance_filter import ScrapedDoc, LabeledDoc, filter_documents

# ── Config (reads from env vars — or hardcode here for testing) ───────────────

load_dotenv(Path(__file__).with_name(".env"))

BRAVE_API_KEY  = os.getenv("BRAVE_API_KEY",  "")   # free at api.search.brave.com/app/keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")   # free at app.tavily.com

MAX_PER_SOURCE = 5    # docs fetched per source
BM25_PREFILTER = 25   # top-N passed to LLM relevance filter
MAX_FINAL      = 10   # results shown to user

# ── Ranker ─────────────────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    doc: ScrapedDoc
    relevance: str
    llm_reason: str
    bm25_score: float
    rank: int


def bm25_rank(query: str, labeled: list[LabeledDoc]) -> list[RankedResult]:
    """BM25-rank the LLM-filtered docs. Relevant docs form tier 1, partial tier 2."""

    def rank_tier(tier: list[LabeledDoc], start: int) -> list[RankedResult]:
        if not tier:
            return []
        corpus = [(d.doc.title + " " + d.doc.content).lower().split() for d in tier]
        bm25   = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(tier, scores), key=lambda x: x[1], reverse=True)
        return [
            RankedResult(d.doc, d.relevance, d.reason, float(s), start + i)
            for i, (d, s) in enumerate(ranked)
        ]

    rel  = [d for d in labeled if d.relevance == "relevant"]
    part = [d for d in labeled if d.relevance == "partial"]
    return rank_tier(rel, 1) + rank_tier(part, len(rel) + 1)


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_results(query: str, results: list[RankedResult], buckets: dict) -> None:
    SEP = "─" * 72
    nr  = len(buckets.get("not_relevant", []))
    r   = len(buckets.get("relevant", []))
    p   = len(buckets.get("partial", []))

    print(f"\n{'═'*72}")
    print(f"  LEGAL SEARCH RESULTS")
    print(f"  Query : {query}")
    print(f"{'═'*72}")
    print(f"\n  Filter  →  {r+p+nr} scraped | {r} relevant | {p} partial | {nr} discarded\n")

    if not results:
        print("  No relevant results found.")
        return

    for res in results[:MAX_FINAL]:
        tag = "✓ RELEVANT" if res.relevance == "relevant" else "~ PARTIAL"
        print(SEP)
        print(f"  [{res.rank}]  {tag}   BM25: {res.bm25_score:.3f}")
        print(f"  {res.doc.title}")
        print(f"  {res.doc.url}")
        print(f"  Source : {res.doc.source}")
        print(textwrap.fill(res.llm_reason, 66,
                            initial_indent="  Reason : ",
                            subsequent_indent="           "))
        preview = " ".join(res.doc.content.split()[:35]) + " ..."
        print(textwrap.fill(preview, 66,
                            initial_indent="  Preview: ",
                            subsequent_indent="           "))

    print(SEP)

    if nr:
        print(f"\n  Discarded (not relevant):")
        for d in buckets["not_relevant"]:
            print(f"    ✗  {d.doc.title[:65]}")
            print(f"       {d.reason}")
    print()


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(query: str) -> list[RankedResult]:
    # Step 1 — Scrape
    scraped = await scrape(
        query,
        max_per_source = MAX_PER_SOURCE,
        brave_api_key  = BRAVE_API_KEY,
        tavily_api_key = TAVILY_API_KEY,
    )
    if not scraped:
        print("[pipeline] No documents scraped. Check keys / network.")
        return []

    # Step 2 — Relevance filter
    print(f"[pipeline] Assessing relevance for {len(scraped)} docs ...")
    buckets = await filter_documents(
        query, scraped,
        bm25_top_n=min(BM25_PREFILTER, len(scraped)),
    )

    # Step 3 — Rank
    passing = buckets["relevant"] + buckets["partial"]
    if not passing:
        print("[pipeline] All documents filtered out.")
        print_results(query, [], buckets)
        return []

    print(f"[pipeline] Ranking {len(passing)} passing docs ...")
    ranked = bm25_rank(query, passing)
    print_results(query, ranked, buckets)
    return ranked


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Legal query: ").strip()
    if not q:
        sys.exit("No query provided.")
    asyncio.run(run_pipeline(q))
