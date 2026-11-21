"""
ik_baseline.py — Phase 4: Indian Kanoon's own ranking, as a baseline run.

For each evaluation query, take Indian Kanoon's top results (its search
pages, the same ones crawl.py reads), keep those that are in OUR corpus, in
Indian Kanoon's order, and write them as a run. Our ranking models (Phase 5)
are compared against it: can we rank our corpus better than the site does?

Search pages are cached in cache/, so re-running gives the same run.

Run:
    python ik_baseline.py          # → eval/runs/ik_native.txt
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

from crawl import search_urls
from evaluate import write_run
from judge import load_queries
from legal_scraper import make_doc_id

BASE_DIR    = Path(__file__).resolve().parent
CORPUS_FILE = BASE_DIR / "data" / "corpus_dedup.jsonl"
RUN_FILE    = BASE_DIR / "eval" / "runs" / "ik_native.txt"
PAGES       = 5                  # 5 search pages × 10 results = Indian Kanoon's top 50

async def main() -> None:
    with CORPUS_FILE.open(encoding="utf-8") as f:
        in_corpus = {json.loads(line)["doc_id"] for line in f}

    results = {}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for qid, q in load_queries().items():
            urls = await search_urls(q["query"], PAGES, client)
            ranked = [make_doc_id("ik", u) for u in urls]
            kept = [d for d in ranked if d in in_corpus]
            # score = 1 / Indian Kanoon's rank, so higher is better as in every run
            results[qid] = [(d, 1 / (ranked.index(d) + 1)) for d in kept]
            print(f"  {qid}: {len(urls):2} results from Indian Kanoon, {len(kept):2} in our corpus  {q['query']}")
    write_run(results, "ik_native", RUN_FILE)
    print(f"[ik_baseline] wrote {RUN_FILE}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
