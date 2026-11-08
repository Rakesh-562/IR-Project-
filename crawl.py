"""
crawl.py — Phase 1: build an offline corpus of Indian legal documents.

How it works (in rounds):

    Round 0   For every seed query in seeds.txt, take the top results from
              Indian Kanoon's search (PAGES_PER_SEED pages × 10 results).
    Round 1   Collect every document those judgments cite (other judgments,
              Acts, Constitution articles). Fetch the MOST-CITED ones first.
    Round 2   Same again, from the documents fetched in round 1.

    It stops when MAX_DOCS documents are collected or the rounds run out.

Following citations is what brings in landmark cases (e.g. Puttaswamy) that
a single search would miss, and it gives us the citation graph for Phase 8.

Output (in data/):
    corpus.jsonl    one JSON document per line (see make_record for fields)
    citations.tsv   src_doc_id <TAB> dst_doc_id, one line per citation

Resuming: documents already in corpus.jsonl are skipped, so if the crawl is
interrupted just run it again. Search pages are cached in cache/; document
pages are NOT (the text is already saved in corpus.jsonl).

Run:
    python crawl.py                                  # full crawl (hours)
    python crawl.py --max-docs 50 --seed-limit 2     # small test
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from legal_scraper import IK_BASE, _clean, _fetch, make_doc_id

# ── Settings (each can be overridden on the command line) ─────────────────────

SEEDS_FILE     = Path(__file__).with_name("seeds.txt")
OUT_DIR        = Path(__file__).with_name("data")
MAX_DOCS       = 5000   # stop after this many documents
MAX_DEPTH      = 2      # how many rounds of citation-following after round 0
PAGES_PER_SEED = 3      # search result pages per seed (10 results each)
MAX_FAILS_IN_A_ROW = 10 # this many failed fetches in a row = network is down
NETWORK_WAIT   = 120    # then wait this many seconds and try again ...
MAX_WAITS      = 5      # ... up to this many times, then stop

# Judgment text is in <div class="judgments">, statute text in <div class="akoma-ntoso">
TEXT_SELECTOR = ".judgments, .akoma-ntoso"

DOC_NUMBER = re.compile(r"/(?:doc|docfragment)/(\d+)/")

# ── Small helpers ─────────────────────────────────────────────────────────────

def canonical_url(href: str) -> str | None:
    """Turn any Indian Kanoon document link into https://indiankanoon.org/doc/<n>/."""
    match = DOC_NUMBER.search(href)
    return f"{IK_BASE}/doc/{match.group(1)}/" if match else None

def parse_date(title: str) -> str:
    """'X vs Y on 22 October, 2019' → '2019-10-22'. Returns '' if there is no date."""
    match = re.search(r" on (\d{1,2} \w+, \d{4})$", title)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%d %B, %Y").date().isoformat()
    except ValueError:
        return ""

# ── Step 1: search pages → document URLs ──────────────────────────────────────

async def search_urls(query: str, pages: int, client: httpx.AsyncClient) -> list[str]:
    """Return the document URLs from the first `pages` search result pages."""
    urls = []
    for page in range(pages):
        search = f"{IK_BASE}/search/?formInput={quote_plus(query)}&pagenum={page}"
        html = await _fetch(search, client)          # search pages are cached
        if not html:
            break
        soup = BeautifulSoup(html, "html.parser")
        results = soup.select("article.result")
        for result in results:
            link = result.select_one("a[href*='/doc/'], a[href*='/docfragment/']")
            url = canonical_url(link["href"]) if link else None
            if url and url not in urls:
                urls.append(url)
        if len(results) < 10:                        # last page of results
            break
    return urls

# ── Step 2: document page → corpus record ─────────────────────────────────────

def make_record(url: str, html: str, depth: int) -> dict | None:
    """Extract one corpus record from a document page. None if there is no text."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(TEXT_SELECTOR)
    if body is None:
        return None

    # Citations: every link to another document inside the text
    cites = []
    for link in body.select("a[href*='/doc/']"):
        cited = canonical_url(link["href"])
        if cited and cited != url and cited not in cites:
            cites.append(cited)

    title_tag = soup.select_one(".doc_title") or soup.title
    court_tag = soup.select_one(".docsource_main")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    text = _clean(body.get_text(separator=" "))
    if not text:
        return None

    return {
        "doc_id":   make_doc_id("ik", url),
        "url":      url,
        "title":    title,
        "court":    court_tag.get_text(" ", strip=True) if court_tag else "",
        "date":     parse_date(title),
        "doc_type": "statute" if "akoma-ntoso" in body.get("class", []) else "judgment",
        "depth":    depth,          # 0 = found by search, 1+ = found via citations
        "text":     text,
        "cites":    cites,          # URLs of cited documents
    }

# ── Step 3: the crawl loop ────────────────────────────────────────────────────

def load_existing(corpus_file: Path) -> dict[str, list[str]]:
    """Read an earlier (possibly interrupted) crawl: url → list of cited URLs."""
    done, good_lines, bad_lines = {}, [], 0
    if not corpus_file.exists():
        return done
    with corpus_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:     # half-written line: the crawl was killed mid-write
                bad_lines += 1
                continue
            done[record["url"]] = record["cites"]
            good_lines.append(line)
    if bad_lines:                            # rewrite the file without the broken line(s)
        print(f"[crawl] removing {bad_lines} half-written line(s) from {corpus_file}")
        corpus_file.write_text("".join(good_lines), encoding="utf-8")
    return done

async def crawl(seeds: list[str], max_docs: int, max_depth: int,
                pages_per_seed: int, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    corpus_file = out_dir / "corpus.jsonl"
    done = load_existing(corpus_file)
    print(f"[crawl] {len(done)} documents already in {corpus_file}")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        with corpus_file.open("a", encoding="utf-8") as out:

            # Round 0: search results for every seed query
            round_urls = []
            for i, query in enumerate(seeds, 1):
                found = await search_urls(query, pages_per_seed, client)
                round_urls += [u for u in found if u not in round_urls]
                print(f"[search {i}/{len(seeds)}] {len(found):2} results  {query}")

            fails_in_a_row = 0
            waits = 0
            network_down = False
            for depth in range(max_depth + 1):
                print(f"\n[crawl] round {depth}: {len(round_urls)} candidate documents")
                for url in round_urls:
                    if len(done) >= max_docs:
                        break
                    if url in done:
                        continue
                    html = await _fetch(url, client, use_cache=False)
                    if html is None:
                        fails_in_a_row += 1
                        if fails_in_a_row >= MAX_FAILS_IN_A_ROW:
                            if waits >= MAX_WAITS:
                                network_down = True
                                break
                            waits += 1
                            print(f"\n[crawl] {fails_in_a_row} failed fetches in a row — "
                                  f"waiting {NETWORK_WAIT}s for the network "
                                  f"(pause {waits}/{MAX_WAITS})")
                            await asyncio.sleep(NETWORK_WAIT)
                            fails_in_a_row = 0
                        continue
                    fails_in_a_row = 0
                    record = make_record(url, html, depth)
                    if record is None:
                        continue
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()                      # safe if interrupted
                    done[url] = record["cites"]
                    print(f"  [{len(done):5}/{max_docs}] {record['doc_type']:8} "
                          f"{record['title'][:70]}")

                if network_down:
                    print(f"\n[crawl] still failing after {MAX_WAITS} pauses — "
                          "the network or the site seems down. Stopping; "
                          "run crawl.py again later to resume.")
                    break
                if len(done) >= max_docs:
                    break
                # Next round: documents cited by this round, most-cited first
                counts = Counter(c for u in round_urls if u in done for c in done[u])
                round_urls = [u for u, _ in counts.most_common() if u not in done]

    write_citations(done, out_dir / "citations.tsv")
    print(f"\n[crawl] done: {len(done)} documents in {corpus_file}")

def write_citations(done: dict[str, list[str]], path: Path) -> None:
    """Write the citation graph. Cited documents may be outside the corpus."""
    edges = 0
    with path.open("w", encoding="utf-8") as f:
        for url, cites in done.items():
            for cited in cites:
                f.write(f"{make_doc_id('ik', url)}\t{make_doc_id('ik', cited)}\n")
                edges += 1
    print(f"[crawl] {edges} citations written to {path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build the legal corpus.")
    parser.add_argument("--max-docs", type=int, default=MAX_DOCS)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--pages-per-seed", type=int, default=PAGES_PER_SEED)
    parser.add_argument("--seed-limit", type=int, default=None,
                        help="use only the first N seed queries (for testing)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    seeds = [line.strip() for line in SEEDS_FILE.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    seeds = seeds[:args.seed_limit] if args.seed_limit else seeds

    asyncio.run(crawl(seeds, args.max_docs, args.max_depth,
                      args.pages_per_seed, args.out))
