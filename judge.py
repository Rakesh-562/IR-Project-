"""
judge.py — Phase 4: build the relevance judgments (qrels) by POOLING.

Nobody can judge all 4,730 documents for 30 queries. Pooling: take the top
POOL_DEPTH documents of every run, merge them, and judge only those. A
document no run ranked highly is assumed not relevant.

The judgments live in one spreadsheet-friendly file, eval/judgments.tsv:

    qid  doc_id  query  title  court  snippet  llm_grade  human_grade

  snippet      the ~60 words of the document that best match the query,
               so a judge can usually decide without opening it
  llm_grade    optional first guess from Google Gemini (needs GEMINI_API_KEY)
  human_grade  0 = not relevant, 1 = partly, 2 = highly relevant, typed in
               by a person (open the file in Excel / LibreOffice)

Steps:
    python judge.py pool eval/runs/*.txt   add the top documents of these runs (keeps grades already given)
    python judge.py llm                    fill llm_grade where it is empty
    python judge.py qrels                  write eval/qrels.txt (human grade, else LLM grade)
    python judge.py kappa                  agreement between LLM and human grades (Cohen's κ)
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from evaluate import load_run
from text_processing import analyze

BASE_DIR        = Path(__file__).resolve().parent
QUERIES_FILE    = BASE_DIR / "eval" / "queries.jsonl"
JUDGMENTS_FILE  = BASE_DIR / "eval" / "judgments.tsv"
QRELS_FILE      = BASE_DIR / "eval" / "qrels.txt"
CORPUS_FILE     = BASE_DIR / "data" / "corpus_dedup.jsonl"

POOL_DEPTH   = 10                    # top documents taken from each run
SNIPPET_WORDS = 60
COLUMNS = ["qid", "doc_id", "query", "title", "court", "snippet", "llm_grade", "human_grade"]

load_dotenv(BASE_DIR / ".env")
GEMINI_MODEL = "gemini-flash-latest"      # change here if Google renames its models
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_DELAY = 5                          # seconds between requests: stay under the free-tier limit

def load_queries() -> dict[str, dict]:
    with QUERIES_FILE.open(encoding="utf-8") as f:
        return {q["qid"]: q for q in map(json.loads, f)}

# ── The judgments file ────────────────────────────────────────────────────────

def read_judgments() -> list[dict]:
    if not JUDGMENTS_FILE.exists():
        return []
    with JUDGMENTS_FILE.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_judgments(rows: list[dict]) -> None:
    rows.sort(key=lambda r: (r["qid"], r["doc_id"]))
    with JUDGMENTS_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

# ── Snippets: the part of a document that best matches the query ──────────────

def best_window(text: str, query: str, size: int) -> str:
    """The `size`-word window with the most query terms in it."""
    words = text.split()
    query_terms = set(analyze(query))
    is_hit = [1 if query_terms & set(analyze(w)) else 0 for w in words]   # does each word match?
    starts = range(0, max(1, len(words) - size + 1), 10)                  # try a window every 10 words
    best = max(starts, key=lambda s: sum(is_hit[s:s + size]))
    return " ".join(words[best:best + size])

def load_documents(doc_ids: set[str]) -> dict[str, dict]:
    """Read only the documents we need from the corpus."""
    docs = {}
    with CORPUS_FILE.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["doc_id"] in doc_ids:
                docs[doc["doc_id"]] = doc
    return docs

# ── Commands ──────────────────────────────────────────────────────────────────

def pool(run_files: list[Path]) -> None:
    queries = load_queries()
    rows = read_judgments()
    have = {(r["qid"], r["doc_id"]) for r in rows}

    new_pairs = set()
    for path in run_files:
        for qid, ranked in load_run(path).items():
            for doc_id in ranked[:POOL_DEPTH]:
                if (qid, doc_id) not in have:
                    new_pairs.add((qid, doc_id))

    docs = load_documents({doc_id for _, doc_id in new_pairs})
    for qid, doc_id in sorted(new_pairs):
        doc, query = docs[doc_id], queries[qid]["query"]
        rows.append({
            "qid": qid, "doc_id": doc_id, "query": query,
            "title": doc["title"], "court": doc["court"],
            "snippet": best_window(doc["text"], query, SNIPPET_WORDS),
            "llm_grade": "", "human_grade": "",
        })
    write_judgments(rows)
    print(f"[judge] {len(new_pairs)} new (query, document) pairs; {len(rows)} in {JUDGMENTS_FILE.name}")

def llm_grade(query: dict, doc: dict, api_key: str) -> str:
    """Ask Gemini for a 0/1/2 grade, showing it the best-matching passage."""
    passage = best_window(doc["text"], query["query"], 200)
    prompt = (
        "You are judging search results for an Indian legal search engine.\n\n"
        f"Query: {query['query']}\n"
        f"What counts as relevant: {query['description']}\n\n"
        f"Document title: {doc['title']}\nCourt / source: {doc['court']}\n"
        f"Most relevant passage:\n\"\"\"\n{passage}\n\"\"\"\n\n"
        "Grade the document: 2 = highly relevant (squarely addresses the query), "
        "1 = partly relevant (related, or addresses it only in passing), "
        "0 = not relevant. Answer with the single digit only."
    )
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 256}}
    for attempt in range(4):
        # The key goes in a header, never in the URL (URLs end up in error messages and logs)
        response = httpx.post(GEMINI_URL, headers={"x-goog-api-key": api_key}, json=body, timeout=60)
        if response.status_code != 429:
            break
        wait = 30 * (attempt + 1)                   # rate-limited: wait longer each time
        print(f"  [llm] rate limited, waiting {wait}s")
        time.sleep(wait)
    response.raise_for_status()
    parts = response.json()["candidates"][0]["content"]["parts"]
    answer = "".join(p.get("text", "") for p in parts)
    match = re.search(r"[012]", answer)
    if not match:
        raise ValueError(f"unexpected answer {answer!r}")
    return match.group()

def llm() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set (see .env.example; free key at "
                 "https://aistudio.google.com/apikey). Or grade by hand in "
                 f"{JUDGMENTS_FILE}.")
    queries, rows = load_queries(), read_judgments()
    todo = [r for r in rows if r["llm_grade"] == ""]
    docs = load_documents({r["doc_id"] for r in todo})
    for i, row in enumerate(todo, 1):
        try:
            row["llm_grade"] = llm_grade(queries[row["qid"]], docs[row["doc_id"]], api_key)
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            print(f"  [llm] {row['qid']} {row['doc_id']} failed: {type(e).__name__}: {e}")
        time.sleep(GEMINI_DELAY)
        if i % 20 == 0 or i == len(todo):            # save often, so a crash loses little
            write_judgments(rows)
            print(f"  [llm] {i}/{len(todo)} graded")

def qrels() -> None:
    rows = read_judgments()
    with QRELS_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            grade = r["human_grade"] or r["llm_grade"]     # a person's grade wins
            if grade:
                f.write(f"{r['qid']} 0 {r['doc_id']} {grade}\n")
    human = sum(1 for r in rows if r["human_grade"])
    llm_only = sum(1 for r in rows if r["llm_grade"] and not r["human_grade"])
    print(f"[judge] wrote {QRELS_FILE.name}: {human} human grades, {llm_only} LLM grades, "
          f"{len(rows) - human - llm_only} not yet graded (left out)")

def cohen_kappa(a: list[str], b: list[str]) -> float:
    """κ = (observed agreement − agreement expected by chance) / (1 − chance agreement)."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    chance = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - chance) / (1 - chance) if chance < 1 else 1.0

def kappa() -> None:
    both = [r for r in read_judgments() if r["llm_grade"] and r["human_grade"]]
    if not both:
        sys.exit("No rows have both an LLM and a human grade yet.")
    llm_g, human_g = [r["llm_grade"] for r in both], [r["human_grade"] for r in both]
    binary = lambda grades: ["1" if g != "0" else "0" for g in grades]
    print(f"{len(both)} rows graded by both")
    print(f"  exact agreement (0/1/2) : {sum(x == y for x, y in zip(llm_g, human_g)) / len(both):.0%}"
          f"   κ = {cohen_kappa(llm_g, human_g):.2f}")
    print(f"  relevant vs not (0 / 1+): κ = {cohen_kappa(binary(llm_g), binary(human_g)):.2f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "pool" and len(sys.argv) > 2:
        pool([Path(p) for p in sys.argv[2:]])
    elif command == "llm":
        llm()
    elif command == "qrels":
        qrels()
    elif command == "kappa":
        kappa()
    else:
        sys.exit(__doc__)
