"""
relevance_filter.py
--------------------
LLM-based relevance filter for a legal search engine.

Pipeline:
    query + scraped docs
        → cheap BM25 pre-filter (top-N candidates)
            → async LLM relevance judgment
                → labeled results ready for ranker

Usage:
    results = asyncio.run(filter_documents(query, scraped_docs))
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx
from rank_bm25 import BM25Okapi  # pip install rank-bm25

# ── Constants ────────────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5"          # fast + cheap; swap to sonnet for accuracy
MAX_TOKENS = 256                    # enough for JSON label + one-line reason
TEMPERATURE = 0                     # deterministic judgments
EXCERPT_WORDS = 400                 # how much of each doc to send to LLM
BM25_TOP_N = 25                     # pre-filter: only top-N go to LLM
MAX_CONCURRENT = 8                  # parallel LLM calls at once

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ScrapedDoc:
    """One document retrieved from the web scraper."""
    doc_id: str
    title: str
    content: str                    # full text
    url: str = ""
    source: str = ""                # e.g. "indiankanoon.org", "legalserviceindia.com"


@dataclass
class LabeledDoc:
    """A document after relevance judgment."""
    doc: ScrapedDoc
    relevance: Literal["relevant", "partial", "not_relevant"]
    reason: str
    bm25_score: float = 0.0
    llm_latency_ms: float = 0.0


# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(query: str, doc: ScrapedDoc) -> str:
    """Build the relevance-assessment prompt for one doc."""
    # Take first EXCERPT_WORDS words of content as the excerpt
    words = doc.content.split()
    excerpt = " ".join(words[:EXCERPT_WORDS])
    if len(words) > EXCERPT_WORDS:
        excerpt += " ..."

    return f"""You are a legal relevance assessor for an Indian court-case and law search engine.

Query: "{query}"

Document:
  Title  : {doc.title}
  Source : {doc.source or doc.url}
  Excerpt:
\"\"\"
{excerpt}
\"\"\"

Task:
Decide whether this document is relevant to the query.

A document is RELEVANT if it:
- Directly addresses the legal right, incident, or issue stated in the query
- Contains a court judgment, statute, constitutional article, or legal commentary
  that applies to the query situation
- Would genuinely help a person understand their legal position regarding the query

A document is PARTIAL if it:
- Discusses a related legal area but does not squarely address the query
- Mentions the topic only in passing

A document is NOT_RELEVANT if it:
- Is about an unrelated legal or non-legal subject
- Is a generic article with no substantive legal content on the query topic

Respond ONLY with valid JSON (no markdown, no explanation outside the JSON):
{{
  "relevance": "relevant" | "partial" | "not_relevant",
  "reason": "<one concise sentence>"
}}"""


# ── BM25 pre-filter ──────────────────────────────────────────────────────────

def bm25_prefilter(
    query: str,
    docs: list[ScrapedDoc],
    top_n: int = BM25_TOP_N,
) -> list[tuple[ScrapedDoc, float]]:
    """
    Rank all scraped docs with BM25 and return the top-N with their scores.
    This avoids sending every doc to the LLM.
    """
    tokenized_corpus = [
        (doc.title + " " + doc.content).lower().split()
        for doc in docs
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


# ── LLM call ─────────────────────────────────────────────────────────────────

async def llm_judge(
    query: str,
    doc: ScrapedDoc,
    bm25_score: float,
    api_key: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> LabeledDoc:
    """Call the Anthropic API to get a relevance label for one document."""
    prompt = build_prompt(query, doc)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with semaphore:
        t0 = time.monotonic()
        try:
            response = await client.post(
                ANTHROPIC_API_URL,
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.monotonic() - t0) * 1000

            raw_text = data["content"][0]["text"].strip()

            # Strip markdown fences if the model adds them despite instructions
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)
            relevance = parsed.get("relevance", "not_relevant").lower()
            reason = parsed.get("reason", "")

            # Normalise to known labels
            if relevance not in ("relevant", "partial", "not_relevant"):
                relevance = "not_relevant"

        except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as exc:
            # On any failure, default to partial so we don't silently drop docs
            relevance = "partial"
            reason = f"[filter error: {exc}]"
            latency_ms = (time.monotonic() - t0) * 1000

    return LabeledDoc(
        doc=doc,
        relevance=relevance,
        reason=reason,
        bm25_score=bm25_score,
        llm_latency_ms=latency_ms,
    )


# ── Main filter function ──────────────────────────────────────────────────────

async def filter_documents(
    query: str,
    scraped_docs: list[ScrapedDoc],
    bm25_top_n: int = BM25_TOP_N,
    max_concurrent: int = MAX_CONCURRENT,
) -> dict[str, list[LabeledDoc]]:
    """
    BM25 pre-filter → parallel LLM judgment when configured.
    Without an Anthropic key, BM25 candidates pass through as partial matches.

    Returns a dict with keys:
        "relevant"     – pass these to your ranker
        "partial"      – pass with a lower prior score
        "not_relevant" – discard (kept for analysis / logging)
    """
    if not scraped_docs:
        return {"relevant": [], "partial": [], "not_relevant": []}

    # Step 1 — cheap BM25 pre-filter
    candidates = bm25_prefilter(query, scraped_docs, top_n=bm25_top_n)
    print(f"[filter] {len(scraped_docs)} scraped → {len(candidates)} BM25 candidates")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "[filter] ANTHROPIC_API_KEY not set; using local BM25 ranking "
            "without LLM relevance judgments."
        )
        buckets: dict[str, list[LabeledDoc]] = {
            "relevant": [],
            "partial": [
                LabeledDoc(
                    doc=doc,
                    relevance="partial",
                    reason="Retained by local BM25 matching; no LLM judgment was run.",
                    bm25_score=float(score),
                )
                for doc, score in candidates
            ],
            "not_relevant": [],
        }
        print(
            f"[filter] relevant=0  partial={len(buckets['partial'])}  "
            "not_relevant=0 (local BM25 fallback)"
        )
        return buckets

    # Step 2 — parallel LLM judgment
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            llm_judge(query, doc, score, api_key, client, semaphore)
            for doc, score in candidates
        ]
        labeled: list[LabeledDoc] = await asyncio.gather(*tasks)

    # Step 3 — bucket by label
    buckets: dict[str, list[LabeledDoc]] = {
        "relevant": [],
        "partial": [],
        "not_relevant": [],
    }
    for item in labeled:
        buckets[item.relevance].append(item)

    # Sort each bucket by BM25 score descending (ranker will re-sort by its own metric)
    for key in buckets:
        buckets[key].sort(key=lambda x: x.bm25_score, reverse=True)

    print(
        f"[filter] relevant={len(buckets['relevant'])}  "
        f"partial={len(buckets['partial'])}  "
        f"not_relevant={len(buckets['not_relevant'])}"
    )
    return buckets


# ── What the ranker receives ──────────────────────────────────────────────────

def get_ranker_input(buckets: dict[str, list[LabeledDoc]]) -> list[ScrapedDoc]:
    """
    Combine relevant + partial docs into a single list for the ranker.
    Relevant docs come first; partial docs are appended after.
    """
    return (
        [item.doc for item in buckets["relevant"]]
        + [item.doc for item in buckets["partial"]]
    )


# ── Demo / smoke test ─────────────────────────────────────────────────────────

async def _demo():
    query = "right to privacy Article 21 surveillance India"

    # Simulate scraped docs
    sample_docs = [
        ScrapedDoc(
            doc_id="d1",
            title="Justice K.S. Puttaswamy v. Union of India – Supreme Court Judgment",
            content=(
                "The nine-judge bench of the Supreme Court unanimously held that "
                "the right to privacy is a fundamental right protected under Article 21 "
                "of the Constitution of India. The judgment overruled earlier decisions "
                "and established privacy as an intrinsic part of life and personal liberty..."
            ),
            url="https://indiankanoon.org/doc/127517806/",
            source="indiankanoon.org",
        ),
        ScrapedDoc(
            doc_id="d2",
            title="Aadhaar Act – Section 33 Disclosure",
            content=(
                "Section 33 of the Aadhaar Act permits disclosure of identity information "
                "in the interest of national security when ordered by an officer not below "
                "the rank of Joint Secretary. Critics argue this provision conflicts with "
                "the privacy ruling in Puttaswamy..."
            ),
            url="https://prsindia.org/aadhaar",
            source="prsindia.org",
        ),
        ScrapedDoc(
            doc_id="d3",
            title="How to File Income Tax Returns Online in India",
            content=(
                "Step 1: Log in to the Income Tax e-filing portal. "
                "Step 2: Select the appropriate ITR form based on your income sources. "
                "Step 3: Fill in the required details and verify using Aadhaar OTP..."
            ),
            url="https://incometax.gov.in/guide",
            source="incometax.gov.in",
        ),
    ]

    print(f"\nQuery: {query}\n{'─'*60}")
    buckets = await filter_documents(query, sample_docs)

    for label, items in buckets.items():
        for item in items:
            print(f"\n[{label.upper()}] {item.doc.title}")
            print(f"  BM25 score : {item.bm25_score:.3f}")
            print(f"  Reason     : {item.reason}")
            print(f"  Latency    : {item.llm_latency_ms:.0f} ms")

    print(f"\n{'─'*60}")
    print("Docs passed to ranker:", [d.title for d in get_ranker_input(buckets)])


if __name__ == "__main__":
    asyncio.run(_demo())
