"""
feedback.py — Phase 7: improving the query with feedback.

A short query often misses relevant documents that use other words
("crpc" vs "code of criminal procedure"). Three kinds of feedback add the
missing words:

1. Legal thesaurus     abbreviations and their expansions, and a few common
                       synonyms. "anticipatory bail 438 crpc" also searches
                       for "code of criminal procedure".

2. Pseudo-relevance    assume the top PRF_DOCS results of the first search are
   feedback (PRF)      relevant, and add their most characteristic words.

3. Implicit feedback   learn from what users do in search_app.py: a user who
   from clicks         opens a result and READS it for a while probably found
                       it relevant; one who comes straight back did not.
                         dwell ≥ LONG_DWELL seconds  → relevant
                         dwell < SHORT_DWELL seconds → not relevant
                       Clicks are logged per query in data/clicks.jsonl.

All three feed the same formula from the course, ROCCHIO:

    new query = α · original query
              + β · average of the RELEVANT documents' vectors
              − γ · average of the NOT-RELEVANT documents' vectors

Documents are TF-IDF vectors (each term: (1 + log tf) · idf, scaled to
length 1). The EXPANSION_TERMS strongest new terms are added to the query,
and the weighted query is ranked with BM25.

Where the relevant / not-relevant documents come from:
    clicks for this query exist → long reads = relevant, quick returns = not relevant
    otherwise, with PRF on      → the top PRF_DOCS results = relevant

Run:
    python feedback.py "anticipatory bail 438 crpc" --thesaurus --prf
    python feedback.py runs          # write the Phase 7 evaluation runs
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import write_run
from index import load
from judge import load_queries
from ranking import RUN_DEPTH, RUNS_DIR, Ranker
from text_processing import analyze

BASE_DIR    = Path(__file__).resolve().parent
CORPUS_FILE = BASE_DIR / "data" / "corpus_dedup.jsonl"
CLICKS_FILE = BASE_DIR / "data" / "clicks.jsonl"

THESAURUS_WEIGHT = 0.5     # weight of a term added by the thesaurus (typed terms: 1.0)
ALPHA, BETA, GAMMA = 1.0, 0.5, 0.25
PRF_DOCS        = 10       # top results assumed relevant by pseudo-relevance feedback
EXPANSION_TERMS = 10       # new terms added by Rocchio
LONG_DWELL      = 30       # seconds: a read this long counts as relevant
SHORT_DWELL     = 10       # seconds: a return this quick counts as not relevant

# ── 1. The legal thesaurus ────────────────────────────────────────────────────

# Each group: phrases that mean the same thing. If the query contains one,
# the others are added. General legal abbreviations and well-known synonyms.
THESAURUS = [
    ["crpc", "code of criminal procedure", "criminal procedure code"],
    ["ipc", "indian penal code", "penal code"],
    ["cpc", "code of civil procedure", "civil procedure code"],
    ["rti", "right to information"],
    ["posh", "sexual harassment of women at workplace"],
    ["ndps", "narcotic drugs and psychotropic substances"],
    ["pocso", "protection of children from sexual offences"],
    ["uapa", "unlawful activities prevention"],
    ["ni act", "negotiable instruments act"],
    ["dv act", "domestic violence act"],
    ["death penalty", "death sentence", "capital punishment"],
    ["cheque bounce", "dishonour of cheque"],
    ["divorce", "dissolution of marriage"],
    ["live-in relationship", "relationship in the nature of marriage"],
    ["euthanasia", "right to die with dignity"],
    ["fir", "first information report"],
]

def contains(terms: list[str], phrase: list[str]) -> bool:
    """Does the term list contain the phrase as consecutive terms?"""
    n = len(phrase)
    return any(terms[i:i + n] == phrase for i in range(len(terms) - n + 1))

def thesaurus_terms(query: str, method: str) -> list[str]:
    """Terms of the synonyms of every thesaurus phrase found in the query."""
    query_terms = analyze(query, method)
    added = []
    for group in THESAURUS:
        phrases = [analyze(p, method) for p in group]
        if any(p and contains(query_terms, p) for p in phrases):
            for p in phrases:
                added += [t for t in p if t not in query_terms and t not in added]
    return added

# ── 2 + 3. Rocchio with pseudo-relevance or click feedback ────────────────────

class DocStore:
    """Read any document's text from the corpus without loading the whole file."""
    def __init__(self, path: Path = CORPUS_FILE):
        self.path, self.offset = path, {}
        with path.open("rb") as f:                          # one pass: where each line starts
            position = 0
            for line in f:
                self.offset[json.loads(line)["doc_id"]] = position
                position += len(line)

    def text(self, doc_id: str) -> dict:
        with self.path.open("rb") as f:
            f.seek(self.offset[doc_id])
            return json.loads(f.readline())

class FeedbackSearch:
    def __init__(self, ranker: Ranker):
        self.ranker = ranker
        self.index = ranker.index
        self.store = DocStore()
        self._vectors = {}                                   # cache: doc number → vector

    def doc_vector(self, doc: int) -> dict[str, float]:
        """TF-IDF vector of a document, scaled to length 1."""
        if doc not in self._vectors:
            text = self.store.text(self.index.doc_ids[doc])
            counts = Counter(analyze(text["title"] + " " + text["text"], self.index.method))
            vec = {t: (1 + math.log(c)) * math.log(self.ranker.N / self.index.df(t))
                   for t, c in counts.items() if t in self.index.postings}
            length = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            self._vectors[doc] = {t: w / length for t, w in vec.items()}
        return self._vectors[doc]

    def centroid(self, docs: list[int]) -> dict[str, float]:
        """Average vector of some documents."""
        total = defaultdict(float)
        for doc in docs:
            for t, w in self.doc_vector(doc).items():
                total[t] += w / len(docs)
        return total

    def rocchio(self, weights: dict[str, float], relevant: list[int],
                not_relevant: list[int]) -> dict[str, float]:
        """α·query + β·centroid(relevant) − γ·centroid(not relevant); keep the best new terms."""
        new = {t: ALPHA * w for t, w in weights.items()}
        good = self.centroid(relevant) if relevant else {}
        bad = self.centroid(not_relevant) if not_relevant else {}
        top = max(good.values(), default=1.0)                # scale so the strongest term gets β
        change = {t: BETA * good.get(t, 0) / top - GAMMA * bad.get(t, 0) / top
                  for t in set(good) | set(bad)}
        # Expansion: the strongest NEW terms (positive change, not already in the query)
        candidates = sorted((t for t in change if t not in new and change[t] > 0),
                            key=change.get, reverse=True)[:EXPANSION_TERMS]
        for t in candidates:
            new[t] = change[t]
        # Terms already in the query move up or down, but never below zero
        for t in weights:
            new[t] = max(0.0, new[t] + change.get(t, 0))
        return {t: w for t, w in new.items() if w > 0}

    def expand(self, query: str, thesaurus: bool = False, prf: bool = False,
               clicks: bool = False) -> tuple[dict[str, float], str]:
        """Weighted query terms after feedback, and a short note on what was used."""
        weights = self.ranker.query_weights(query)
        notes = []
        if thesaurus:
            added = [t for t in thesaurus_terms(query, self.index.method) if t in self.index.postings]
            for t in added:
                weights[t] = THESAURUS_WEIGHT
            if added:
                notes.append(f"thesaurus +{added}")

        relevant, not_relevant = [], []
        if clicks:
            relevant, not_relevant = click_feedback(query, self.index)
            if relevant or not_relevant:
                notes.append(f"clicks: {len(relevant)} long reads, {len(not_relevant)} quick returns")
        if not relevant and prf:
            relevant = [doc for doc, _ in self.ranker.rank(weights, "bm25", PRF_DOCS)]
            notes.append(f"pseudo-relevance: top {len(relevant)}")
        if relevant or not_relevant:
            weights = self.rocchio(weights, relevant, not_relevant)
        return weights, "; ".join(notes) or "no feedback"

    def search(self, query: str, k: int = 10, **feedback) -> tuple[list[tuple[int, float]], str]:
        weights, note = self.expand(query, **feedback)
        return self.ranker.rank(weights, "bm25", k), note

# ── The click log (implicit feedback) ─────────────────────────────────────────

def query_key(query: str) -> str:
    """Queries that analyze to the same terms share their clicks ("Bail" = "bail")."""
    return " ".join(sorted(set(analyze(query))))

def log_click(query: str, doc_id: str, rank: int, dwell_seconds: float) -> None:
    """Append one click to data/clicks.jsonl (called by search_app.py)."""
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "query": query, "key": query_key(query),
              "doc_id": doc_id, "rank": rank, "dwell": round(dwell_seconds, 1)}
    with CLICKS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def click_feedback(query: str, index) -> tuple[list[int], list[int]]:
    """Documents read long (relevant) and left quickly (not relevant) for this query."""
    if not CLICKS_FILE.exists():
        return [], []
    key = query_key(query)
    longest = {}                                            # doc_id → longest dwell seen
    with CLICKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            click = json.loads(line)
            if click["key"] == key:
                longest[click["doc_id"]] = max(longest.get(click["doc_id"], 0), click["dwell"])
    number = {doc_id: n for n, doc_id in enumerate(index.doc_ids) if doc_id}
    relevant = [number[d] for d, s in longest.items() if s >= LONG_DWELL and d in number]
    not_relevant = [number[d] for d, s in longest.items() if s < SHORT_DWELL and d in number]
    return relevant, not_relevant

# ── Evaluation runs ───────────────────────────────────────────────────────────

def write_runs() -> None:
    fs = FeedbackSearch(Ranker(load("porter")))
    queries = load_queries()
    for name, options in [("bm25_thes", {"thesaurus": True}),
                          ("bm25_prf", {"prf": True}),
                          ("bm25_thes_prf", {"thesaurus": True, "prf": True})]:
        results = {}
        for qid, q in queries.items():
            ranked, _ = fs.search(q["query"], RUN_DEPTH, **options)
            results[qid] = [(fs.index.doc_ids[doc], score) for doc, score in ranked]
        write_run(results, name, RUNS_DIR / f"{name}.txt")
        print(f"[feedback] wrote {name}.txt")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search with query expansion / feedback.")
    parser.add_argument("query", nargs="+", help='a query, or "runs"')
    parser.add_argument("--thesaurus", action="store_true")
    parser.add_argument("--prf", action="store_true", help="pseudo-relevance feedback")
    parser.add_argument("--clicks", action="store_true", help="use logged clicks for this query")
    args = parser.parse_args()

    if args.query == ["runs"]:
        write_runs()
    else:
        fs = FeedbackSearch(Ranker(load("porter")))
        query = " ".join(args.query)
        weights, note = fs.expand(query, args.thesaurus, args.prf, args.clicks)
        print(f"feedback: {note}")
        print("query   : " + ", ".join(f"{t} {w:.2f}" for t, w in
                                       sorted(weights.items(), key=lambda x: -x[1])))
        for rank, (doc, score) in enumerate(fs.ranker.rank(weights, "bm25"), 1):
            print(f"{rank:2}. {score:8.3f}  {fs.index.titles[doc][:85]}")
