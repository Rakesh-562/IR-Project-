"""
ranking.py — Phase 5: three ranking models over our inverted index.

All three score a document by adding up a contribution for each query term,
term at a time: walk down the term's posting list and add to an
"accumulator" (a running score) for each document in it.

1. TF-IDF cosine (vector space model, SMART "lnc.ltc")
       document weight  w(t,d) = (1 + log tf) / |d|        |d| = length of d's weight vector
       query weight     w(t,q) = (1 + log qtf) · idf / |q|  idf = log(N / df)
       score = Σ w(t,q) · w(t,d)  = the cosine of the angle between the two vectors

2. Okapi BM25
       score = Σ idf(t) · tf·(k1 + 1) / (tf + k1·(1 − b + b·dl/avgdl))
       idf(t) = log(1 + (N − df + 0.5) / (df + 0.5))
       k1 limits how much repeating a term helps (saturation);
       b  controls how much long documents are penalised (length normalisation).

3. Query likelihood with Dirichlet smoothing (language model)
       score = Σ log( (tf + μ·P(t|C)) / (dl + μ) )
       P(t|C) = how common t is in the whole collection. A document is a "bag
       of words" model; we ask how likely it is to generate the query.
       Smoothing gives unseen terms a small probability instead of zero.

Each query term counts once (repeats in the query are ignored), and the
query goes through the same analyze() as the documents.

Run:
    python ranking.py "anticipatory bail conditions" --model bm25
    python ranking.py runs                   # write eval/runs/*.txt for the 30 queries
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

from evaluate import write_run
from index import InvertedIndex, index_file, load
from judge import load_queries
from text_processing import analyze

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "eval" / "runs"

K1, B = 1.2, 0.75          # common defaults; tuned once judgments exist
MU = 2000                  # common default for Dirichlet smoothing
RUN_DEPTH = 100            # documents per query written to a run

class Ranker:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.N = index.num_docs
        self.avgdl = index.avg_doc_len
        self.total_terms = sum(index.doc_len)          # |C|, for the language model
        self._doc_norms = None                         # computed on first TF-IDF query

    def query_terms(self, query: str) -> list[str]:
        """Analyzed query terms that occur in the index, each once."""
        terms = dict.fromkeys(analyze(query, self.index.method))   # unique, keeps order
        return [t for t in terms if t in self.index.postings]

    # ── 1. TF-IDF cosine ──────────────────────────────────────────────────────

    def doc_norms(self) -> list[float]:
        """|d| for every document: sqrt(Σ (1 + log tf)²) over all its terms (one pass over the index)."""
        if self._doc_norms is None:
            squares = [0.0] * (self.N + 1)
            for plist in self.index.postings.values():
                for k, doc in enumerate(plist.docs):
                    squares[doc] += (1 + math.log(plist.tf(k))) ** 2
            self._doc_norms = [math.sqrt(s) for s in squares]
        return self._doc_norms

    def tfidf(self, terms: list[str]) -> dict[int, float]:
        norms = self.doc_norms()
        query_weights = {t: math.log(self.N / self.index.df(t)) for t in terms}   # qtf = 1
        query_norm = math.sqrt(sum(w * w for w in query_weights.values())) or 1.0
        scores = defaultdict(float)
        for t in terms:
            plist = self.index.postings[t]
            wq = query_weights[t] / query_norm
            for k, doc in enumerate(plist.docs):
                scores[doc] += wq * (1 + math.log(plist.tf(k))) / norms[doc]
        return scores

    # ── 2. BM25 ───────────────────────────────────────────────────────────────

    def bm25_idf(self, df: int) -> float:
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def bm25(self, terms: list[str], k1: float = K1, b: float = B) -> dict[int, float]:
        scores = defaultdict(float)
        for t in terms:
            plist = self.index.postings[t]
            idf = self.bm25_idf(len(plist))
            for k, doc in enumerate(plist.docs):
                tf = plist.tf(k)
                length_norm = 1 - b + b * self.index.doc_len[doc] / self.avgdl
                scores[doc] += idf * tf * (k1 + 1) / (tf + k1 * length_norm)
        return scores

    # ── 3. Query likelihood, Dirichlet smoothing ──────────────────────────────

    def lm_dirichlet(self, terms: list[str], mu: float = MU) -> dict[int, float]:
        # tf of each query term in each document that contains it
        tfs = {}
        for t in terms:
            plist = self.index.postings[t]
            tfs[t] = {doc: plist.tf(k) for k, doc in enumerate(plist.docs)}
        candidates = set().union(*tfs.values()) if tfs else set()

        scores = {}
        for doc in candidates:                        # documents with at least one query term
            dl = self.index.doc_len[doc]
            score = 0.0
            for t in terms:
                p_collection = len(self.index.postings[t].positions) / self.total_terms
                tf = tfs[t].get(doc, 0)               # 0 → smoothing still gives a small probability
                score += math.log((tf + mu * p_collection) / (dl + mu))
            scores[doc] = score
        return scores

    # ── Searching ─────────────────────────────────────────────────────────────

    def search(self, query: str, model: str = "bm25", k: int = 10) -> list[tuple[int, float]]:
        """Top k (document number, score), best first; ties broken by document number."""
        terms = self.query_terms(query)
        if not terms:
            return []
        scores = {"tfidf": self.tfidf, "bm25": self.bm25, "lm_dir": self.lm_dirichlet}[model](terms)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]

# ── Writing runs for evaluation ───────────────────────────────────────────────

def write_runs() -> None:
    queries = load_queries()
    jobs = [("porter", model, model) for model in ("tfidf", "bm25", "lm_dir")]
    # Normalization comparison (Phase 5): BM25 on the other indexes, if built
    jobs += [(method, "bm25", f"bm25_{method}") for method in ("none", "lemma")
             if index_file(method).exists()]

    rankers = {}
    for method, model, run_name in jobs:
        if method not in rankers:
            rankers[method] = Ranker(load(method))
        ranker = rankers[method]
        results = {
            qid: [(ranker.index.doc_ids[doc], score)
                  for doc, score in ranker.search(q["query"], model, RUN_DEPTH)]
            for qid, q in queries.items()
        }
        write_run(results, run_name, RUNS_DIR / f"{run_name}.txt")
        print(f"[ranking] wrote {run_name}.txt  ({method} index, {model})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Rank the corpus for a query, or write evaluation runs.")
    parser.add_argument("query", nargs="+", help='a query, or "runs" to write all evaluation runs')
    parser.add_argument("--model", default="bm25", choices=["tfidf", "bm25", "lm_dir"])
    parser.add_argument("--method", default="porter", help="which index (normalization) to use")
    args = parser.parse_args()

    if args.query == ["runs"]:
        write_runs()
    else:
        r = Ranker(load(args.method))
        for rank, (doc, score) in enumerate(r.search(" ".join(args.query), args.model), 1):
            print(f"{rank:2}. {score:8.3f}  {r.index.titles[doc][:85]}")
