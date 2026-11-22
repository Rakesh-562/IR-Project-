"""
evaluate.py — Phase 4: score ranked runs against relevance judgments.

Files (standard TREC formats):
    eval/qrels.txt          qid 0 doc_id grade        grade 0 = not, 1 = partly, 2 = highly relevant
    eval/runs/<name>.txt    qid Q0 doc_id rank score name

Metrics (per query, then averaged over queries):
    P@k      fraction of the top k that is relevant (grade ≥ 1)
    R@k      fraction of all relevant documents found in the top k
    AP       average of P@i over the ranks i where a relevant document appears,
             divided by the number of relevant documents   →  MAP = mean of AP
    RR       1 / rank of the first relevant document        →  MRR = mean of RR
    nDCG@k   graded: DCG = Σ (2^grade − 1) / log2(rank + 1), divided by the
             DCG of the ideal ordering (best grades first)

Documents nobody judged count as NOT relevant (the usual pooling assumption).
Queries without any relevant document are skipped.

Significance: a paired randomization test on per-query AP against a baseline
run. If the two runs were really equal, swapping their scores on any query
would make no difference; we swap at random 10,000 times and count how often
the difference is at least as large as the one observed (= p-value).

Run:
    python evaluate.py eval/runs/*.txt
    python evaluate.py eval/runs/*.txt --baseline ik_native --per-query
"""

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
QRELS_FILE = BASE_DIR / "eval" / "qrels.txt"

# ── Reading files ─────────────────────────────────────────────────────────────

def load_qrels(path: Path = QRELS_FILE) -> dict[str, dict[str, int]]:
    """{qid: {doc_id: grade}}"""
    qrels = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, _, doc_id, grade = line.split()
            qrels[qid][doc_id] = int(grade)
    return dict(qrels)

def load_run(path: Path) -> dict[str, list[str]]:
    """{qid: [doc_id, …] in rank order}"""
    rows = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, _, doc_id, rank, _score, _name = line.split()
            rows[qid].append((int(rank), doc_id))
    return {qid: [doc for _, doc in sorted(r)] for qid, r in rows.items()}

def write_run(results: dict[str, list[tuple[str, float]]], name: str, path: Path) -> None:
    """Save {qid: [(doc_id, score), …] best first} in TREC format (used by the ranking code)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, ranked in results.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {name}\n")

# ── Metrics for one query ─────────────────────────────────────────────────────
# ranked = the run's documents, best first; grades = {doc_id: grade} from the qrels

def precision_at(ranked, grades, k):
    return sum(1 for d in ranked[:k] if grades.get(d, 0) >= 1) / k

def recall_at(ranked, grades, k):
    relevant = sum(1 for g in grades.values() if g >= 1)
    return sum(1 for d in ranked[:k] if grades.get(d, 0) >= 1) / relevant

def average_precision(ranked, grades):
    relevant = sum(1 for g in grades.values() if g >= 1)
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if grades.get(d, 0) >= 1:
            hits += 1
            total += hits / i                     # precision at this rank
    return total / relevant

def reciprocal_rank(ranked, grades):
    for i, d in enumerate(ranked, start=1):
        if grades.get(d, 0) >= 1:
            return 1 / i
    return 0.0

def ndcg_at(ranked, grades, k):
    def dcg(grade_list):
        return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grade_list, start=1))
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return dcg([grades.get(d, 0) for d in ranked[:k]]) / ideal

METRICS = {
    "P@5":     lambda r, g: precision_at(r, g, 5),
    "P@10":    lambda r, g: precision_at(r, g, 10),
    "R@20":    lambda r, g: recall_at(r, g, 20),
    "MAP":     average_precision,
    "MRR":     reciprocal_rank,
    "nDCG@10": lambda r, g: ndcg_at(r, g, 10),
}

def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """{metric: {qid: value}} over the queries that have at least one relevant document."""
    scores = {m: {} for m in METRICS}
    for qid, grades in qrels.items():
        if not any(g >= 1 for g in grades.values()):
            continue
        ranked = run.get(qid, [])                 # a query the run skipped scores 0
        for name, metric in METRICS.items():
            scores[name][qid] = metric(ranked, grades)
    return scores

def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

# ── Significance ──────────────────────────────────────────────────────────────

def randomization_test(a: dict[str, float], b: dict[str, float], trials: int = 10_000) -> float:
    """Two-sided p-value for 'runs a and b have the same mean' (paired by query)."""
    diffs = [a[q] - b[q] for q in a]
    observed = abs(mean(diffs))
    rng = random.Random(0)
    at_least_as_big = 0
    for _ in range(trials):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]   # swap a and b on random queries
        if abs(mean(flipped)) >= observed - 1e-12:
            at_least_as_big += 1
    return at_least_as_big / trials

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate runs against eval/qrels.txt.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--qrels", type=Path, default=QRELS_FILE)
    parser.add_argument("--baseline", help="run name to test the others against (MAP)")
    parser.add_argument("--per-query", action="store_true", help="also print AP per query")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    results = {p.stem: evaluate_run(load_run(p), qrels) for p in args.runs}
    n_queries = len(next(iter(results.values()))["MAP"])
    if n_queries == 0:
        sys.exit(f"No query has a relevant document in {args.qrels}. Grade eval/judgments.tsv, "
                 "then run: python judge.py qrels")
    print(f"{n_queries} queries with relevant documents in {args.qrels.name}\n")

    print("| Run | " + " | ".join(METRICS) + (" | p (MAP vs baseline) |" if args.baseline else " |"))
    print("|---" * (len(METRICS) + 1 + bool(args.baseline)) + "|")
    for name, scores in results.items():
        row = f"| {name} | " + " | ".join(f"{mean(scores[m].values()):.3f}" for m in METRICS)
        if args.baseline:
            if name == args.baseline:
                row += " | (baseline)"
            else:
                row += f" | {randomization_test(scores['MAP'], results[args.baseline]['MAP']):.3f}"
        print(row + " |")

    if args.per_query:
        names = list(results)
        print("\nAP per query:\n\n| qid | " + " | ".join(names) + " |")
        print("|---" * (len(names) + 1) + "|")
        for qid in sorted(results[names[0]]["MAP"]):
            print(f"| {qid} | " + " | ".join(f"{results[n]['MAP'][qid]:.3f}" for n in names) + " |")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
