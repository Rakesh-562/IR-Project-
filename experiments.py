"""
experiments.py — Phases 5–7 evaluated with the human judgments.

1. Main table       every run in eval/runs/, scored two ways:
                      lenient: grade ≥ 1 counts as relevant
                      strict:  only grade 2 counts as relevant
                    (nDCG@10 always uses the full grades)

2. Tuning with      each model has settings (BM25 k1, b; LM μ; PageRank w;
   2-fold cross-    PRF documents and β). Choosing them on the same queries we
   validation       report on would flatter them. So the 30 queries are split
                    in two folds (odd / even qid): pick the best setting on one
                    fold, score it on the other, then swap. The reported number
                    only ever comes from queries the setting was NOT chosen on.

3. Simulated        there are no real users yet, so a simulated user looks at
   click feedback   BM25's top CLICK_DEPTH results: a highly relevant result
                    (grade 2) is read for a long time; any other is left at
                    once. Those "clicks" go through the same Rocchio code as
                    real clicks (feedback.py). To be fair, the documents the
                    user already saw are removed from both rankings before
                    scoring ("residual collection"): feedback must find NEW
                    relevant documents, not just re-rank the ones clicked.

4. Per query        where PageRank and PRF help or hurt.

Metric for tuning and comparisons: nDCG@10 (it uses the 0/1/2 grades;
P@10 is almost always 1.0 here because nearly every pooled document was
judged at least partly relevant).

Run:
    python experiments.py          # ~5 minutes → analysis/results.md
"""

import random
import sys
import time
from itertools import product
from pathlib import Path

import feedback
from spelling import SpellCorrector
from evaluate import METRICS, evaluate_run, load_qrels, load_run, mean, ndcg_at, randomization_test
from feedback import FeedbackSearch
from index import load
from judge import load_queries
from ranking import Ranker

BASE_DIR    = Path(__file__).resolve().parent
RUNS_DIR    = BASE_DIR / "eval" / "runs"
REPORT_FILE = BASE_DIR / "analysis" / "results.md"
RUN_ORDER   = ["ik_native", "tfidf", "bm25", "lm_dir", "bm25_none", "bm25_lemma",
               "bm25_pr", "bm25_thes", "bm25_prf", "bm25_thes_prf"]
DEPTH       = 100           # documents ranked per query in the experiments
CLICK_DEPTH = 5             # results the simulated user looks at

GRIDS = {
    "bm25":   [{"k1": k1, "b": b} for k1, b in product([0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9])],
    "lm_dir": [{"mu": mu} for mu in [100, 300, 1000, 2000, 4000]],
    "bm25_pr": [{"w": w} for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]],
    "bm25_prf": [{"docs": n, "beta": beta} for n, beta in product([3, 5, 10], [0.25, 0.5, 1.0])],
}
DEFAULTS = {"bm25": {"k1": 1.2, "b": 0.75}, "lm_dir": {"mu": 2000},
            "bm25_pr": {"w": 1.0}, "bm25_prf": {"docs": 10, "beta": 0.5}}

def ndcg_per_query(run: dict[str, list[str]], qrels) -> dict[str, float]:
    return evaluate_run(run, qrels)["nDCG@10"]

def fmt(setting: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in setting.items())

def add_typo(query: str, rng: random.Random) -> str:
    """One realistic typo in the longest word: drop, swap or replace a letter (not the first)."""
    words = query.split()
    i = max(range(len(words)), key=lambda j: len(words[j]) if words[j].isalpha() else 0)
    w = words[i]
    pos = rng.randrange(1, len(w) - 1)
    kind = rng.choice(["drop", "swap", "replace"])
    if kind == "drop":
        w = w[:pos] + w[pos + 1:]
    elif kind == "swap":
        w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
    else:
        w = w[:pos] + rng.choice([c for c in "aeiou" if c != w[pos]]) + w[pos + 1:]
    words[i] = w
    return " ".join(words)

# ── Running one setting over all queries ──────────────────────────────────────

class Experiment:
    def __init__(self):
        self.queries = load_queries()
        self.qrels = load_qrels()
        self.fs = FeedbackSearch(Ranker(load("porter")))
        self.ranker = self.fs.ranker
        self._bm25_cache = {}

    def to_run(self, ranked_by_query: dict[str, list[tuple[int, float]]]) -> dict[str, list[str]]:
        ids = self.ranker.index.doc_ids
        return {qid: [ids[d] for d, _ in ranked] for qid, ranked in ranked_by_query.items()}

    @staticmethod
    def top(scores: dict[int, float], k: int = DEPTH) -> list[tuple[int, float]]:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]

    def run(self, model: str, s: dict) -> dict[str, list[str]]:
        """The ranking of every query for one model and setting."""
        out = {}
        for qid, q in self.queries.items():
            weights = self.ranker.query_weights(q["query"])
            if model == "bm25":
                scores = self.ranker.bm25(weights, s["k1"], s["b"])
            elif model == "lm_dir":
                scores = self.ranker.lm_dirichlet(weights, s["mu"])
            elif model == "bm25_pr":
                if qid not in self._bm25_cache:
                    self._bm25_cache[qid] = self.ranker.bm25(weights)
                prior = self.ranker.log_pagerank()
                scores = {d: v + s["w"] * prior[d] for d, v in self._bm25_cache[qid].items()}
            elif model == "bm25_prf":
                feedback.PRF_DOCS, feedback.BETA = s["docs"], s["beta"]
                expanded, _ = self.fs.expand(q["query"], prf=True)
                scores = self.ranker.bm25(expanded)
            out[qid] = self.top(scores)
        return self.to_run(out)

    # ── 2-fold cross-validation ───────────────────────────────────────────────

    def cross_validate(self, model: str) -> dict:
        per_setting = []                                   # (setting, {qid: nDCG})
        for s in GRIDS[model]:
            per_setting.append((s, ndcg_per_query(self.run(model, s), self.qrels)))
        qids = sorted(per_setting[0][1])
        folds = [qids[0::2], qids[1::2]]                  # q01, q03, … / q02, q04, …
        tested, chosen = {}, []
        for train, test in (folds, folds[::-1]):
            best_s, best_scores = max(per_setting, key=lambda p: mean(p[1][q] for q in train))
            chosen.append(best_s)
            tested.update({q: best_scores[q] for q in test})   # scored only on unseen queries
        default_scores = next(sc for s, sc in per_setting if s == DEFAULTS[model])
        best_all = max(per_setting, key=lambda p: mean(p[1].values()))
        return {"model": model, "chosen": chosen, "cv": tested, "default": default_scores,
                "best_all": best_all, "grid": per_setting}

    # ── Simulated click feedback ──────────────────────────────────────────────

    def simulate_clicks(self) -> dict:
        base_run, click_run = {}, {}
        for qid, q in self.queries.items():
            weights = self.ranker.query_weights(q["query"])
            baseline = self.top(self.ranker.bm25(weights))
            seen = [d for d, _ in baseline[:CLICK_DEPTH]]
            grades = self.qrels.get(qid, {})
            ids = self.ranker.index.doc_ids
            long_reads  = [d for d in seen if grades.get(ids[d], 0) == 2]
            quick_backs = [d for d in seen if grades.get(ids[d], 0) < 2]
            new_weights = self.fs.rocchio(weights, long_reads, quick_backs)
            after = self.top(self.ranker.bm25(new_weights))
            # residual collection: drop what the user already saw, from both rankings
            base_run[qid]  = [ids[d] for d, _ in baseline if d not in seen]
            click_run[qid] = [ids[d] for d, _ in after if d not in seen]
        base, clicks = ndcg_per_query(base_run, self.qrels), ndcg_per_query(click_run, self.qrels)
        judged = lambda run: mean(sum((q, d) in self.judged_pairs for d in docs[:10]) / 10
                                  for q, docs in run.items())
        # "Condensed" scoring: drop unjudged documents from both rankings, so neither
        # side is punished just for finding documents nobody graded.
        condense = lambda run: {q: [d for d in docs if (q, d) in self.judged_pairs] for q, docs in run.items()}
        base_c = ndcg_per_query(condense(base_run), self.qrels)
        clicks_c = ndcg_per_query(condense(click_run), self.qrels)
        return {"base": base, "clicks": clicks, "p": randomization_test(clicks, base),
                "judged_base": judged(base_run), "judged_clicks": judged(click_run),
                "base_c": base_c, "clicks_c": clicks_c, "p_c": randomization_test(clicks_c, base_c)}

    # ── Spelling correction ───────────────────────────────────────────────────

    def spelling_test(self) -> dict:
        """Put one typo in each query's longest word, then search with and without correction."""
        speller = SpellCorrector()
        versions = {"original": {}, "typo": {}, "corrected": {}}
        examples, recovered, untouched = [], 0, 0
        for qid, q in self.queries.items():
            original = q["query"].lower()
            typo = add_typo(original, random.Random(qid))
            corrected = speller.correct_query(typo)
            recovered += corrected == " ".join(original.split())
            untouched += speller.correct_query(original) == " ".join(original.split())
            examples.append((qid, typo, corrected))
            for name, text in (("original", original), ("typo", typo), ("corrected", corrected)):
                versions[name][qid] = self.top(self.ranker.bm25(self.ranker.query_weights(text)))
        scores = {name: ndcg_per_query(self.to_run(v), self.qrels) for name, v in versions.items()}
        return {"scores": scores, "recovered": recovered, "untouched": untouched,
                "examples": examples, "p": randomization_test(scores["corrected"], scores["typo"])}

    @property
    def judged_pairs(self) -> set:
        return {(q, d) for q, grades in self.qrels.items() for d in grades}

# ── Report ────────────────────────────────────────────────────────────────────

def main_table(qrels, min_grade: int) -> list[str]:
    runs = {n: load_run(RUNS_DIR / f"{n}.txt") for n in RUN_ORDER if (RUNS_DIR / f"{n}.txt").exists()}
    results = {n: evaluate_run(r, qrels, min_grade) for n, r in runs.items()}
    lines = ["| Run | " + " | ".join(METRICS) + " | p (nDCG@10 vs bm25) |",
             "|---" * (len(METRICS) + 2) + "|"]
    for n, sc in results.items():
        p = "—" if n == "bm25" else f"{randomization_test(sc['nDCG@10'], results['bm25']['nDCG@10']):.3f}"
        lines.append(f"| {n} | " + " | ".join(f"{mean(sc[m].values()):.3f}" for m in METRICS) + f" | {p} |")
    return lines

def per_query_changes(qrels, run_name: str, queries) -> list[str]:
    base = evaluate_run(load_run(RUNS_DIR / "bm25.txt"), qrels)["nDCG@10"]
    other = evaluate_run(load_run(RUNS_DIR / f"{run_name}.txt"), qrels)["nDCG@10"]
    diffs = sorted(((other[q] - base[q], q) for q in base), reverse=True)
    helped = sum(1 for d, _ in diffs if d > 0.005); hurt = sum(1 for d, _ in diffs if d < -0.005)
    lines = [f"`{run_name}` vs `bm25`: better on {helped}, worse on {hurt}, "
             f"unchanged on {len(diffs) - helped - hurt} queries.", "",
             "| Query | bm25 | " + run_name + " | change |", "|---|---|---|---|"]
    for d, q in diffs[:3] + diffs[-3:]:
        lines.append(f"| {q} {queries[q]['query'][:55]} | {base[q]:.3f} | {other[q]:.3f} | {d:+.3f} |")
    return lines

def main() -> None:
    start = time.time()
    ex = Experiment()
    print("[experiments] cross-validating ...")
    cv = {m: ex.cross_validate(m) for m in GRIDS}
    print("[experiments] simulating clicks ...")
    clicks = ex.simulate_clicks()
    print("[experiments] testing spelling correction ...")
    spell = ex.spelling_test()

    lines = ["# Evaluation results (Phases 5–7)", "",
             "Generated by `experiments.py` from the human judgments in `eval/qrels.txt` "
             f"({sum(len(g) for g in ex.qrels.values())} graded pairs, 30 queries). "
             "p-values: paired randomization test on nDCG@10 against `bm25`.", "",
             "## 1. All runs — lenient (grade ≥ 1 relevant)", "", *main_table(ex.qrels, 1), "",
             "## 2. All runs — strict (only grade 2 relevant)", "", *main_table(ex.qrels, 2), "",
             "## 3. Tuning with 2-fold cross-validation (nDCG@10)", "",
             "| Model | Default setting | Default nDCG@10 | Chosen (fold 1 / fold 2) | Cross-validated nDCG@10 | p vs default | Best on all queries (optimistic) |",
             "|---|---|---|---|---|---|---|"]
    for m, r in cv.items():
        lines.append(f"| {m} | {fmt(DEFAULTS[m])} | {mean(r['default'].values()):.3f} | "
                     f"{fmt(r['chosen'][0])} / {fmt(r['chosen'][1])} | {mean(r['cv'].values()):.3f} | "
                     f"{randomization_test(r['cv'], r['default']):.3f} | "
                     f"{fmt(r['best_all'][0])}: {mean(r['best_all'][1].values()):.3f} |")
    lines += ["", "PageRank weight w (BM25 + w · max(0, log(PR·N)), judgments only), nDCG@10 on all 30 queries:", "",
              "| w | " + " | ".join(str(s["w"]) for s, _ in cv["bm25_pr"]["grid"]) + " |",
              "|---" * (len(cv["bm25_pr"]["grid"]) + 1) + "|",
              "| nDCG@10 | " + " | ".join(f"{mean(sc.values()):.3f}" for _, sc in cv["bm25_pr"]["grid"]) + " |",
              "", "## 4. Simulated click feedback (residual collection)", "",
              f"The simulated user reads BM25's top {CLICK_DEPTH}: grade 2 → long read (relevant), "
              "otherwise → quick return (not relevant). Rocchio then re-ranks; the 5 seen documents "
              "are removed from both rankings before scoring.", "",
              "| | nDCG@10 (unjudged = not relevant) | judged in top 10 | nDCG@10 condensed (unjudged removed) |",
              "|---|---|---|---|",
              f"| BM25, no feedback | {mean(clicks['base'].values()):.3f} | {clicks['judged_base']:.0%} "
              f"| {mean(clicks['base_c'].values()):.3f} |",
              f"| BM25 + click feedback | {mean(clicks['clicks'].values()):.3f} | {clicks['judged_clicks']:.0%} "
              f"| {mean(clicks['clicks_c'].values()):.3f} |",
              "", f"p = {clicks['p']:.3f} (unjudged = not relevant); p = {clicks['p_c']:.3f} (condensed). "
              "Feedback surfaces more documents outside the judged pool, so the first column "
              "is biased against it; the condensed column removes that bias.", "",
              "## 5. Where PageRank and PRF help or hurt (nDCG@10 per query)", "",
              *per_query_changes(ex.qrels, "bm25_pr", ex.queries), "",
              *per_query_changes(ex.qrels, "bm25_prf", ex.queries), "",
              "## 6. Spelling correction on misspelled queries", "",
              "Each query gets one typo in its longest word (a letter dropped, swapped or "
              "replaced; fixed random seed). BM25, nDCG@10:", "",
              "| Query version | nDCG@10 |", "|---|---|",
              *[f"| {name} | {mean(sc.values()):.3f} |" for name, sc in spell["scores"].items()],
              "", f"Typos corrected back to the original query: {spell['recovered']}/30. "
              f"Correct queries left unchanged by the corrector: {spell['untouched']}/30. "
              f"p (corrected vs typo) = {spell['p']:.3f}.", "",
              "| qid | query with typo | after correction |", "|---|---|---|",
              *[f"| {q} | {t} | {c} |" for q, t, c in spell["examples"][:8]]]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))
    print(f"[experiments] done in {time.time() - start:.0f}s → {REPORT_FILE}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
