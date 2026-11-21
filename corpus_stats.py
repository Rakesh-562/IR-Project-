"""
corpus_stats.py — Phase 2: Zipf's law and Heaps' law on our legal corpus.

Zipf's law   The k-th most frequent word occurs about C / k^s times, with s ≈ 1.
             On a log-log plot of frequency against rank this is a straight
             line with slope -s.

Heaps' law   After reading T tokens, the vocabulary has V ≈ K · T^β distinct
             words, with β ≈ 0.4–0.6: the vocabulary keeps growing, but more
             and more slowly. On a log-log plot this is a line with slope β.

Both are fitted with a straight line through the log-log points
(numpy.polyfit). Stemming and lemmatization merge word forms, so they shrink
the vocabulary; we measure by how much for each method.

Input:   data/corpus_dedup.jsonl
Output:  analysis/zipf.png, analysis/heaps.png, analysis/corpus_stats.md
         data/vocabulary.tsv   (word <TAB> frequency, used by spelling.py)

Run:
    python corpus_stats.py
    python corpus_stats.py --max-docs 500     # quick test
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                         # write image files, no window
import matplotlib.pyplot as plt
import numpy as np

from text_processing import STOPWORDS, normalize, tokenize

BASE_DIR     = Path(__file__).resolve().parent
CORPUS_FILE  = BASE_DIR / "data" / "corpus_dedup.jsonl"
VOCAB_FILE   = BASE_DIR / "data" / "vocabulary.tsv"
ANALYSIS_DIR = BASE_DIR / "analysis"

METHODS = ["none", "porter", "lemma"]         # snowball ≈ porter, so left out here
ZIPF_RANKS = (10, 10_000)                     # fit Zipf on the middle ranks only

# ── Counting ──────────────────────────────────────────────────────────────────

def count_terms(max_docs: int | None):
    """
    Read the corpus once. For each method, count every term and record the
    vocabulary size after each document (the Heaps curve).
    Stopwords are kept: both laws are about ALL the words of a text.
    """
    counts = {m: Counter() for m in METHODS}
    heaps  = {m: [] for m in METHODS}          # list of (tokens seen, vocabulary size)
    tokens_seen = 0
    with CORPUS_FILE.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            tokens = tokenize(json.loads(line)["text"])
            tokens_seen += len(tokens)
            for m in METHODS:
                counts[m].update(normalize(tokens, m))
                heaps[m].append((tokens_seen, len(counts[m])))
            if n % 500 == 0:
                print(f"  {n} documents, {tokens_seen:,} tokens")
            if max_docs and n >= max_docs:
                break
    return counts, heaps

# ── Fitting the two laws ──────────────────────────────────────────────────────

def fit_zipf(counter: Counter) -> tuple[float, float]:
    """Fit log(freq) = log(C) - s·log(rank) on ranks ZIPF_RANKS. Returns (s, C)."""
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    lo, hi = ZIPF_RANKS
    slope, intercept = np.polyfit(np.log10(ranks[lo - 1:hi]), np.log10(freqs[lo - 1:hi]), 1)
    return -slope, 10 ** intercept

def fit_heaps(points: list[tuple[int, int]]) -> tuple[float, float]:
    """Fit log(V) = log(K) + β·log(T). Returns (K, β)."""
    T, V = np.array(points, dtype=float).T
    beta, intercept = np.polyfit(np.log10(T), np.log10(V), 1)
    return 10 ** intercept, beta

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_zipf(counts: dict, fits: dict, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for m in METHODS:
        freqs = sorted(counts[m].values(), reverse=True)
        plt.loglog(range(1, len(freqs) + 1), freqs, label=f"{m} (s = {fits[m][0]:.2f})")
    s, C = fits["none"]
    ranks = np.array([1, len(counts["none"])])
    plt.loglog(ranks, C / ranks ** s, "k--", linewidth=1, label="Zipf fit (none)")
    plt.xlabel("rank"); plt.ylabel("frequency"); plt.title("Zipf's law — legal corpus")
    plt.legend(); plt.grid(True, which="major", alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()

def plot_heaps(heaps: dict, fits: dict, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for m in METHODS:
        T, V = zip(*heaps[m])
        K, beta = fits[m]
        plt.plot(T, V, label=f"{m} (K = {K:.1f}, β = {beta:.3f})")
    plt.xlabel("tokens seen (T)"); plt.ylabel("vocabulary size (V)")
    plt.title("Heaps' law — legal corpus"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

# ── Report ────────────────────────────────────────────────────────────────────

def write_report(counts, zipf, heaps_fit, n_docs, path: Path) -> None:
    none = counts["none"]
    total = sum(none.values())
    hapax = sum(1 for c in none.values() if c == 1)
    stop_share = sum(c for w, c in none.items() if w in STOPWORDS) / total
    top = ", ".join(f"{w} ({c:,})" for w, c in none.most_common(20))

    lines = [
        "# Corpus statistics (Phase 2)",
        "",
        f"Corpus: `data/corpus_dedup.jsonl`, {n_docs:,} documents, {total:,} tokens.",
        "Generated by `corpus_stats.py`.",
        "",
        "| Normalization | Vocabulary size | vs none | Zipf s | Heaps K | Heaps β |",
        "|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        V = len(counts[m])
        lines.append(f"| {m} | {V:,} | {V / len(none):.0%} | {zipf[m][0]:.2f} "
                     f"| {heaps_fit[m][0]:.1f} | {heaps_fit[m][1]:.3f} |")
    lines += [
        "",
        f"- Words that occur only once (hapax legomena): {hapax:,} "
        f"({hapax / len(none):.0%} of the vocabulary).",
        f"- Stopwords make up {stop_share:.0%} of all tokens.",
        f"- Zipf exponent fitted on ranks {ZIPF_RANKS[0]:,}–{ZIPF_RANKS[1]:,}.",
        "",
        f"Top 20 words: {top}",
        "",
        "![Zipf](zipf.png)",
        "",
        "![Heaps](heaps.png)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Zipf's and Heaps' laws on the corpus.")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    print(f"[stats] reading {CORPUS_FILE}")
    counts, heaps = count_terms(args.max_docs)
    n_docs = len(heaps["none"])

    zipf      = {m: fit_zipf(counts[m]) for m in METHODS}
    heaps_fit = {m: fit_heaps(heaps[m]) for m in METHODS}
    for m in METHODS:
        print(f"[stats] {m:6}  V = {len(counts[m]):>8,}   Zipf s = {zipf[m][0]:.2f}   "
              f"Heaps K = {heaps_fit[m][0]:6.1f}  β = {heaps_fit[m][1]:.3f}")

    ANALYSIS_DIR.mkdir(exist_ok=True)
    plot_zipf(counts, zipf, ANALYSIS_DIR / "zipf.png")
    plot_heaps(heaps, heaps_fit, ANALYSIS_DIR / "heaps.png")
    write_report(counts, zipf, heaps_fit, n_docs, ANALYSIS_DIR / "corpus_stats.md")

    # Vocabulary with frequencies, for spelling correction
    with VOCAB_FILE.open("w", encoding="utf-8") as f:
        for word, freq in counts["none"].most_common():
            f.write(f"{word}\t{freq}\n")
    print(f"[stats] wrote analysis/zipf.png, analysis/heaps.png, "
          f"analysis/corpus_stats.md, {VOCAB_FILE.name}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
