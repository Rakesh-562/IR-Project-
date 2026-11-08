"""
dedup.py — Phase 1: find near-duplicate documents in the corpus.

Three steps, as in the course:

  1. Shingling   Each document becomes the SET of its word 5-grams
                 ("shingles"). Two documents are similar if their shingle sets
                 overlap a lot: Jaccard(A, B) = |A ∩ B| / |A ∪ B|.

  2. MinHash     Shingle sets are huge, so each one is summarised by a
                 signature of NUM_HASHES numbers. For each of NUM_HASHES random
                 hash functions h, the signature keeps min(h(s) for s in set).
                 Key property: P[minhashes equal] = Jaccard(A, B), so
                 the fraction of equal positions estimates the Jaccard.

  3. LSH         Comparing every pair of signatures is O(n²). Instead the
                 signature is cut into BANDS bands of ROWS rows. Documents
                 whose band is identical land in the same bucket and become
                 candidate pairs. Only candidates are compared.
                 P[candidate] = 1 - (1 - s^ROWS)^BANDS   (the "S-curve")

Input:   data/corpus.jsonl  (from crawl.py)
Output:  data/duplicates.tsv       doc_a <TAB> doc_b <TAB> estimated Jaccard
         data/corpus_dedup.jsonl   corpus without the duplicates

Run:
    python dedup.py              # uses data/
    python dedup.py some/folder  # uses another folder
"""

import json
import sys
import zlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Settings ──────────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).with_name("data")
SHINGLE_SIZE = 5       # words per shingle
NUM_HASHES   = 128     # signature length
BANDS        = 32      # LSH bands   (BANDS × ROWS must equal NUM_HASHES)
ROWS         = 4       # rows per band
THRESHOLD    = 0.8     # estimated Jaccard at or above this = near-duplicate

PRIME = 2**31 - 1      # hash functions are h(x) = (a·x + b) mod PRIME
rng = np.random.default_rng(seed=42)               # fixed seed → same results every run
A = rng.integers(1, PRIME, size=NUM_HASHES, dtype=np.int64)
B = rng.integers(0, PRIME, size=NUM_HASHES, dtype=np.int64)

# ── Step 1: shingling ─────────────────────────────────────────────────────────

def shingles(text: str) -> set[int]:
    """The set of word 5-grams of a text, each stored as a number (CRC32 hash)."""
    words = text.lower().split()
    return {
        zlib.crc32(" ".join(words[i:i + SHINGLE_SIZE]).encode()) % PRIME
        for i in range(len(words) - SHINGLE_SIZE + 1)
    }

def jaccard(a: set[int], b: set[int]) -> float:
    """Exact Jaccard similarity — used only to check the MinHash estimate."""
    return len(a & b) / len(a | b) if a or b else 0.0

# ── Step 2: MinHash signatures ────────────────────────────────────────────────

def minhash(shingle_set: set[int]) -> np.ndarray:
    """Signature: for each hash function, the smallest hash over all shingles."""
    x = np.fromiter(shingle_set, dtype=np.int64)
    # Row i of the table = hash function i applied to every shingle
    # (a, x < 2^31, so a·x < 2^62 fits in int64 without overflow)
    hashed = (np.outer(A, x) + B[:, None]) % PRIME
    return hashed.min(axis=1)

def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of signature positions that agree ≈ Jaccard similarity."""
    return float(np.mean(sig_a == sig_b))

# ── Step 3: LSH banding ───────────────────────────────────────────────────────

def candidate_pairs(signatures: dict[str, np.ndarray]) -> set[tuple[str, str]]:
    """Documents sharing at least one identical band become candidate pairs."""
    buckets = defaultdict(list)
    for doc_id, sig in signatures.items():
        for band in range(BANDS):
            rows = sig[band * ROWS:(band + 1) * ROWS]
            buckets[(band, tuple(rows))].append(doc_id)

    pairs = set()
    for docs in buckets.values():
        for a, b in combinations(sorted(docs), 2):
            pairs.add((a, b))
    return pairs

def print_s_curve() -> None:
    """Show how likely a pair with similarity s is to become a candidate."""
    print(f"\nS-curve for {BANDS} bands × {ROWS} rows "
          f"(threshold ≈ (1/b)^(1/r) = {(1 / BANDS) ** (1 / ROWS):.2f}):")
    for s in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = 1 - (1 - s ** ROWS) ** BANDS
        print(f"  Jaccard {s:.1f} → P(candidate) = {p:.3f}  {'█' * round(p * 40)}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: Path) -> None:
    corpus_file = data_dir / "corpus.jsonl"

    # Pass 1: one signature per document (texts are not kept in memory)
    signatures, titles = {}, {}
    with corpus_file.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            sh = shingles(doc["text"])
            if sh:                                    # skip texts under 5 words
                signatures[doc["doc_id"]] = minhash(sh)
                titles[doc["doc_id"]] = doc["title"]
    print(f"[dedup] {len(signatures)} documents signed")

    print_s_curve()

    # LSH → candidates → keep pairs whose estimated Jaccard ≥ THRESHOLD
    candidates = candidate_pairs(signatures)
    duplicates = []
    for a, b in candidates:
        sim = estimated_jaccard(signatures[a], signatures[b])
        if sim >= THRESHOLD:
            duplicates.append((a, b, sim))
    duplicates.sort(key=lambda d: -d[2])

    all_pairs = len(signatures) * (len(signatures) - 1) // 2
    print(f"\n[dedup] {all_pairs} possible pairs → {len(candidates)} LSH candidates "
          f"→ {len(duplicates)} near-duplicates (Jaccard ≥ {THRESHOLD})")

    with (data_dir / "duplicates.tsv").open("w", encoding="utf-8") as f:
        for a, b, sim in duplicates:
            f.write(f"{a}\t{b}\t{sim:.3f}\n")

    # Of each duplicate pair, drop the second document (doc_ids are sorted
    # inside a pair, so the same document of a pair is always the one kept)
    removed = {b for _, b, _ in duplicates}
    kept = 0
    with corpus_file.open(encoding="utf-8") as f, \
         (data_dir / "corpus_dedup.jsonl").open("w", encoding="utf-8") as out:
        for line in f:
            if json.loads(line)["doc_id"] not in removed:
                out.write(line)
                kept += 1
    print(f"[dedup] removed {len(removed)} documents → {kept} kept in corpus_dedup.jsonl")

    if duplicates:
        print_examples(corpus_file, duplicates[:10], titles)

def print_examples(corpus_file: Path, top: list, titles: dict[str, str]) -> None:
    """Show the top pairs, with the exact Jaccard next to the MinHash estimate."""
    wanted = {doc_id for a, b, _ in top for doc_id in (a, b)}
    texts = {}
    with corpus_file.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["doc_id"] in wanted:
                texts[doc["doc_id"]] = doc["text"]

    print("\nTop near-duplicate pairs (estimated vs exact Jaccard):")
    for a, b, sim in top:
        exact = jaccard(shingles(texts[a]), shingles(texts[b]))
        print(f"  est {sim:.2f} | exact {exact:.2f}  {titles[a][:55]}")
        print(f"                       {titles[b][:55]}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR)
