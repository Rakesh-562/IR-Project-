"""
spelling.py — Phase 2: edit distances and spelling correction for queries.

Edit distances (written from scratch, dynamic programming):
    hamming(a, b)          positions that differ; strings must be the same length
    lcs(a, b)              longest common subsequence (letters in order, gaps allowed)
    levenshtein(a, b)      fewest insertions, deletions and replacements
    edit_operations(a, b)  the operations themselves (backtrace through the table)

Spelling correction (the k-gram method from the course):
    1. Vocabulary = words of our corpus that occur at least MIN_FREQ times
       (from data/vocabulary.tsv, written by corpus_stats.py).
    2. A word in the vocabulary is left alone.
    3. Otherwise, candidates = vocabulary words sharing enough character
       bigrams with it (Jaccard ≥ MIN_JACCARD), found with a bigram index.
    4. Keep candidates within Levenshtein distance MAX_DISTANCE; pick the
       closest, and among equally close ones the most frequent.

Case-name matching: find judgments whose title contains words close to the
query, e.g. "putaswami" → "Justice K.S.Puttaswamy (Retd) vs Union Of India".

Run:
    python spelling.py demo                       # edit distances on examples
    python spelling.py correct "anticipatary bial habeus corpas"
    python spelling.py case "putaswami"
    python spelling.py evaluate                   # accuracy on eval/misspellings.tsv
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent
VOCAB_FILE    = BASE_DIR / "data" / "vocabulary.tsv"
CORPUS_FILE   = BASE_DIR / "data" / "corpus_dedup.jsonl"
TEST_FILE     = BASE_DIR / "eval" / "misspellings.tsv"

MIN_FREQ      = 5      # rarer corpus words may themselves be typos, so ignore them
MIN_JACCARD   = 0.3    # bigram overlap needed to be a candidate
MAX_DISTANCE  = 2      # largest edit distance we correct

# ══════════════════════════════════════════════════════════════════════════════
# Edit distances
# ══════════════════════════════════════════════════════════════════════════════

def hamming(a: str, b: str) -> int:
    """Number of positions where the letters differ (same-length strings only)."""
    if len(a) != len(b):
        raise ValueError("Hamming distance needs strings of equal length")
    return sum(x != y for x, y in zip(a, b))

def lcs(a: str, b: str) -> str:
    """Longest common subsequence: letters of both strings, in order, gaps allowed."""
    # L[i][j] = length of the LCS of a[:i] and b[:j]
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # Backtrace from the bottom-right corner to read the subsequence
    out, i, j = [], len(a), len(b)
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1]); i -= 1; j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))

def edit_table(a: str, b: str) -> list[list[int]]:
    """D[i][j] = Levenshtein distance between a[:i] and b[:j]."""
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i                                   # delete all of a[:i]
    for j in range(len(b) + 1):
        D[0][j] = j                                   # insert all of b[:j]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            same = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1,            # delete a[i-1]
                          D[i][j - 1] + 1,            # insert b[j-1]
                          D[i - 1][j - 1] + same)     # keep or replace
    return D

def levenshtein(a: str, b: str) -> int:
    return edit_table(a, b)[len(a)][len(b)]

def edit_operations(a: str, b: str) -> list[str]:
    """The steps that turn a into b, found by walking back through the table."""
    D = edit_table(a, b)
    ops, i, j = [], len(a), len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            ops.append(f"keep {a[i - 1]}"); i -= 1; j -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            ops.append(f"replace {a[i - 1]}→{b[j - 1]}"); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(f"delete {a[i - 1]}"); i -= 1
        else:
            ops.append(f"insert {b[j - 1]}"); j -= 1
    return list(reversed(ops))

def similarity(a: str, b: str) -> float:
    """1.0 = identical, 0.0 = nothing in common (Levenshtein, scaled by length)."""
    return 1 - levenshtein(a, b) / max(len(a), len(b), 1)

# ══════════════════════════════════════════════════════════════════════════════
# Spelling correction with a bigram index
# ══════════════════════════════════════════════════════════════════════════════

def bigrams(word: str) -> set[str]:
    """Character bigrams with '$' marking the word boundaries: 'bail' → $b ba ai il l$."""
    w = f"${word}$"
    return {w[i:i + 2] for i in range(len(w) - 1)}

class SpellCorrector:
    def __init__(self, vocab_file: Path = VOCAB_FILE):
        self.freq: dict[str, int] = {}
        with vocab_file.open(encoding="utf-8") as f:
            for line in f:
                word, count = line.rstrip("\n").split("\t")
                if int(count) >= MIN_FREQ and word.isalpha() and len(word) > 2:
                    self.freq[word] = int(count)

        # Bigram index: bigram → words containing it
        self.index: dict[str, list[str]] = defaultdict(list)
        for word in self.freq:
            for g in bigrams(word):
                self.index[g].append(word)

    def candidates(self, word: str) -> list[str]:
        """Vocabulary words whose bigram sets overlap the word's by Jaccard ≥ MIN_JACCARD."""
        grams = bigrams(word)
        shared = defaultdict(int)                       # candidate → shared bigrams
        for g in grams:
            for other in self.index.get(g, []):
                shared[other] += 1
        result = []
        for other, n in shared.items():
            union = len(grams) + len(bigrams(other)) - n
            if n / union >= MIN_JACCARD:
                result.append(other)
        return result

    def correct_word(self, word: str) -> str:
        if word in self.freq or not word.isalpha() or len(word) <= 2:
            return word                                 # known word, number or too short
        best = None                                     # (distance, -frequency, word)
        for cand in self.candidates(word):
            if abs(len(cand) - len(word)) > MAX_DISTANCE:
                continue                                # can't be close enough
            d = levenshtein(word, cand)
            if d <= MAX_DISTANCE:
                key = (d, -self.freq[cand], cand)
                if best is None or key < best:
                    best = key
        return best[2] if best else word

    def correct_query(self, query: str) -> str:
        return " ".join(self.correct_word(w) for w in query.lower().split())

# ══════════════════════════════════════════════════════════════════════════════
# Case-name matching
# ══════════════════════════════════════════════════════════════════════════════

def load_titles(corpus_file: Path = CORPUS_FILE) -> list[str]:
    with corpus_file.open(encoding="utf-8") as f:
        return [json.loads(line)["title"] for line in f]

def match_case_names(query: str, titles: list[str], top: int = 5) -> list[tuple[float, str]]:
    """
    Score each title by how well the query words match its words:
    for every query word, the best similarity to any title word; then average.
    """
    query_words = [w for w in query.lower().split() if len(w) > 2]
    best_cache: dict[tuple[str, str], float] = {}       # (query word, title word) → similarity
    scored = []
    for title in titles:
        # Split on anything that isn't a letter: "K.S.Puttaswamy(Retd)&" → k, s, puttaswamy, retd
        title_words = set(re.findall(r"[a-z]+", title.lower()))
        total = 0.0
        for q in query_words:
            best = 0.0
            for t in title_words:
                if len(t) > 2:
                    if (q, t) not in best_cache:
                        best_cache[(q, t)] = similarity(q, t)
                    best = max(best, best_cache[(q, t)])
            total += best
        scored.append((total / max(len(query_words), 1), title))
    scored.sort(reverse=True)
    return scored[:top]

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation and demo
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(corrector: SpellCorrector, test_file: Path = TEST_FILE) -> None:
    """Accuracy on hand-made misspellings: file lines are 'wrong <TAB> right'."""
    pairs = [line.split("\t") for line in test_file.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    correct, missing = 0, []
    print(f"{'misspelt':16} {'expected':16} {'got':16}")
    for wrong, right in pairs:
        right = right.strip()
        if right not in corrector.freq:
            missing.append(right)                       # can't be found: not in vocabulary
        got = corrector.correct_word(wrong)
        ok = got == right
        correct += ok
        print(f"{wrong:16} {right:16} {got:16} {'✓' if ok else '✗'}")
    print(f"\nAccuracy: {correct}/{len(pairs)} = {correct / len(pairs):.0%}")
    if missing:
        print(f"Expected words not in the vocabulary (≥{MIN_FREQ} occurrences): {missing}")

def demo() -> None:
    for a, b in [("bail", "jail"), ("privacy", "privecy"), ("habeas", "habeus")]:
        print(f"hamming({a}, {b}) = {hamming(a, b)}")
    for a, b in [("puttaswamy", "putaswami"), ("anticipatory", "anticipatary"),
                 ("judgment", "judgement")]:
        print(f"\n{a} → {b}")
        print(f"  lcs         = {lcs(a, b)!r} (length {len(lcs(a, b))})")
        print(f"  levenshtein = {levenshtein(a, b)}")
        print(f"  operations  = {', '.join(op for op in edit_operations(a, b) if not op.startswith('keep'))}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    text = " ".join(sys.argv[2:])
    if command == "demo":
        demo()
    elif command == "correct":
        print(SpellCorrector().correct_query(text))
    elif command == "case":
        for score, title in match_case_names(text, load_titles()):
            print(f"{score:.2f}  {title}")
    elif command == "evaluate":
        evaluate(SpellCorrector())
    else:
        sys.exit(__doc__)
