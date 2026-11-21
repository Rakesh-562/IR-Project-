"""
index.py — Phase 3: a positional inverted index over the corpus.

For every term, the index stores its POSTING LIST: which documents contain
it, and at which positions. Example for the term "bail":

    docs      = [3, 17, 42]                     documents containing "bail" (increasing)
    starts    = [0, 2, 3, 7]                    where each document's positions begin
    positions = [12, 80,  5,  1, 9, 33, 40]     word positions inside the documents
                 └doc 3┘ └17┘ └─── doc 42 ───┘

    positions of the k-th document = positions[starts[k] : starts[k+1]]
    term frequency (tf)            = starts[k+1] - starts[k]

The three lists are Python `array`s of 4-byte integers. That is far smaller
than lists of Python ints (~36 bytes each), so the ~30 million positions of
the corpus fit in memory.

Documents and positions are numbered from 1, so every gap between two
numbers is at least 1. The compression codes in compression.py (γ, Golomb)
cannot encode 0.

Positions are counted AFTER stopwords are removed (text_processing.analyze),
so the phrase "right to privacy" is stored as "right privaci" at
neighbouring positions.

Run:
    python index.py build                  # Porter-stemmed index → data/index_porter.pkl
    python index.py build --method lemma   # another normalization (compared in Phase 5)
    python index.py info                   # statistics of a built index
"""

import argparse
import json
import pickle
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from text_processing import METHODS, analyze

BASE_DIR    = Path(__file__).resolve().parent
CORPUS_FILE = BASE_DIR / "data" / "corpus_dedup.jsonl"

def index_file(method: str) -> Path:
    return BASE_DIR / "data" / f"index_{method}.pkl"

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PostingList:
    docs: array          # document numbers, increasing
    starts: array        # len(docs) + 1 entries; see the module docstring
    positions: array     # all positions, document after document

    def tf(self, k: int) -> int:
        """How often the term occurs in the k-th document of the list."""
        return self.starts[k + 1] - self.starts[k]

    def positions_of(self, k: int) -> array:
        """Positions of the term in the k-th document of the list."""
        return self.positions[self.starts[k]:self.starts[k + 1]]

    def __len__(self) -> int:              # document frequency (df)
        return len(self.docs)

@dataclass
class InvertedIndex:
    method: str                            # normalization used ("porter", "lemma", …)
    postings: dict[str, PostingList]       # term → posting list
    doc_ids: list[str]                     # document number → corpus doc_id ([0] unused)
    titles: list[str]                      # document number → title          ([0] unused)
    doc_len: array                         # document number → number of terms ([0] unused)

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids) - 1

    @property
    def avg_doc_len(self) -> float:
        return sum(self.doc_len) / self.num_docs

    def df(self, term: str) -> int:
        return len(self.postings[term]) if term in self.postings else 0

# ── Building ──────────────────────────────────────────────────────────────────

def build(method: str = "porter", corpus_file: Path = CORPUS_FILE) -> InvertedIndex:
    """Read the corpus once and build the posting lists."""
    postings: dict[str, PostingList] = {}
    doc_ids, titles, doc_len = [""], [""], array("I", [0])

    with corpus_file.open(encoding="utf-8") as f:
        for doc_num, line in enumerate(f, start=1):
            doc = json.loads(line)
            terms = analyze(doc["title"] + " " + doc["text"], method)

            # Positions of each term inside this document (counting from 1)
            where = defaultdict(list)
            for position, term in enumerate(terms, start=1):
                where[term].append(position)

            # Append this document to the posting list of each of its terms.
            # Documents are read in order, so every list stays sorted.
            for term, positions in where.items():
                plist = postings.get(term)
                if plist is None:
                    plist = postings[term] = PostingList(array("I"), array("I", [0]), array("I"))
                plist.docs.append(doc_num)
                plist.positions.extend(positions)
                plist.starts.append(len(plist.positions))

            doc_ids.append(doc["doc_id"])
            titles.append(doc["title"])
            doc_len.append(len(terms))
            if doc_num % 500 == 0:
                print(f"  {doc_num} documents, {len(postings):,} terms")

    return InvertedIndex(method, postings, doc_ids, titles, doc_len)

# The index is saved as plain data (strings, lists and arrays), not as the
# classes above: a pickled class remembers which module it came from, and
# "python index.py build" would record it as __main__, which other scripts
# can't load.

def save(index: InvertedIndex) -> Path:
    data = {
        "method":   index.method,
        "postings": {t: (p.docs, p.starts, p.positions) for t, p in index.postings.items()},
        "doc_ids":  index.doc_ids,
        "titles":   index.titles,
        "doc_len":  index.doc_len,
    }
    path = index_file(index.method)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def load(method: str = "porter") -> InvertedIndex:
    path = index_file(method)
    if not path.exists():
        sys.exit(f"No index at {path}. Build it first: python index.py build --method {method}")
    with path.open("rb") as f:
        data = pickle.load(f)
    postings = {t: PostingList(*arrays) for t, arrays in data["postings"].items()}
    return InvertedIndex(data["method"], postings, data["doc_ids"], data["titles"], data["doc_len"])

# ── Statistics ────────────────────────────────────────────────────────────────

def info(index: InvertedIndex) -> None:
    n_postings  = sum(len(p) for p in index.postings.values())
    n_positions = sum(len(p.positions) for p in index.postings.values())
    by_df = sorted(index.postings, key=index.df, reverse=True)
    print(f"method              : {index.method}")
    print(f"documents           : {index.num_docs:,}")
    print(f"terms (vocabulary)  : {len(index.postings):,}")
    print(f"postings (term,doc) : {n_postings:,}")
    print(f"positions           : {n_positions:,}")
    print(f"average doc length  : {index.avg_doc_len:,.0f} terms")
    print(f"index file          : {index_file(index.method).stat().st_size / 1e6:,.0f} MB")
    print(f"terms in 1 document : {sum(1 for p in index.postings.values() if len(p) == 1):,}")
    print("highest df          : " + ", ".join(f"{t} ({index.df(t)})" for t in by_df[:10]))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build or inspect the inverted index.")
    parser.add_argument("command", choices=["build", "info"])
    parser.add_argument("--method", default="porter", choices=METHODS)
    args = parser.parse_args()

    if args.command == "build":
        start = time.time()
        idx = build(args.method)
        path = save(idx)
        print(f"[index] built in {time.time() - start:.0f}s → {path}")
    info(load(args.method))
