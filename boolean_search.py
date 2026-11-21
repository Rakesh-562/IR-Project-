"""
boolean_search.py — Phase 3: Boolean retrieval over the positional index.

Query language (AND binds tighter than OR; no parentheses):

    bail AND anticipatory                    both words
    bail anticipatory                        same (AND is implied)
    privacy OR surveillance                  either word
    bail AND NOT murder                      bail, but not murder
    "right to privacy"                       exact phrase
    bail /5 437                              the two words at most 5 words apart
    "article 21" AND privacy OR sedition     (article-21 phrase AND privacy) OR sedition

A query is an OR of CLAUSES; a clause is an AND of ITEMS; an item is a word,
a "phrase" or a near-pair, possibly with NOT in front.

The merge algorithms are written out by hand (two pointers walking down two
sorted lists), as in the course. AND-clauses are evaluated RAREST TERM FIRST:
intersecting the shortest lists first keeps the intermediate results small.

Run:
    python boolean_search.py "anticipatory AND bail AND NOT murder"
    python boolean_search.py '"right to privacy" AND surveillance'
    python boolean_search.py "bail /5 437"
"""

import re
import sys

from index import InvertedIndex, load
from text_processing import analyze

# ── Merging two sorted lists of document numbers ──────────────────────────────

def intersect(a: list[int], b: list[int]) -> list[int]:
    """Documents in both a and b (AND)."""
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result

def union(a: list[int], b: list[int]) -> list[int]:
    """Documents in a or b (OR)."""
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    return result + list(a[i:]) + list(b[j:])          # whatever is left over

def and_not(a: list[int], b: list[int]) -> list[int]:
    """Documents in a but not in b (AND NOT)."""
    result, i, j = [], 0, 0
    while i < len(a):
        if j < len(b) and b[j] < a[i]:
            j += 1
        elif j < len(b) and b[j] == a[i]:
            i += 1; j += 1                              # in both: skip it
        else:
            result.append(a[i]); i += 1
    return result

# ── Positional queries: phrases and proximity ─────────────────────────────────

def positions_by_doc(index: InvertedIndex, term: str) -> dict[int, list[int]]:
    """{document number: positions of the term in it}."""
    plist = index.postings.get(term)
    if plist is None:
        return {}
    return {doc: list(plist.positions_of(k)) for k, doc in enumerate(plist.docs)}

def phrase_docs(index: InvertedIndex, terms: list[str]) -> list[int]:
    """Documents where the terms appear next to each other, in order."""
    candidates = and_all(index, terms)                  # must contain every term first
    where = [positions_by_doc(index, t) for t in terms]
    result = []
    for doc in candidates:
        later = [set(w[doc]) for w in where[1:]]
        # Is there a start p with term i at position p + i for every i?
        if any(all(p + i in later[i - 1] for i in range(1, len(terms))) for p in where[0][doc]):
            result.append(doc)
    return result

def near_docs(index: InvertedIndex, term1: str, term2: str, k: int) -> list[int]:
    """Documents where term1 and term2 occur at most k positions apart (either order)."""
    candidates = and_all(index, [term1, term2])
    where1, where2 = positions_by_doc(index, term1), positions_by_doc(index, term2)
    result = []
    for doc in candidates:
        p1, p2, i, j = where1[doc], where2[doc], 0, 0
        while i < len(p1) and j < len(p2):              # two pointers over sorted positions
            if abs(p1[i] - p2[j]) <= k:
                result.append(doc)
                break
            if p1[i] < p2[j]:
                i += 1
            else:
                j += 1
    return result

# ── Evaluating a query ────────────────────────────────────────────────────────

def docs_of(index: InvertedIndex, term: str) -> list[int]:
    plist = index.postings.get(term)
    return list(plist.docs) if plist else []

def and_all(index: InvertedIndex, terms: list[str]) -> list[int]:
    """AND of several terms, rarest first."""
    ordered = sorted(set(terms), key=index.df)
    result = docs_of(index, ordered[0])
    for term in ordered[1:]:
        if not result:
            break                                       # nothing left: stop early
        result = intersect(result, docs_of(index, term))
    return result

def parse(query: str) -> list[list[tuple[bool, tuple]]]:
    """
    Split a query into clauses (OR) of items (AND). Each item is
    (negated, ("word", w) | ("phrase", text) | ("near", w1, w2, k)).
    """
    parts = re.findall(r'"[^"]*"|\S+', query)
    clauses, current, negate, i = [], [], False, 0
    while i < len(parts):
        part = parts[i]
        if part == "OR":
            clauses.append(current); current = []
        elif part == "NOT":
            negate = True
        elif part != "AND":
            if i + 2 < len(parts) and re.fullmatch(r"/\d+", parts[i + 1]):
                item = ("near", part, parts[i + 2], int(parts[i + 1][1:]))
                i += 2
            elif part.startswith('"'):
                item = ("phrase", part.strip('"'))
            else:
                item = ("word", part)
            current.append((negate, item))
            negate = False
        i += 1
    clauses.append(current)
    return [c for c in clauses if c]

def evaluate_item(index: InvertedIndex, item: tuple) -> list[int] | None:
    """Documents matching one item. None if the item has no searchable words."""
    kind = item[0]
    if kind == "near":
        t1, t2 = analyze(item[1], index.method), analyze(item[2], index.method)
        return near_docs(index, t1[0], t2[0], item[3]) if t1 and t2 else None
    terms = analyze(item[1], index.method)              # same processing as the documents
    if not terms:
        return None                                     # only stopwords, e.g. "the"
    if kind == "word" and len(terms) == 1:
        return docs_of(index, terms[0])
    return phrase_docs(index, terms)                    # a phrase, or "u/s" → 2 terms

def search(index: InvertedIndex, query: str, verbose: bool = False) -> list[int]:
    all_docs = list(range(1, index.num_docs + 1))
    result: list[int] = []
    for clause in parse(query):
        positives, negatives = [], []
        for negated, item in clause:
            docs = evaluate_item(index, item)
            if docs is not None:
                (negatives if negated else positives).append(docs)

        # AND the positive items, shortest list first; then remove the negatives
        positives.sort(key=len)
        if verbose:
            print(f"  clause: AND of lists with sizes {[len(p) for p in positives]}"
                  + (f", NOT {[len(n) for n in negatives]}" if negatives else ""))
        docs = positives[0] if positives else all_docs
        for other in positives[1:]:
            docs = intersect(docs, other)
        for other in negatives:
            docs = and_not(docs, other)
        result = union(result, docs)
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    query = " ".join(sys.argv[1:]) or input("Boolean query: ")
    idx = load("porter")
    hits = search(idx, query, verbose=True)
    print(f"\n{len(hits)} documents match: {query}\n")
    for doc in hits[:10]:
        print(f"  {idx.doc_ids[doc]}  {idx.titles[doc][:80]}")
