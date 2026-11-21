"""
text_processing.py — Phase 2: turn raw legal text into index terms.

    text ──► tokenize() ──► remove_stopwords() ──► normalize() ──► terms

1. tokenize()          lowercase; rewrite legal abbreviations ("s. 302" →
                       "section 302", "Art. 21" → "article 21", "Cr.P.C." →
                       "crpc"); split on anything that isn't a letter or digit.
                       Numbers are KEPT: "21", "437", "304b" matter in law.
2. remove_stopwords()  drop very common words ("the", "of", "and" …).
3. normalize()         reduce words to a common form, one of:
                         "none"      judgments → judgments
                         "porter"    judgments → judgment   (Porter stemmer)
                         "snowball"  judgments → judgment   (Snowball / Porter2)
                         "lemma"     judgments → judgment   (WordNet lemmatizer)
                       Stemmers chop suffixes by rules ("arguing" → "argu");
                       the lemmatizer looks words up in a dictionary
                       ("arguing" stays "arguing" without a part of speech).

analyze(text, method) runs all three steps; it is what the index (Phase 3)
will call. Phase 5 compares the methods on the evaluation queries.
"""

import re

from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# ── 1. Tokenization ───────────────────────────────────────────────────────────

# Legal abbreviations, rewritten BEFORE splitting (the dots would be lost after).
# (?=\s*\d) means "only when a number follows", so "s." in other places is untouched.
LEGAL_ABBREVIATIONS = [
    (r"\bu/s\.?",              " under section "),   # u/s 302
    (r"\bss?\.(?=\s*\d)",      " section "),         # s. 302, ss. 3
    (r"\bsecs?\.?(?=\s*\d)",   " section "),         # sec 302, sec. 302
    (r"\barts?\.?(?=\s*\d)",   " article "),         # art. 21, arts 14
    (r"\bcr\.\s*p\.\s*c\b\.?", " crpc "),            # cr.p.c.
    (r"\bi\.\s*p\.\s*c\b\.?",  " ipc "),             # i.p.c.
    (r"\bc\.\s*p\.\s*c\b\.?",  " cpc "),             # c.p.c.
    (r"\br\.\s*t\.\s*i\b\.?",  " rti "),             # r.t.i.
]
TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    """Lowercase, expand legal abbreviations, split into letter/digit tokens."""
    text = text.lower()
    for pattern, replacement in LEGAL_ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    # Drop single letters ("a", "b" from "(b)"); keep every number ("2", "21")
    return [t for t in TOKEN.findall(text) if len(t) > 1 or t.isdigit()]

# ── 2. Stopwords ──────────────────────────────────────────────────────────────

# Standard English function words. Legal words ("court", "section", "act")
# are NOT stopwords: they are frequent but still carry meaning for a query.
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself me more most my myself no nor
not of off on once only or other ought our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
also may shall upon said thus hereby therein thereof whereas
""".split())

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]

# ── 3. Normalization (stemming / lemmatization) ───────────────────────────────

METHODS = ["none", "porter", "snowball", "lemma"]

_porter    = PorterStemmer()
_snowball  = SnowballStemmer("english")
_lemmatizer = WordNetLemmatizer()          # needs: nltk.download("wordnet")

# A corpus has millions of tokens but far fewer DIFFERENT words, so each word
# is normalized once and the answer remembered (a cache per method).
_cache: dict[str, dict[str, str]] = {method: {} for method in METHODS}

def _normalize_word(word: str, method: str) -> str:
    if word.isdigit() or method == "none":
        return word                                  # never stem numbers
    if method == "porter":
        return _porter.stem(word)
    if method == "snowball":
        return _snowball.stem(word)
    if method == "lemma":
        return _lemmatizer.lemmatize(word)
    raise ValueError(f"unknown method {method!r}; use one of {METHODS}")

def normalize(tokens: list[str], method: str = "porter") -> list[str]:
    cache = _cache[method]
    result = []
    for t in tokens:
        if t not in cache:
            cache[t] = _normalize_word(t, method)
        result.append(cache[t])
    return result

# ── All three steps ───────────────────────────────────────────────────────────

def analyze(text: str, method: str = "porter", stopwords: bool = True) -> list[str]:
    """text → index terms. The SAME function must be used for documents and queries."""
    tokens = tokenize(text)
    if stopwords:
        tokens = remove_stopwords(tokens)
    return normalize(tokens, method)


if __name__ == "__main__":
    sample = ("The accused was arrested u/s 302 I.P.C. and s. 34; bail under "
              "Sec. 437 Cr.P.C. was refused. Art. 21 guarantees the right to privacy "
              "(see Puttaswamy, paras 3(b) and 4).")
    print("text     :", sample)
    print("tokens   :", tokenize(sample))
    print("no stop  :", remove_stopwords(tokenize(sample)))
    for m in METHODS:
        print(f"{m:9}:", analyze(sample, m))
