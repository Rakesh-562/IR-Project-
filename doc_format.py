"""
doc_format.py — make a stored document readable on the search website.

The crawler stored each document as ONE line of text (all line breaks were
collapsed when the page was cleaned), starting with Indian Kanoon's header:

    [Cites 25 , Cited by 1 ] Bombay High Court Vinit Kumar vs CBI on 22 October, 2019
    Equivalent citations: … Author: Ranjit More Bench: Ranjit More , N. J. Jamadar …

and sprinkled with stamps from the court's PDF ("Digitally signed by … Date:",
"Signature Not Verified", "Page 3 of 12"). To show it properly:

    split_header()  takes the header apart: cites, cited by, citations, author, bench
    remove_noise()  deletes signature / download stamps and page counters
    paragraphs()    rebuilds paragraphs where judgments and statutes start a new
                    one: numbered paragraphs ("12. The …"), headings (JUDGMENT,
                    ORDER), sub-sections ("(1)", "Explanation"). Anything still
                    longer than MAX_PARAGRAPH_WORDS is cut at a sentence end.
    highlight()     wraps words that match the query in <mark>, using the same
                    analyze() as the index, so "privacy" also marks "privacies".
"""

import html
import re

from text_processing import analyze

MAX_PARAGRAPH_WORDS = 180

# ── 1. The header ─────────────────────────────────────────────────────────────

CITES = re.compile(r"^\s*\[Cites (\d+) , Cited by (\d+) \]\s*")
BREADCRUMBS = re.compile(r"^(?:\[ [^\]]{1,40} \]\s*)+")           # "[ Entire Act ] [ Section 125 ]"
# field → (where its value ends, longest value allowed)
FIELD_ENDS = {
    "Equivalent citations": (r"\s(?=Author:|Bench:)", 2000),   # Supreme Court lists can be long
    "Author": (r"\s(?=Bench:)", 200),
    # The bench list ends where the judgment's own text starts: a word in capitals,
    # a "*", a signature stamp, or a number.
    "Bench": (r"\s(?=[A-Z]{2,}\b|\*|Digitally|Signature|\d)", 300),
}

def split_header(text: str, title: str, court: str) -> tuple[dict, str]:
    """Return ({"cites", "cited_by", "Equivalent citations", "Author", "Bench"}, body text)."""
    meta = {}
    match = CITES.match(text)
    if match:
        meta["cites"], meta["cited_by"] = int(match.group(1)), int(match.group(2))
        text = text[match.end():]
    text = BREADCRUMBS.sub("", text)
    for repeated in (court, title):                                   # the header repeats both
        if repeated and text.startswith(repeated):
            text = text[len(repeated):].lstrip()
    for name, (end, longest) in FIELD_ENDS.items():
        if text.startswith(name + ":"):
            rest = text[len(name) + 1:]
            stop = re.search(end, rest[:longest])
            cut = stop.start() if stop else min(len(rest), longest)
            meta[name] = rest[:cut].strip(" ,")
            text = rest[cut:].lstrip()
    return meta, text

# ── 2. Noise from the court PDFs ──────────────────────────────────────────────

NOISE = [
    r"Signature Not Verified\s*(?:Digitally Signed\s*By\s*:?\s*[A-Z][A-Z .]*?)?\s*"
    r"(?:Signing Date\s*:?\s*[\d.]+(?:\s+[\d:]+)?)?",
    r"Digitally signed by .{0,80}?Date:\s?\d{4}[.\-]\d{2}[.\-]\d{2}(?:\s\d{2}:\d{2}:\d{2}(?:\s?[+-]\d{4})?)?",
    r"\b\d{2}:\d{2}:\d{2}\s?[+-]\d{4}\b",                  # a signature's time, split off by the PDF layout
    r":::\s*(?:Downloaded|Uploaded) on\s*-\s*[\d/]+\s+[\d:]+\s*:::",
    r"http://www\.judis\.nic\.in",
    r"\bPage \d+ of \d+\b",
]

def remove_noise(text: str) -> str:
    for pattern in NOISE:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

# ── 3. Paragraphs ─────────────────────────────────────────────────────────────

# A new paragraph starts (the split happens at the space BEFORE these):
PARAGRAPH_START = re.compile(
    r"(?<=[.:;\"”’)\]])\s(?=\d{1,3}\.\s+[A-Z“\"(])"                  # "… end. 12. The …"
    r"|\s(?=(?:JUDGMENT|J U D G M E N T|ORDER|O R D E R|HEADNOTE:|CORAM:)\s)"
    r"|\s(?=\(\d{1,2}[A-Z]?\)\s)"                                      # statute sub-section "(1) "
    r"|\s(?=Explanation\s*[.\d—–-])"                                   # statute explanation
)
SENTENCE_END = re.compile(r"(?<=[.?!])\s(?=[A-Z“\"(])")

def paragraphs(text: str) -> list[str]:
    result = []
    for part in PARAGRAPH_START.split(text):
        words = part.split()
        if len(words) <= MAX_PARAGRAPH_WORDS:
            if part.strip():
                result.append(part.strip())
            continue
        # Too long: cut at the sentence end nearest to every MAX_PARAGRAPH_WORDS words
        current = []
        for sentence in SENTENCE_END.split(part):
            current.append(sentence)
            if sum(len(s.split()) for s in current) >= MAX_PARAGRAPH_WORDS:
                result.append(" ".join(current).strip())
                current = []
        if current:
            result.append(" ".join(current).strip())
    return result

# ── 4. Highlighting the query ─────────────────────────────────────────────────

def highlight(paragraph: str, query_terms: set[str], cache: dict) -> tuple[str, int]:
    """HTML for one paragraph with matching words in <mark>; also the number of matches."""
    out, hits = [], 0
    for word in paragraph.split():
        if word not in cache:                       # each distinct word is analyzed once
            cache[word] = bool(query_terms & set(analyze(word)))
        if cache[word]:
            out.append(f"<mark>{html.escape(word)}</mark>")
            hits += 1
        else:
            out.append(html.escape(word))
    return " ".join(out), hits

def format_document(doc: dict, query: str, max_words: int) -> dict:
    """Everything the document page needs: metadata, paragraph HTML, best paragraph."""
    meta, body = split_header(doc["text"], doc["title"], doc["court"])
    body = remove_noise(body)
    words = body.split()
    shown = " ".join(words[:max_words])
    query_terms, cache = set(analyze(query)), {}
    html_paragraphs, hits = [], []
    for p in paragraphs(shown):
        marked, n = highlight(p, query_terms, cache)
        html_paragraphs.append(marked)
        hits.append(n)
    best = max(range(len(hits)), key=hits.__getitem__) if hits and max(hits) > 0 else None
    return {"meta": meta, "paragraphs": html_paragraphs, "best": best,
            "total_words": len(words), "cut": len(words) > max_words}

def clean_body(doc: dict) -> str:
    """The document text without header and stamps (used for result snippets)."""
    return remove_noise(split_header(doc["text"], doc["title"], doc["court"])[1])
