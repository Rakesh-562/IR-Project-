"""
rag.py — Local-First Hybrid Retrieval-Augmented Generation (RAG) for Indian Legal Documents.

Design (follows the "Word Embedding in IR" and "Advanced IR / RAG" lecture decks):

 1. Representation  : SBERT-style *sentence* embeddings (bi-encoder / siamese network).
                      Chosen over Word2Vec/GloVe (context-independent, one vector per word, e.g.
                      "bank" = river bank = court of law) and over raw BERT token averaging
                      (the deck notes it "didn't work well"). Legal text is full of polysemy
                      ("order", "party", "consideration", "suit"), so contextual,
                      sentence-level vectors are the right fit.
 2. Hybrid retrieval: BM25 (exact legal strings: "Article 21", "Section 302", party names) +
                      dense SBERT retrieval, fused with Reciprocal Rank Fusion (RRF).
 3. Chunking        : syntax-based (paragraph / sentence aware, legal-abbreviation safe) with
                      sentence overlap — instead of blind fixed-length windows
                      ("chunking matters a lot").
 4. Max-match       : long judgments are sliced into windows, every window is scored against
                      the short query and the best windows win (Shashavali et al. idea).
 5. Light GraphRAG  : in-corpus citation graph (`cites`) -> "local approach / look at neighbours";
                      statutory-provision index (Article/Section) acts as a mini knowledge graph
                      to boost documents/passages sharing the provisions named in the query.
 6. Routing         : local corpus first (lexical OR semantic confidence); otherwise a
                      query-specific crawl via crawl.py.
 7. Generation      : grounded prompt -> Gemini / Anthropic / OpenAI / Ollama, with citation
                      validation; offline extractive summarizer if no LLM is available.

Graceful degradation: if `sentence-transformers` is not installed (or --no-dense is passed) the
system falls back to BM25-only retrieval, exactly like the previous version.

Install:
    pip install rank-bm25 httpx python-dotenv numpy sentence-transformers

Optional .env settings:
    EMBED_MODEL=BAAI/bge-small-en-v1.5      # any sentence-transformers model
                                            # (e.g. BAAI/bge-base-en-v1.5, intfloat/e5-base-v2,
                                            #  multi-qa-mpnet-base-dot-v1, law-ai/InLegalBERT-based
                                            #  sentence models if you have one fine-tuned)
    ANTHROPIC_MODEL / OPENAI_MODEL / OLLAMA_MODEL / GEMINI_MODELS (comma separated)

Usage:
    python rag.py "right to privacy Article 21 Puttaswamy"
    python rag.py "digital personal data protection bill exemptions" --force-web
    python rag.py --no-dense "section 302 IPC murder"      # lexical only
    python rag.py                                          # Interactive mode (loops)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

import crawl

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EMBED_DIR = DATA_DIR / "embeddings"
LOCAL_CORPUS_DEDUP = DATA_DIR / "corpus_dedup.jsonl"
LOCAL_CORPUS_RAW = DATA_DIR / "corpus.jsonl"
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "")  # e.g. http://localhost:11434
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-flash-latest,gemini-3.8-flash",
    ).split(",") if m.strip()
]

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval tuning constants
RRF_K = 60                  # Reciprocal Rank Fusion constant
LEXICAL_WEIGHT = 1.0        # legal text is term-sensitive: keep BM25 on equal footing with dense
DENSE_WEIGHT = 1.0
DOC_INDEX_CHARS = 6000      # chars per document used for BM25 / provision index
DOC_EMBED_CHARS = 1800      # chars per document used for the document-level embedding
PASSAGE_DENSE_FLOOR = 0.50  # a passage with no query term overlap must beat this cosine to qualify
GRAPH_MIN_DENSE = 0.50      # min cosine for a citation-neighbour to be pulled in without BM25 hit
PROVISION_BOOST = 0.008     # RRF-scale boost per query provision matched (capped below)
PROVISION_BOOST_CAP = 0.024
CASE_NAME_BOOST = 0.03
HOLDING_BOOST = 0.003


# ── Tokenization (legal-aware) ────────────────────────────────────────────────

# Common English and query stopwords for coverage checks
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "tell", "explain", "summarize", "find", "search", "give", "regarding", "versus", "vs", "v",
}

# "Article 21", "Art. 14", "Section 302", "Sec 8(1)(a)", "s. 420", "Order 39 Rule 1" ...
# Normalised to compact tokens such as art_21 / sec_302 / ord_39 / rule_1 so that
# "Article 21(1)" in a judgment lexically matches "article 21" in a query.
_PROV_KIND = {
    "article": "art", "articles": "art", "art": "art",
    "section": "sec", "sections": "sec", "sec": "sec", "s": "sec",
    "order": "ord", "rule": "rule", "schedule": "sch", "clause": "cl",
}
PROVISION_RE = re.compile(
    r"\b(?:(articles?|art|sections?|sec|order|rule|schedule|clause)(?:\.\s*|\s+)|(s)\.\s*)"
    r"(\d{1,4}[a-z]{0,2})((?:\s*\(\s*[0-9a-z]{1,4}\s*\))*)",
    re.IGNORECASE,
)
# Human-readable provision mentions (for the summary), keeps sub-clauses.
PROVISION_DISPLAY_RE = re.compile(
    r"(?:Article|Art\.|Section|Sec\.|s\.)\s*\d{1,4}[A-Z]{0,2}(?:\s*\([0-9a-zA-Z]{1,4}\))*",
    re.IGNORECASE,
)


def extract_provisions(text: str) -> set[str]:
    """Return canonical provision tokens (e.g. {'art_21', 'sec_302'}) mentioned in text."""
    found: set[str] = set()
    for m in PROVISION_RE.finditer(text):
        kind = _PROV_KIND[(m.group(1) or m.group(2)).lower()]
        found.add(f"{kind}_{m.group(3).lower()}")
    return found


def tokenize(text: str) -> list[str]:
    """Word tokens + canonical provision tokens (so 'Article 21' is matched as one unit too)."""
    tokens = re.findall(r"\w+", text.lower())
    tokens.extend(extract_provisions(text))
    return tokens


def content_tokens(text: str) -> list[str]:
    """Query tokens without stopwords (keeps numbers and provision tokens)."""
    toks = tokenize(text)
    out = [t for t in toks if t not in STOPWORDS and (len(t) > 2 or t.isdigit())]
    return out or toks


CASE_RE = re.compile(
    r"([A-Za-z][\w.&' -]{2,60}?)\s+(?:v\.?|vs\.?|versus)\s+([A-Za-z][\w.&' -]{2,60})",
    re.IGNORECASE,
)


def parse_case_query(query: str) -> list[set[str]] | None:
    """'Kesavananda Bharati v. State of Kerala' -> [{'kesavananda','bharati'}, {'state','kerala'}]."""
    m = CASE_RE.search(query)
    if not m:
        return None
    left = [t for t in re.findall(r"\w+", m.group(1).lower()) if t not in STOPWORDS and len(t) > 2]
    right = [t for t in re.findall(r"\w+", m.group(2).lower()) if t not in STOPWORDS and len(t) > 2]
    left, right = left[-4:], right[:4]
    if not left or not right:
        return None
    return [set(left), set(right)]


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ── Unified Data Structures ───────────────────────────────────────────────────

@dataclass
class Document:
    """Unified document representation for both local and scraped documents."""
    doc_id: str
    title: str
    content: str
    url: str
    court: str = ""
    date: str = ""
    doc_type: str = "document"
    source: str = "local"           # 'local_corpus', 'indiankanoon.org', 'indiacode.gov.in', 'tavily'
    cites: list = field(default_factory=list)
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    retrieved_via: str = "hybrid"   # 'hybrid' | 'citation-graph'
    provisions: set = field(default_factory=set)


@dataclass
class Passage:
    """A chunked excerpt from a document for precision context retrieval."""
    passage_id: str
    doc: Document
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0


@dataclass
class RetrievalEval:
    """Outcome of the local-corpus evaluation step."""
    is_confident: bool
    best_bm25: float
    best_dense: float
    term_coverage: float
    docs: list[Document]
    reason: str


@dataclass
class RAGResult:
    """The complete result produced by the RAG pipeline."""
    query: str
    retrieval_mode: Literal["LOCAL_CORPUS", "QUERY_CRAWL"]
    reason: str
    top_docs: list[Document]
    selected_passages: list[Passage]
    summary: str
    generation_method: str
    embedding_model: str = "none (BM25 only)"


def doc_key(doc: Document) -> str:
    return doc.doc_id or doc.url or doc.title


def number_sources(passages: list[Passage]) -> tuple[list[Document], dict[str, int]]:
    """Number the *unique documents* behind the passages, in order of first appearance."""
    docs: list[Document] = []
    index: dict[str, int] = {}
    for p in passages:
        k = doc_key(p.doc)
        if k not in index:
            index[k] = len(docs) + 1
            docs.append(p.doc)
    return docs, index


# ── Sentence embeddings (SBERT bi-encoder) ────────────────────────────────────

class Embedder:
    """
    Lazy wrapper around a sentence-transformers (SBERT) model.

    Why SBERT:  averaging BERT token vectors is a poor sentence
    representation, whereas a siamese/bi-encoder trained with classification, regression or
    triplet objectives yields vectors that can be compared directly with cosine similarity.
    Vectors are L2-normalised, so cosine == dot product.
    """

    def __init__(self, model_name: str = EMBED_MODEL, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._failed = False
        self._memo: dict[str, np.ndarray] = {}
        name = model_name.lower()
        if "e5" in name:
            self.q_prefix, self.p_prefix = "query: ", "passage: "
        elif "bge" in name and "m3" not in name:
            self.q_prefix, self.p_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self.q_prefix, self.p_prefix = "", ""

    @property
    def available(self) -> bool:
        if not self.enabled or self._failed:
            return False
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
            print(f"[Embedder] Loading sentence-embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            max_len = getattr(self._model, "max_seq_length", None) or 512
            self._model.max_seq_length = min(int(max_len), 512)
            return True
        except Exception as e:  # missing package, offline, bad model name ...
            print(f"[Embedder] Dense retrieval disabled ({e}). Falling back to BM25 only.")
            self._failed = True
            return False

    @property
    def label(self) -> str:
        return self.model_name if (self.enabled and not self._failed and self._model is not None) \
            else "none (BM25 only)"

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.q_prefix + t for t in texts])

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """No in-memory memo (used for the big corpus matrix, which has its own disk cache)."""
        return self._encode([self.p_prefix + t for t in texts], show_progress)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Memoised encoding for passages / sentences (repeat queries stay fast)."""
        keys = [hashlib.sha1((self.p_prefix + t).encode("utf-8")).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._memo]
        if missing:
            vecs = self._encode([self.p_prefix + texts[i] for i in missing])
            for i, v in zip(missing, vecs):
                self._memo[keys[i]] = v
        if not keys:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._memo[k] for k in keys])


# ── Step 1: Local Corpus Retriever (hybrid BM25 + SBERT + citation graph) ─────

class LocalRetriever:
    """
    Loads data/corpus_dedup.jsonl (from crawl.py + dedup.py) and builds:
      * a BM25 index                         (lexical channel)
      * a document-embedding matrix, cached  (semantic channel, incremental disk cache)
      * a provision index + citation graph   (light GraphRAG / knowledge-graph signals)
    """

    def __init__(self, corpus_path: Path | None = None, embedder: Embedder | None = None):
        if corpus_path:
            self.corpus_path = corpus_path
        elif LOCAL_CORPUS_DEDUP.exists():
            self.corpus_path = LOCAL_CORPUS_DEDUP
        elif LOCAL_CORPUS_RAW.exists():
            self.corpus_path = LOCAL_CORPUS_RAW
        else:
            self.corpus_path = None

        self.embedder = embedder or Embedder(enabled=False)
        self.docs: list[Document] = []
        self.bm25: BM25Okapi | None = None
        self.doc_emb: np.ndarray | None = None
        self.neighbors: dict[int, set[int]] = defaultdict(set)
        self._is_loaded = False

    def invalidate(self) -> None:
        """Force a reload (e.g. after a crawl appended documents)."""
        self._is_loaded = False
        self.doc_emb = None
        self.docs, self.bm25 = [], None
        self.neighbors = defaultdict(set)

    # ---- index construction -------------------------------------------------

    def load_index(self) -> bool:
        if self._is_loaded:
            return True
        if not self.corpus_path or not self.corpus_path.exists():
            print(f"[LocalRetriever] Warning: No local corpus found at {self.corpus_path}")
            return False

        print(f"[LocalRetriever] Loading local corpus from {self.corpus_path.name} ...")
        loaded_docs: list[Document] = []
        tokenized_corpus: list[list[str]] = []

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = data.get("title") or ""
                    text = data.get("text") or data.get("content") or ""
                    if not text and not title:
                        continue

                    head = title + " " + title + " " + text[:DOC_INDEX_CHARS]
                    cites = data.get("cites", [])
                    doc = Document(
                        doc_id=data.get("doc_id", ""),
                        title=title,
                        content=text,
                        url=data.get("url", ""),
                        court=data.get("court", ""),
                        date=data.get("date", ""),
                        doc_type=data.get("doc_type", "judgment"),
                        source="local_corpus",
                        cites=cites if isinstance(cites, list) else [],
                        provisions=extract_provisions(title + " " + text[:DOC_INDEX_CHARS]),
                    )
                    loaded_docs.append(doc)
                    tokenized_corpus.append(tokenize(head))

            if not loaded_docs:
                return False

            self.docs = loaded_docs
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._build_citation_graph()
            self._is_loaded = True
            print(f"[LocalRetriever] Indexed {len(self.docs):,} documents "
                  f"({sum(len(v) for v in self.neighbors.values()) // 2:,} citation links).")
            return True
        except Exception as e:
            print(f"[LocalRetriever] Error loading corpus: {e}")
            return False

    def _build_citation_graph(self) -> None:
        """Undirected in-corpus citation graph from the `cites` field (id / url / title)."""
        by_key: dict[str, int] = {}
        for i, d in enumerate(self.docs):
            for key in (d.doc_id, d.url, _norm_key(d.title) if d.title else ""):
                if key:
                    by_key.setdefault(key, i)
        for i, d in enumerate(self.docs):
            for c in d.cites or []:
                keys: list[str] = []
                if isinstance(c, str):
                    keys = [c, _norm_key(c)]
                elif isinstance(c, dict):
                    keys = [str(c[k]) for k in ("doc_id", "url", "title") if c.get(k)]
                    keys += [_norm_key(k) for k in keys]
                for k in keys:
                    j = by_key.get(k)
                    if j is not None and j != i:
                        self.neighbors[i].add(j)
                        self.neighbors[j].add(i)
                        break

    def _ensure_dense(self) -> bool:
        """Build (or load from the incremental disk cache) the document embedding matrix."""
        if self.doc_emb is not None:
            return True
        if not self.docs or not self.embedder.available:
            return False

        texts = [f"{d.title}. {d.content[:DOC_EMBED_CHARS]}" for d in self.docs]
        hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for t in texts]

        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedder.model_name)
        cache_file = EMBED_DIR / f"{safe_model}.npz"
        cache: dict[str, np.ndarray] = {}
        if cache_file.exists():
            try:
                z = np.load(cache_file, allow_pickle=False)
                cache = dict(zip(z["hashes"].tolist(), z["emb"]))
            except Exception as e:
                print(f"[LocalRetriever] Ignoring unreadable embedding cache ({e}).")

        missing = [i for i, h in enumerate(hashes) if h not in cache]
        if missing:
            print(f"[LocalRetriever] Embedding {len(missing):,} new document(s) "
                  f"(cached: {len(hashes) - len(missing):,}) ...")
            uniq: dict[str, int] = {}
            for i in missing:
                uniq.setdefault(hashes[i], i)
            order = list(uniq.items())
            vecs = self.embedder.encode_documents(
                [texts[i] for _, i in order], show_progress=len(order) > 64
            )
            for (h, _), v in zip(order, vecs):
                cache[h] = v
            try:
                tmp = cache_file.with_suffix(".tmp.npz")
                keys = list(cache.keys())
                np.savez(tmp, hashes=np.array(keys), emb=np.stack([cache[k] for k in keys]))
                tmp.replace(cache_file)
            except Exception as e:
                print(f"[LocalRetriever] Could not save embedding cache: {e}")

        self.doc_emb = np.stack([cache[h] for h in hashes]).astype(np.float32)
        return True

    # ---- search ---------------------------------------------------------------

    @staticmethod
    def _title_matches(title: str, sides: list[set[str]]) -> bool:
        words = set(re.findall(r"\w+", title.lower()))
        return all(len(s & words) / len(s) >= 0.6 for s in sides)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 12.0,
        min_coverage: float = 0.50,
        dense_threshold: float = 0.60,
        use_graph: bool = True,
        graph_extras: int = 2,
    ) -> RetrievalEval:
        """
        Hybrid retrieval + confidence evaluation.

        Confident if EITHER
          (a) lexical: best BM25 >= threshold AND term coverage >= min_coverage, OR
          (b) semantic: best cosine >= dense_threshold AND coverage >= 0.6 * min_coverage
              (the coverage guard stops "semantically near but legally different" topics
               from being treated as an answer).
        """
        if not self.load_index() or not self.bm25 or not self.docs:
            return RetrievalEval(False, 0.0, 0.0, 0.0, [], "Local corpus not available or empty")

        q_tokens = content_tokens(query)
        if not q_tokens:
            return RetrievalEval(False, 0.0, 0.0, 0.0, [], "Empty query")

        n = len(self.docs)
        bm = np.asarray(self.bm25.get_scores(q_tokens), dtype=float)

        dense: np.ndarray | None = None
        if self._ensure_dense():
            qv = self.embedder.encode_queries([query])[0]
            dense = self.doc_emb @ qv

        cand_n = min(n, max(top_k * 5, 25))
        fused: dict[int, float] = defaultdict(float)

        for rank, i in enumerate(np.argsort(-bm)[:cand_n]):
            if bm[i] > 0:
                fused[int(i)] += LEXICAL_WEIGHT / (RRF_K + rank + 1)
        if dense is not None:
            for rank, i in enumerate(np.argsort(-dense)[:cand_n]):
                fused[int(i)] += DENSE_WEIGHT / (RRF_K + rank + 1)

        if not fused:
            return RetrievalEval(False, 0.0, 0.0, 0.0, [], "No local documents matched the query terms")

        # Knowledge-graph style boosts: shared statutory provisions and named case parties.
        q_prov = extract_provisions(query)
        case_sides = parse_case_query(query)
        for i in list(fused):
            d = self.docs[i]
            if q_prov:
                fused[i] += min(PROVISION_BOOST_CAP, PROVISION_BOOST * len(q_prov & d.provisions))
            if case_sides and self._title_matches(d.title, case_sides):
                fused[i] += CASE_NAME_BOOST

        order = sorted(fused, key=fused.get, reverse=True)
        main = order[:top_k]

        # Local GraphRAG: pull in cited / citing neighbours of the best documents.
        extras: list[int] = []
        if use_graph and self.neighbors and graph_extras > 0:
            seen = set(main)
            for i in main[:3]:
                def rel(j: int) -> float:
                    return float(bm[j]) + (float(dense[j]) * 10 if dense is not None else 0.0)
                for j in sorted(self.neighbors.get(i, ()), key=rel, reverse=True):
                    if j in seen:
                        continue
                    relevant = bm[j] > 0 or (dense is not None and dense[j] >= GRAPH_MIN_DENSE)
                    if relevant:
                        extras.append(j)
                        seen.add(j)
                    if len(extras) >= graph_extras:
                        break
                if len(extras) >= graph_extras:
                    break

        def build(i: int, via: str) -> Document:
            return replace(
                self.docs[i],
                bm25_score=float(bm[i]),
                dense_score=float(dense[i]) if dense is not None else 0.0,
                fused_score=float(fused.get(i, 0.0)),
                retrieved_via=via,
            )

        top_docs = [build(i, "hybrid") for i in main] + [build(j, "citation-graph") for j in extras]

        # Confidence: term coverage over the best 3 fused documents.
        coverage = 0.0
        for i in main[:3]:
            d = self.docs[i]
            word_set = set(tokenize((d.title or "") + " " + d.content[:6000]))
            cov = sum(1 for w in q_tokens if w in word_set) / len(q_tokens)
            coverage = max(coverage, cov)

        best_bm = max((d.bm25_score for d in top_docs[:top_k]), default=0.0)
        best_dense = max((d.dense_score for d in top_docs[:top_k]), default=0.0)

        lexical_ok = best_bm >= threshold and coverage >= min_coverage
        semantic_ok = (dense is not None and best_dense >= dense_threshold
                       and coverage >= 0.6 * min_coverage)

        if lexical_ok or semantic_ok:
            how = "lexical" if lexical_ok else "semantic"
            reason = (
                f"Local corpus confident ({how}): BM25 {best_bm:.2f} (>= {threshold:.1f}), "
                f"cosine {best_dense:.2f} (>= {dense_threshold:.2f}), "
                f"term coverage {coverage*100:.0f}%."
            )
            return RetrievalEval(True, best_bm, best_dense, coverage, top_docs, reason)

        reason = (
            f"Local corpus insufficient: BM25 {best_bm:.2f} (need {threshold:.1f}), "
            f"cosine {best_dense:.2f} (need {dense_threshold:.2f}), "
            f"term coverage {coverage*100:.0f}% (need {min_coverage*100:.0f}%)."
        )
        return RetrievalEval(False, best_bm, best_dense, coverage, top_docs, reason)


# ── Step 2: Query-specific corpus crawl (Fallback) ────────────────────────────

class QueryCrawlRetriever:
    """Crawl Indian Kanoon for the query, then retrieve from the updated corpus."""

    def __init__(self, corpus_dir: Path = DATA_DIR, embedder: Embedder | None = None):
        self.corpus_dir = corpus_dir
        self.corpus_path = corpus_dir / "corpus.jsonl"
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5, use_graph: bool = True) -> list[Document]:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        existing_count = 0
        if self.corpus_path.exists():
            with self.corpus_path.open(encoding="utf-8") as corpus_file:
                existing_count = sum(1 for line in corpus_file if line.strip())

        print(f"\n[QueryCrawlRetriever] Crawling for query: '{query}'")
        await crawl.crawl(
            seeds=[query],
            max_docs=existing_count + 10,
            max_depth=0,
            pages_per_seed=1,
            out_dir=self.corpus_dir,
        )

        retriever = LocalRetriever(corpus_path=self.corpus_path, embedder=self.embedder)
        ev = retriever.search(
            query, top_k=top_k, threshold=0.0, min_coverage=0.0,
            dense_threshold=0.0, use_graph=use_graph,
        )
        return ev.docs


# ── Step 3: Syntax-based chunking, max-match & hybrid passage re-ranking ──────

HOLDING_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:held|hold|holds|holding)\s+that\b",
        r"\bwe\s+are\s+of\s+the\s+(?:view|opinion)\s+that\b",
        r"\bin\s+our\s+(?:view|opinion)\b",
        r"\b(?:observed|opined|concluded|ruled)\s+that\b",
        r"\bratio\s+decidendi\b",
        r"\bthe\s+(?:appeal|petition|writ\s+petition|suit)\s+(?:is|are|stands?)\s+(?:allowed|dismissed|disposed)",
        r"\bin\s+conclusion\b",
    )
]


def has_holding_language(text: str) -> bool:
    return any(p.search(text) for p in HOLDING_PATTERNS)


class PassageChunker:
    """
    Segments long judgments / statutes into passages using their *syntax*
    (numbered paragraphs -> sentences, legal-abbreviation aware) with sentence overlap,
    then scores every window against the query and keeps the best ones (max-match).
    """

    ABBREV = {
        "v", "vs", "no", "nos", "art", "sec", "s", "ss", "cl", "sch", "r", "o", "ors", "anr",
        "mr", "mrs", "ms", "dr", "shri", "smt", "hon", "j", "jj", "cj", "ltd", "pvt", "co",
        "inc", "corp", "dept", "govt", "st", "ch", "para", "paras", "ex", "viz", "etc", "cf",
        "i.e", "e.g", "ibid", "sd", "w.p", "crl", "cri", "misc", "am", "pm", "u/s", "r/w",
    }
    _PARA_SPLIT = re.compile(r"\n\s*\n|\n(?=\s*\(?\d{1,3}[.)]\s+[A-Z])")
    _SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["“(\[]?[A-Z0-9])')

    @classmethod
    def _ends_with_abbrev(cls, s: str) -> bool:
        last = s.split()[-1] if s.split() else ""
        if not last.endswith("."):
            return False
        core = last.rstrip(".").lower()
        return core in cls.ABBREV or (len(core) == 1 and core.isalpha())

    @classmethod
    def split_sentences(cls, text: str) -> list[str]:
        sentences: list[str] = []
        for para in cls._PARA_SPLIT.split(text):
            para = " ".join(para.split())
            if not para:
                continue
            out: list[str] = []
            for piece in cls._SENT_SPLIT.split(para):
                piece = piece.strip()
                if not piece:
                    continue
                if out and (cls._ends_with_abbrev(out[-1])
                            or re.fullmatch(r"\(?\d{1,3}[.)]", out[-1])):   # '12.' paragraph marker
                    out[-1] += " " + piece
                else:
                    out.append(piece)
            sentences.extend(out)
        return sentences

    @classmethod
    def chunk_document(
        cls, doc: Document, chunk_words: int = 220, overlap_sentences: int = 1
    ) -> list[Passage]:
        text = doc.content or ""
        sentences = cls.split_sentences(text)
        if not sentences:
            return []
        key = doc_key(doc)

        if len(text.split()) <= chunk_words:
            return [Passage(passage_id=f"{key}_p0", doc=doc, text=" ".join(sentences))]

        chunks: list[list[str]] = []
        cur: list[str] = []
        cur_words = 0
        for s in sentences:
            w = len(s.split())
            if w > chunk_words:                       # pathological run-on sentence: hard split
                if cur:
                    chunks.append(cur)
                    cur, cur_words = [], 0
                words = s.split()
                for i in range(0, len(words), chunk_words):
                    chunks.append([" ".join(words[i:i + chunk_words])])
                continue
            if cur and cur_words + w > chunk_words:
                chunks.append(cur)
                tail = cur[-overlap_sentences:] if overlap_sentences else []
                tail_words = sum(len(t.split()) for t in tail)
                cur = list(tail) if tail_words < chunk_words // 3 else []
                cur_words = sum(len(t.split()) for t in cur)
            cur.append(s)
            cur_words += w
        if cur:
            chunks.append(cur)

        passages: list[Passage] = []
        for chunk in chunks:
            body = " ".join(chunk)
            if len(body.split()) < 12 and len(chunks) > 1:   # stray headings / fragments
                continue
            passages.append(Passage(passage_id=f"{key}_p{len(passages)}", doc=doc, text=body))
        return passages

    @classmethod
    def extract_top_passages(
        cls,
        docs: list[Document],
        query: str,
        embedder: Embedder | None = None,
        top_k: int = 6,
        chunk_words: int = 220,
        max_per_doc: int = 2,
        max_dense_passages: int = 500,
    ) -> list[Passage]:
        """Chunk -> BM25 + SBERT scores -> RRF fusion -> provision/holding boosts -> diverse top-K."""
        all_p: list[Passage] = []
        for doc in docs:
            all_p.extend(cls.chunk_document(doc, chunk_words=chunk_words))

        q_tokens = content_tokens(query)
        if not all_p or not q_tokens:
            return []
        q_set = set(q_tokens)
        q_prov = extract_provisions(query)

        tok = [tokenize(p.text) for p in all_p]
        bm_scores = np.asarray(BM25Okapi(tok).get_scores(q_tokens), dtype=float)
        overlap = [len(q_set & set(t)) for t in tok]
        lex_order = sorted((i for i in range(len(all_p)) if overlap[i] > 0), key=lambda i: -bm_scores[i])

        # Dense channel (bounded cost: BM25 shortlist + lead chunk of each doc + strided remainder)
        dense_scores: dict[int, float] = {}
        p_emb: dict[int, np.ndarray] = {}
        if embedder is not None and embedder.available:
            if len(all_p) <= max_dense_passages:
                idxs = list(range(len(all_p)))
            else:
                chosen = set(lex_order[:max_dense_passages])
                first_seen: set[str] = set()
                for i, p in enumerate(all_p):          # lead chunk = headnote / operative preamble
                    k = doc_key(p.doc)
                    if k not in first_seen:
                        first_seen.add(k)
                        chosen.add(i)
                room = max_dense_passages - len(chosen)
                rest = [i for i in range(len(all_p)) if i not in chosen]
                if room > 0 and rest:
                    chosen |= set(rest[::max(1, len(rest) // room)][:room])
                idxs = sorted(chosen)
            texts = [f"{all_p[i].doc.title}. {all_p[i].text}" for i in idxs]   # contextual header
            embs = embedder.encode_passages(texts)
            qv = embedder.encode_queries([query])[0]
            sims = embs @ qv
            dense_scores = {i: float(s) for i, s in zip(idxs, sims)}
            p_emb = {i: e for i, e in zip(idxs, embs)}

        fused: dict[int, float] = defaultdict(float)
        for r, i in enumerate(lex_order):
            fused[i] += LEXICAL_WEIGHT / (RRF_K + r + 1)
        if dense_scores:
            qualified = [i for i in dense_scores if overlap[i] > 0 or dense_scores[i] >= PASSAGE_DENSE_FLOOR]
            for r, i in enumerate(sorted(qualified, key=lambda i: -dense_scores[i])):
                fused[i] += DENSE_WEIGHT / (RRF_K + r + 1)

        for i in list(fused):
            if q_prov:
                hit = len(q_prov & extract_provisions(all_p[i].text))
                fused[i] += min(PROVISION_BOOST_CAP, PROVISION_BOOST * hit)
            if has_holding_language(all_p[i].text):
                fused[i] += HOLDING_BOOST

        cap = max(max_per_doc, math.ceil(top_k / max(1, len(docs))))
        per_doc: Counter = Counter()
        selected: list[int] = []
        seen_snips: set[str] = set()

        def too_similar(i: int) -> bool:
            for j in selected:
                if i in p_emb and j in p_emb:
                    if float(p_emb[i] @ p_emb[j]) > 0.93:
                        return True
                else:
                    a, b = set(tok[i]), set(tok[j])
                    if a and b and len(a & b) / len(a | b) > 0.75:
                        return True
            return False

        for i in sorted(fused, key=fused.get, reverse=True):
            p = all_p[i]
            k = doc_key(p.doc)
            snip = p.text[:80]
            if per_doc[k] >= cap or snip in seen_snips or too_similar(i):
                continue
            seen_snips.add(snip)
            per_doc[k] += 1
            selected.append(i)
            if len(selected) >= top_k:
                break

        result: list[Passage] = []
        for i in selected:
            p = all_p[i]
            p.bm25_score = float(bm_scores[i])
            p.dense_score = float(dense_scores.get(i, 0.0))
            p.fused_score = float(fused[i])
            result.append(p)
        return result


# ── Step 4: RAG Synthesizer & Summarizer ───────────────────────────────────────

class RAGSynthesizer:
    """
    Synthesizes the retrieved legal passages into a grounded summary.
    Supports Gemini, Anthropic, OpenAI, Ollama, and a built-in offline extractive summarizer.
    """

    SYSTEM_PROMPT = (
        "You are an expert Indian Legal AI Assistant. Answer the user's query STRICTLY from the "
        "numbered excerpts of Indian court judgments and statutes supplied in the context.\n\n"
        "Rules:\n"
        "1. Direct Answer: begin with the holding / answer in 2-4 sentences.\n"
        "2. Provisions & Precedents: list the Articles, Sections, Acts and cases that appear in the "
        "excerpts, with the court and year when given. Note which court decided (Supreme Court > "
        "High Court) and flag any apparent conflict between authorities.\n"
        "3. Reasoning (ratio decidendi): summarize the legal tests and reasoning the court used.\n"
        "4. Strict grounding: use ONLY facts, citations and provisions present in the excerpts. "
        "Never invent cases, section numbers, dates or quotations. If the excerpts do not answer "
        "the query (or only partly), say exactly what is missing instead of guessing.\n"
        "5. Citations: every substantive statement must carry a bracketed source like [Doc 1] or "
        "[Doc 2]. Use only the Doc numbers provided.\n"
        "6. Keep quotations short (under 25 words). Close with one line noting this is an "
        "informational summary, not legal advice."
    )

    @staticmethod
    def build_context(passages: list[Passage]) -> tuple[str, int]:
        docs, index = number_sources(passages)
        blocks = []
        for n, d in enumerate(docs, 1):
            meta = f"Title: {d.title} | Court/Source: {d.court or d.source}"
            if d.date:
                meta += f" | Date: {d.date}"
            if d.url:
                meta += f" | URL: {d.url}"
            excerpts = [p for p in passages if index[doc_key(p.doc)] == n]
            body = "\n".join(
                f"  Excerpt {chr(96 + k)}: {p.text}" for k, p in enumerate(excerpts, 1)
            )
            blocks.append(f"[Doc {n}] {meta}\n{body}")
        return "\n\n---\n\n".join(blocks), len(docs)

    @staticmethod
    def check_citations(summary: str, n_docs: int) -> str:
        cited = {int(x) for x in re.findall(r"Doc\s*(\d+)", summary)}
        notes = []
        bad = sorted(c for c in cited if c < 1 or c > n_docs)
        if bad:
            notes.append(f"citation(s) to non-existent source(s) {bad} detected — ignore them")
        if not cited:
            notes.append("the answer contains no [Doc n] citations — verify against the sources below")
        if notes:
            summary += "\n\n> ⚠ Grounding check: " + "; ".join(notes) + "."
        return summary

    @classmethod
    async def generate_summary(
        cls,
        query: str,
        passages: list[Passage],
        retrieval_mode: str,
        offline: bool = False,
        embedder: Embedder | None = None,
    ) -> tuple[str, str]:
        """Returns: (summary_text, provider_name)"""
        context_text, n_docs = cls.build_context(passages)
        prompt = (
            f"User Legal Query: {query}\n\n"
            f"Retrieved Context ({n_docs} source document(s)):\n\n{context_text}\n\n"
            f"Write the legal summary now, following the rules."
        )

        if offline:
            print("[Synthesizer] Offline mode requested: using Built-in Extractive Legal Summarizer.")
            return (cls._extractive_summary(query, passages, retrieval_mode, embedder),
                    "Built-in Extractive Legal Summarizer (Offline)")

        providers = []
        if GEMINI_API_KEY:
            providers.append(("Google Gemini", cls._generate_gemini))
        if ANTHROPIC_API_KEY:
            providers.append(("Anthropic Claude", cls._generate_anthropic))
        if OPENAI_API_KEY:
            providers.append(("OpenAI", cls._generate_openai))
        if OLLAMA_HOST:
            providers.append((f"Ollama ({OLLAMA_MODEL})", cls._generate_ollama))

        for name, fn in providers:
            try:
                print(f"[Synthesizer] Using {name}.")
                summary = await fn(prompt)
                if summary:
                    return cls.check_citations(summary, n_docs), name
            except Exception as e:
                print(f"[Synthesizer] {name} failed: {e}. Falling back...")

        print("[Synthesizer] Using Built-in Extractive Legal Summarizer.")
        return (cls._extractive_summary(query, passages, retrieval_mode, embedder),
                "Built-in Extractive Legal Summarizer (Offline)")

    # ---- LLM providers --------------------------------------------------------

    @classmethod
    async def _generate_gemini(cls, prompt: str) -> str | None:
        payload = {
            "system_instruction": {"parts": [{"text": cls.SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1800},
        }
        last_error = None
        async with httpx.AsyncClient(timeout=45.0) as client:
            for model in GEMINI_MODELS:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                try:
                    resp = await client.post(
                        url, json=payload,
                        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                    )
                    if resp.status_code == 200:
                        candidates = resp.json().get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(
                                part.get("text", "") for part in parts
                                if isinstance(part.get("text"), str)
                            ).strip()
                            if text:
                                return text
                    elif resp.status_code == 404:
                        continue  # model not available on this tier, try next
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                except Exception as e:
                    last_error = str(e)
        if last_error:
            raise RuntimeError(last_error)
        raise RuntimeError("No supported Gemini models responded successfully.")

    @classmethod
    async def _generate_anthropic(cls, prompt: str) -> str | None:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": ANTHROPIC_MODEL,
            "system": cls.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1800,
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return "".join(c.get("text", "") for c in resp.json().get("content", [])).strip()

    @classmethod
    async def _generate_openai(cls, prompt: str) -> str | None:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1800,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()["choices"][0]["message"]["content"].strip()

    @classmethod
    async def _generate_ollama(cls, prompt: str) -> str | None:
        payload = {
            "model": OLLAMA_MODEL,
            "system": cls.SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{OLLAMA_HOST.rstrip('/')}/api/generate", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json().get("response", "").strip()

    # ---- offline extractive summarizer ---------------------------------------

    @classmethod
    def _extractive_summary(
        cls, query: str, passages: list[Passage], retrieval_mode: str,
        embedder: Embedder | None = None,
    ) -> str:
        """
        Query-focused extractive summarizer: sentences are ranked by query-term overlap,
        SBERT similarity (when available) and judicial-holding language — no topic-specific rules.
        """
        docs, index = number_sources(passages)
        q_set = set(content_tokens(query))

        lines = ["### Executive Legal Summary",
                 f"Based on retrieval from **{retrieval_mode}**, the following authorities and "
                 f"operative passages were identified for: *\"{query}\"*.", ""]

        statutes: set[str] = set()
        for p in passages:
            for m in PROVISION_DISPLAY_RE.findall(p.text):
                statutes.add(" ".join(m.split()))
        if statutes:
            lines.append("### Key Constitutional & Statutory Provisions Cited")
            lines.append("- " + ", ".join(f"`{s}`" for s in sorted(statutes)[:12]))
            lines.append("")

        cases = [d for d in docs if re.search(r"\bv(?:s|ersus)?\.?\s", d.title, re.IGNORECASE)]
        if cases:
            lines.append("### Primary Cases & Authorities")
            for d in cases[:5]:
                num = index[doc_key(d)]
                meta = ", ".join(x for x in (d.court, d.date) if x)
                lines.append(f"- **{d.title}**{f' ({meta})' if meta else ''} — [Doc {num}]")
            lines.append("")

        # Candidate sentences
        cands: list[tuple[str, int, int]] = []   # (sentence, doc_number, order)
        order = 0
        for p in passages:
            num = index[doc_key(p.doc)]
            for s in PassageChunker.split_sentences(p.text):
                if 60 <= len(s) <= 450:
                    cands.append((s, num, order))
                    order += 1

        lines.append("### Key Judicial Holdings & Operative Findings")
        picked: list[tuple[str, int, int]] = []
        if cands:
            sims = np.zeros(len(cands))
            if embedder is not None and embedder.available:
                cands = cands[:150]
                sims = np.asarray(embedder.encode_passages([c[0] for c in cands]) @
                                  embedder.encode_queries([query])[0])
            scored = []
            for k, (s, num, o) in enumerate(cands):
                toks = set(tokenize(s))
                score = len(q_set & toks) / max(1, len(q_set)) + float(sims[k] if k < len(sims) else 0.0)
                if has_holding_language(s):
                    score += 0.4
                scored.append((score, k))
            for _, k in sorted(scored, reverse=True):
                s, num, o = cands[k]
                st = set(tokenize(s))
                if any(len(st & set(tokenize(x[0]))) / max(1, len(st | set(tokenize(x[0])))) > 0.6 for x in picked):
                    continue
                picked.append(cands[k])
                if len(picked) >= 5:
                    break
            for s, num, _ in sorted(picked, key=lambda x: (x[1], x[2])):
                lines.append(f"> \"{s}\" — [Doc {num}]")
                lines.append("")
        if not picked:
            for p in passages[:3]:
                num = index[doc_key(p.doc)]
                preview = " ".join(p.text.split()[:55]) + "..."
                lines.append(f"**[Doc {num}] {p.doc.title}** ({p.doc.court or p.doc.source}):")
                lines.append(f"> \"{preview}\"")
                lines.append("")

        lines.append("---")
        lines.append("*(Note: this is an extractive summary. For full generative synthesis add "
                     "`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `OLLAMA_HOST` to `.env`. "
                     "Informational only — not legal advice.)*")
        return "\n".join(lines)


# ── Step 5: End-to-End RAG Pipeline ───────────────────────────────────────────

class RAGPipeline:
    """
    Query -> Hybrid local retrieval (BM25 + SBERT + citation graph)
          -> (confident ? local docs : query crawl)
          -> syntax-aware chunking -> hybrid passage re-ranking (max-match)
          -> grounded synthesis with citation check.
    """

    def __init__(
        self,
        local_threshold: float = 12.0,
        min_coverage: float = 0.50,
        dense_threshold: float = 0.60,
        embed_model: str = EMBED_MODEL,
        use_dense: bool = True,
        use_graph: bool = True,
    ):
        self.local_threshold = local_threshold
        self.min_coverage = min_coverage
        self.dense_threshold = dense_threshold
        self.use_graph = use_graph
        self.embedder = Embedder(embed_model, enabled=use_dense)
        self.local_retriever = LocalRetriever(embedder=self.embedder)
        self.crawl_retriever = QueryCrawlRetriever(embedder=self.embedder)

    async def run(
        self,
        query: str,
        force_web: bool = False,
        local_only: bool = False,
        offline: bool = False,
        top_k_docs: int = 5,
        top_k_passages: int = 6,
    ) -> RAGResult:
        query = query.strip()
        print(f"\n{'═'*72}")
        print(f"  LEGAL RAG PIPELINE")
        print(f"  Query : {query}")
        print(f"{'═'*72}\n")

        retrieval_mode: Literal["LOCAL_CORPUS", "QUERY_CRAWL"]
        selected_docs: list[Document] = []
        routing_reason = ""

        if force_web:
            retrieval_mode = "QUERY_CRAWL"
            routing_reason = "--force-web requested; running a query-specific crawl."
            print(f"[Routing] {routing_reason}")
            selected_docs = await self.crawl_retriever.retrieve(query, top_k=top_k_docs, use_graph=self.use_graph)
            self.local_retriever.invalidate()
        else:
            print("[Routing] Step 1: Evaluating local corpus (hybrid BM25 + semantic)...")
            ev = self.local_retriever.search(
                query,
                top_k=top_k_docs,
                threshold=self.local_threshold,
                min_coverage=self.min_coverage,
                dense_threshold=self.dense_threshold,
                use_graph=self.use_graph,
            )
            if ev.is_confident:
                retrieval_mode = "LOCAL_CORPUS"
                routing_reason = ev.reason
                selected_docs = ev.docs
                print(f"[Routing] ✓ HIT LOCAL DATA: {routing_reason}")
            else:
                print(f"[Routing] ✗ LOCAL MISS / INSUFFICIENT: {ev.reason}")
                if local_only:
                    retrieval_mode = "LOCAL_CORPUS"
                    routing_reason = f"{ev.reason} (--local-only specified; continuing with available local docs)."
                    selected_docs = ev.docs
                else:
                    retrieval_mode = "QUERY_CRAWL"
                    routing_reason = f"{ev.reason} -> Falling back to a query-specific crawl."
                    print(f"[Routing] → {routing_reason}")
                    selected_docs = await self.crawl_retriever.retrieve(
                        query, top_k=top_k_docs, use_graph=self.use_graph
                    )
                    self.local_retriever.invalidate()

        if not selected_docs:
            return RAGResult(
                query=query, retrieval_mode=retrieval_mode, reason=routing_reason,
                top_docs=[], selected_passages=[],
                summary="No relevant documents could be retrieved from the local corpus or query crawl.",
                generation_method="none", embedding_model=self.embedder.label,
            )

        print(f"\n[Retrieval] Retrieved {len(selected_docs)} candidate documents.")
        for i, d in enumerate(selected_docs, 1):
            print(f"  [{i}] ({d.retrieved_via}) BM25 {d.bm25_score:.2f} | cos {d.dense_score:.2f} "
                  f"| {d.title[:55]} | {d.url}")

        print(f"\n[Chunking] Segmenting documents and re-ranking top {top_k_passages} passages...")
        top_passages = PassageChunker.extract_top_passages(
            selected_docs, query=query, embedder=self.embedder, top_k=top_k_passages,
        )
        print(f"[Chunking] Extracted {len(top_passages)} high-relevance passages.")
        if not top_passages:
            return RAGResult(
                query=query, retrieval_mode=retrieval_mode, reason=routing_reason,
                top_docs=selected_docs, selected_passages=[],
                summary="Retrieved documents contained no passages matching the query; no legal analysis was generated.",
                generation_method="none", embedding_model=self.embedder.label,
            )

        print(f"\n[Synthesis] Generating legal answer and summary...")
        summary, method = await RAGSynthesizer.generate_summary(
            query=query, passages=top_passages, retrieval_mode=retrieval_mode,
            offline=offline, embedder=self.embedder,
        )

        return RAGResult(
            query=query, retrieval_mode=retrieval_mode, reason=routing_reason,
            top_docs=selected_docs, selected_passages=top_passages,
            summary=summary, generation_method=method, embedding_model=self.embedder.label,
        )


# ── Pretty Output Printer ─────────────────────────────────────────────────────

def print_rag_result(result: RAGResult) -> None:
    SEP = "─" * 74
    print(f"\n{'═'*74}")
    print(f"  RAG SYNTHESIZED LEGAL REPORT")
    print(f"{'═'*74}")
    print(f"  Query          : {result.query}")
    print(f"  Retrieval Mode : [{result.retrieval_mode}]")
    print(f"  Embeddings     : {result.embedding_model}")
    print(f"  Routing Reason : {result.reason}")
    print(f"  Synthesizer    : {result.generation_method}")
    print(SEP)

    print("\n" + result.summary + "\n")

    print(SEP)
    print("  SOURCES & AUTHORITIES RETRIEVED  (numbers match [Doc n] citations):")
    docs, index = number_sources(result.selected_passages)
    for n, doc in enumerate(docs, 1):
        court_info = f" ({doc.court})" if doc.court else ""
        mine = [p for p in result.selected_passages if index[doc_key(p.doc)] == n]
        print(f"  [Doc {n}] {doc.title}{court_info}")
        print(f"      URL    : {doc.url or 'N/A'}")
        print(f"      Source : {doc.source} ({doc.retrieved_via})  |  {len(mine)} passage(s)")
        for k, p in enumerate(mine, 1):
            print(f"      · passage {chr(96 + k)}: BM25 {p.bm25_score:.2f} | cos {p.dense_score:.2f} "
                  f"| fused {p.fused_score:.4f}")
    print(SEP + "\n")


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Local-First Hybrid (BM25 + SBERT) Indian Legal RAG with Query Crawl Fallback"
    )
    parser.add_argument("query", nargs="*", help="Legal question or query")
    parser.add_argument("--force-web", action="store_true", help="Bypass local corpus and run a query-specific crawl")
    parser.add_argument("--local-only", action="store_true", help="Disable crawling; only search local corpus")
    parser.add_argument("--offline", action="store_true", help="Use built-in extractive legal summarizer without LLM API calls")
    parser.add_argument("--no-dense", action="store_true", help="Disable SBERT embeddings (BM25 only)")
    parser.add_argument("--no-graph", action="store_true", help="Disable citation-graph neighbour expansion")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help=f"sentence-transformers model (default: {EMBED_MODEL})")
    parser.add_argument("--threshold", type=float, default=12.0, help="BM25 confidence threshold for local corpus (default: 12.0)")
    parser.add_argument("--dense-threshold", type=float, default=0.60, help="Cosine confidence threshold (default: 0.60; tune per model)")
    parser.add_argument("--min-coverage", type=float, default=0.50, help="Min term coverage fraction (default: 0.50)")
    parser.add_argument("--top-docs", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--top-passages", type=int, default=6, help="Number of passages for summarizer (default: 6)")

    args = parser.parse_args()

    pipeline = RAGPipeline(
        local_threshold=args.threshold,
        min_coverage=args.min_coverage,
        dense_threshold=args.dense_threshold,
        embed_model=args.embed_model,
        use_dense=not args.no_dense,
        use_graph=not args.no_graph,
    )

    def answer(q: str) -> None:
        result = asyncio.run(
            pipeline.run(
                query=q,
                force_web=args.force_web,
                local_only=args.local_only,
                offline=args.offline,
                top_k_docs=args.top_docs,
                top_k_passages=args.top_passages,
            )
        )
        print_rag_result(result)

    query_str = " ".join(args.query).strip()
    if query_str:
        answer(query_str)
        return

    # Interactive mode: keep models / indexes warm between questions.
    while True:
        try:
            q = input("\nEnter your legal query (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q)


if __name__ == "__main__":
    main()
