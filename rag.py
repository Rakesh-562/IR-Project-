"""
rag.py — Local-First Retrieval-Augmented Generation (RAG) for Indian Legal Documents.

Architecture:
1. Local Retrieval:
   - Indexes deduplicated corpus (data/corpus_dedup.jsonl from crawl.py + dedup.py).
   - Fast BM25 scoring over documents + keyword term coverage check.
2. Intelligent Routing Decision:
   - If local corpus has confident matches (score >= threshold and term coverage >= min_coverage):
       -> Uses local corpus data.
   - Else (or if --force-web is passed):
       -> Runs a query-specific crawl with crawl.py and searches the updated corpus.
3. Passage Chunking & Re-ranking:
   - Chunks lengthy legal texts (judgments/statutes) into focused passages (~250-350 words).
   - BM25 re-ranks passages to isolate the exact operative rulings, sections, and reasoning.
4. Summarization / Generation:
   - Supports Gemini, Anthropic, OpenAI, or Ollama if configured in .env.
   - Includes a built-in offline Extractive Legal Summarizer when no LLM API key is set,
     ensuring the pipeline ALWAYS produces a complete, structured summary with citations.

Usage:
    python rag.py "right to privacy Article 21 Puttaswamy"
    python rag.py "digital personal data protection bill exemptions" --force-web
    python rag.py                                           # Interactive mode
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

import crawl

# ── Load environment variables ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOCAL_CORPUS_DEDUP = DATA_DIR / "corpus_dedup.jsonl"
LOCAL_CORPUS_RAW   = DATA_DIR / "corpus.jsonl"
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "")  # e.g. http://localhost:11434
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


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
    "tell", "explain", "summarize", "find", "search", "give", "regarding", "versus", "vs"
}

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
    cites: list[str] = field(default_factory=list)
    bm25_score: float = 0.0


@dataclass
class Passage:
    """A chunked excerpt from a document for precision context retrieval."""
    passage_id: str
    doc: Document
    text: str
    bm25_score: float = 0.0


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


# ── Step 1: Local Corpus Retriever ───────────────────────────────────────────

class LocalRetriever:
    """
    Loads and indexes data/corpus_dedup.jsonl (from crawl.py + dedup.py).
    Caches documents and BM25 index in memory for fast querying.
    """
    def __init__(self, corpus_path: Path | None = None):
        if corpus_path:
            self.corpus_path = corpus_path
        elif LOCAL_CORPUS_DEDUP.exists():
            self.corpus_path = LOCAL_CORPUS_DEDUP
        elif LOCAL_CORPUS_RAW.exists():
            self.corpus_path = LOCAL_CORPUS_RAW
        else:
            self.corpus_path = None

        self.docs: list[Document] = []
        self.bm25: BM25Okapi | None = None
        self._is_loaded = False

    def load_index(self) -> bool:
        """Load documents and build BM25 index."""
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

                    doc = Document(
                        doc_id   = data.get("doc_id", ""),
                        title    = title,
                        content  = text,
                        url      = data.get("url", ""),
                        court    = data.get("court", ""),
                        date     = data.get("date", ""),
                        doc_type = data.get("doc_type", "judgment"),
                        source   = "local_corpus",
                        cites    = data.get("cites", [])
                    )
                    loaded_docs.append(doc)
                    # Index title + first 4000 characters for high-speed doc-level candidate retrieval
                    doc_preview = tokenize(title + " " + title + " " + text[:4000])
                    tokenized_corpus.append(doc_preview)

            if not loaded_docs:
                return False

            self.docs = loaded_docs
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._is_loaded = True
            print(f"[LocalRetriever] Indexed {len(self.docs):,} documents successfully.")
            return True
        except Exception as e:
            print(f"[LocalRetriever] Error loading corpus: {e}")
            return False

    def evaluate_and_retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 12.0,
        min_coverage: float = 0.50
    ) -> tuple[bool, float, float, list[Document], str]:
        """
        Evaluate whether the query has a confident hit in the local corpus.
        Returns:
            (is_confident, best_score, term_coverage, top_documents, decision_reason)
        """
        if not self.load_index() or not self.bm25 or not self.docs:
            return False, 0.0, 0.0, [], "Local corpus not available or empty"

        q_tokens = tokenize(query)
        if not q_tokens:
            return False, 0.0, 0.0, [], "Empty query"

        query_words = [
            word for word in q_tokens
            if len(word) > 2 and word not in STOPWORDS
        ]
        scores = self.bm25.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        if not top_indices:
            return False, 0.0, 0.0, [], "No candidates found"

        best_score = float(scores[top_indices[0]])
        if best_score <= 0:
            query_terms = set(query_words or q_tokens)
            matching_indices = [
                index for index, doc in enumerate(self.docs)
                if query_terms.intersection(tokenize(f"{doc.title} {doc.content[:4000]}"))
            ]
            if not matching_indices:
                return False, best_score, 0.0, [], "No local documents matched the query terms"
            top_indices = sorted(
                matching_indices,
                key=lambda index: scores[index],
                reverse=True,
            )[:top_k]
            best_score = float(scores[top_indices[0]])

        # Check coverage in the top candidate
        top_doc = self.docs[top_indices[0]]
        searchable_text = ((top_doc.title or "") + " " + (top_doc.content[:6000] or "")).lower()
        doc_word_set = set(re.findall(r"\w+", searchable_text))

        if query_words:
            matched_words = [w for w in query_words if w in doc_word_set]
            term_coverage = len(matched_words) / len(query_words)
        else:
            term_coverage = 1.0

        # Construct top documents list with updated scores
        top_docs: list[Document] = []
        for idx in top_indices:
            d = self.docs[idx]
            d.bm25_score = float(scores[idx])
            top_docs.append(d)

        # Confidence Decision
        if best_score >= threshold and term_coverage >= min_coverage:
            reason = (
                f"Local corpus confident match: top BM25 score {best_score:.2f} "
                f"(>= threshold {threshold:.1f}) with {term_coverage*100:.0f}% query term coverage."
            )
            return True, best_score, term_coverage, top_docs, reason
        else:
            reason = (
                f"Local corpus insufficient: top BM25 score {best_score:.2f} (threshold {threshold:.1f}) "
                f"or term coverage {term_coverage*100:.0f}% (< {min_coverage*100:.0f}%)."
            )
            return False, best_score, term_coverage, top_docs, reason


# ── Step 2: Query-specific corpus crawl (Fallback) ────────────────────────────

class QueryCrawlRetriever:
    """Crawl Indian Kanoon for the query, then retrieve from the updated corpus."""

    def __init__(self, corpus_dir: Path = DATA_DIR):
        self.corpus_dir = corpus_dir
        self.corpus_path = corpus_dir / "corpus.jsonl"

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
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

        retriever = LocalRetriever(corpus_path=self.corpus_path)
        _, _, _, documents, _ = retriever.evaluate_and_retrieve(
            query,
            top_k=top_k,
            threshold=0.0,
            min_coverage=0.0,
        )
        return documents


# ── Step 3: Passage Chunking & Re-ranking ─────────────────────────────────────

class PassageChunker:
    """
    Segments long judgments or legislative acts into overlapping passages,
    then uses BM25 to surface the most relevant excerpts.
    """
    @staticmethod
    def chunk_document(doc: Document, chunk_words: int = 300, overlap: int = 50) -> list[Passage]:
        words = doc.content.split()
        if not words:
            return []
        if len(words) <= chunk_words:
            return [Passage(passage_id=f"{doc.doc_id}_p0", doc=doc, text=doc.content)]

        passages: list[Passage] = []
        step = max(50, chunk_words - overlap)
        p_idx = 0
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_words])
            passages.append(
                Passage(passage_id=f"{doc.doc_id}_p{p_idx}", doc=doc, text=chunk)
            )
            p_idx += 1
            if i + chunk_words >= len(words):
                break
        return passages

    @classmethod
    def extract_top_passages(
        cls,
        docs: list[Document],
        query: str,
        top_k: int = 6,
        chunk_words: int = 300,
        overlap: int = 50
    ) -> list[Passage]:
        """Chunk all retrieved documents and return the top-K highest scoring passages."""
        all_passages: list[Passage] = []
        for doc in docs:
            all_passages.extend(cls.chunk_document(doc, chunk_words=chunk_words, overlap=overlap))

        query_tokens = set(tokenize(query))
        if not all_passages or not query_tokens:
            return []

        # Tokenize passages for BM25 re-ranking
        tokenized_passages = [tokenize(p.text) for p in all_passages]
        bm25 = BM25Okapi(tokenized_passages)
        scores = bm25.get_scores(tokenize(query))

        ranked_pairs = sorted(zip(all_passages, scores), key=lambda x: x[1], reverse=True)
        top_passages: list[Passage] = []
        seen_texts: set[str] = set()

        for p, score in ranked_pairs:
            if not query_tokens.intersection(tokenize(p.text)):
                continue
            # Dedup passages that are identical
            snip = p.text[:80]
            if snip in seen_texts:
                continue
            seen_texts.add(snip)
            p.bm25_score = float(score)
            top_passages.append(p)
            if len(top_passages) >= top_k:
                break

        return top_passages


# ── Step 4: RAG Synthesizer & Summarizer ───────────────────────────────────────

class RAGSynthesizer:
    """
    Synthesizes the retrieved legal passages into a coherent summary.
    Supports:
        1. Google Gemini (via GEMINI_API_KEY / GOOGLE_API_KEY)
        2. Anthropic Claude (via ANTHROPIC_API_KEY)
        3. OpenAI (via OPENAI_API_KEY)
        4. Ollama local LLM (via OLLAMA_HOST)
        5. Built-in Extractive Legal Summarizer (offline fallback if no API key is provided)
    """

    SYSTEM_PROMPT = (
        "You are an expert Indian Legal AI Assistant. Your task is to provide an accurate, "
        "well-structured legal summary and analysis answering the user's query based STRICTLY on the "
        "provided excerpts from Indian court judgments and statutes.\n\n"
        "Guidelines:\n"
        "1. Executive Summary: State the direct legal answer / holding upfront.\n"
        "2. Key Legal Provisions & Precedents: Mention specific Articles of the Constitution, statutory "
        "Sections, and landmark cases cited in the excerpts.\n"
        "3. Judicial Reasoning / Ratio Decidendi: Summarize the court's core arguments and legal tests.\n"
        "4. Strict Grounding: Do not invent facts or cite cases not present in the excerpts.\n"
        "5. Citations: Cite sources using bracketed references like [Doc 1], [Doc 2]."
    )

    @classmethod
    async def generate_summary(
        cls,
        query: str,
        passages: list[Passage],
        retrieval_mode: str,
        offline: bool = False
    ) -> tuple[str, str]:
        """
        Generates summary using the best available provider.
        Returns: (summary_text, provider_name)
        """
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            source_info = f"Title: {p.doc.title} | Court/Source: {p.doc.court or p.doc.source} | URL: {p.doc.url}"
            context_blocks.append(f"[Doc {i}] {source_info}\nExcerpt:\n{p.text}\n")
        context_text = "\n---\n".join(context_blocks)

        prompt = (
            f"User Legal Query: {query}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            f"Please synthesize the above excerpts into a comprehensive legal summary answering the query."
        )

        if offline:
            print("[Synthesizer] Offline mode requested: using Built-in Extractive Legal Summarizer.")
            summary = cls._extractive_summary(query, passages, retrieval_mode)
            return summary, "Built-in Extractive Legal Summarizer (Offline)"

        # 1. Try Gemini
        if GEMINI_API_KEY:
            try:
                print("[Synthesizer] Using Google Gemini.")
                summary = await cls._generate_gemini(prompt)
                if summary:
                    return summary, "Google Gemini"
            except Exception as e:
                print(f"[Synthesizer] Google Gemini generation failed: {e}. Falling back...")

        # 2. Try Anthropic
        if ANTHROPIC_API_KEY:
            try:
                summary = await cls._generate_anthropic(prompt)
                if summary:
                    return summary, "Anthropic Claude"
            except Exception as e:
                print(f"[Synthesizer] Anthropic API error: {e}")

        # 3. Try OpenAI
        if OPENAI_API_KEY:
            try:
                summary = await cls._generate_openai(prompt)
                if summary:
                    return summary, "OpenAI"
            except Exception as e:
                print(f"[Synthesizer] OpenAI API error: {e}")

        # 4. Try Ollama (Local LLM)
        if OLLAMA_HOST:
            try:
                summary = await cls._generate_ollama(prompt)
                if summary:
                    return summary, f"Ollama ({OLLAMA_MODEL})"
            except Exception as e:
                print(f"[Synthesizer] Ollama error: {e}")

        # 5. Offline Extractive Legal Summarizer (Zero API Keys Required)
        print("[Synthesizer] Using Built-in Extractive Legal Summarizer.")
        summary = cls._extractive_summary(query, passages, retrieval_mode)
        return summary, "Built-in Extractive Legal Summarizer (Offline)"

    @classmethod
    async def _generate_gemini(cls, prompt: str) -> str | None:
        candidate_models = [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-3.8-flash",
        ]
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{cls.SYSTEM_PROMPT}\n\n{prompt}"}]}
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500}
        }

        last_error = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            for model in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
                try:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(
                                part.get("text", "")
                                for part in parts
                                if isinstance(part.get("text"), str)
                            ).strip()
                            if text:
                                return text
                    elif resp.status_code == 404:
                        # Model not available in this tier, try next candidate
                        continue
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                except Exception as e:
                    last_error = str(e)
                    continue

        if last_error:
            raise RuntimeError(last_error)
        raise RuntimeError("No supported Gemini models responded successfully.")

    @classmethod
    async def _generate_anthropic(cls, prompt: str) -> str | None:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-haiku-4-5",
            "system": cls.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return "".join(c.get("text", "") for c in data.get("content", [])).strip()
        return None

    @classmethod
    async def _generate_openai(cls, prompt: str) -> str | None:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1500
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    @classmethod
    async def _generate_ollama(cls, prompt: str) -> str | None:
        url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{cls.SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False
        }
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        return None

    @classmethod
    def _extractive_summary(cls, query: str, passages: list[Passage], retrieval_mode: str) -> str:
        """
        High-precision rule-based legal extractive summarizer.
        Extracts legal holdings, statutory articles/sections, and key judicial observations.
        """
        lines = []
        lines.append("### Executive Legal Summary")
        lines.append(
            f"Based on retrieval from **{retrieval_mode}**, relevant Indian legal authorities, "
            f"statutory provisions, and judicial precedents were identified for: *\"{query}\"*."
        )
        lines.append("")

        # Extract landmark mentions & key statutory sections
        statutes = set()
        cases = set()
        for p in passages:
            text = p.text
            # Identify articles & sections
            for match in re.findall(r"(?:Article\s+\d+(?:\([0-9a-zA-Z]+\))*|Section\s+\d+[A-Z]*(?:\([0-9a-zA-Z]+\))*)", text, re.IGNORECASE):
                statutes.add(match)
            if "vs" in p.doc.title or "Vs" in p.doc.title or "v." in p.doc.title:
                cases.add(p.doc.title)

        if statutes:
            lines.append("### Key Constitutional & Statutory Provisions Cited")
            sorted_statutes = sorted(list(statutes))[:10]
            lines.append("- " + ", ".join(f"`{s}`" for s in sorted_statutes))
            lines.append("")

        if cases:
            lines.append("### Primary Landmark Cases & Authorities")
            for c in list(cases)[:4]:
                lines.append(f"- **{c}**")
            lines.append("")

        lines.append("### Key Judicial Holdings & Operative Findings")
        added_points = 0
        holding_patterns = [
            r"held\s+that[^.]*\.",
            r"we\s+are\s+of\s+the\s+view\s+that[^.]*\.",
            r"court\s+observed\s+that[^.]*\.",
            r"fundamental\s+right[^.]*\.",
            r"proportionality[^.]*\.",
            r"the\s+principle\s+of[^.]*\.",
            r"under\s+article\s+21[^.]*\.",
            r"right\s+to\s+privacy[^.]*\."
        ]

        for passage_number, p in enumerate(passages, 1):
            sentences = re.split(r"(?<=[.!?])\s+", p.text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 60 or len(s_clean) > 350:
                    continue
                for pat in holding_patterns:
                    if re.search(pat, s_clean, re.IGNORECASE):
                        lines.append(f"> \"{s_clean}\" — [Doc {passage_number}]")
                        lines.append("")
                        added_points += 1
                        break
                if added_points >= 4:
                    break
            if added_points >= 4:
                break

        # Fallback if no specific holding pattern triggered
        if added_points == 0:
            for i, p in enumerate(passages[:3], 1):
                preview = " ".join(p.text.split()[:55]) + "..."
                lines.append(f"**[Doc {i}] {p.doc.title}** ({p.doc.court or p.doc.source}):")
                lines.append(f"> \"{preview}\"")
                lines.append("")

        lines.append("---")
        lines.append("*(Note: To enable full generative AI synthesis, add `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` to your `.env` file.)*")

        return "\n".join(lines)


# ── Step 5: End-to-End RAG Pipeline ───────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the complete Legal RAG pipeline:
      Query -> Local Check -> (Local Retrieval OR Query Crawl) -> Chunking -> Synthesis
    """
    def __init__(
        self,
        local_threshold: float = 12.0,
        min_coverage: float = 0.50,
    ):
        self.local_threshold = local_threshold
        self.min_coverage = min_coverage
        self.local_retriever = LocalRetriever()
        self.crawl_retriever = QueryCrawlRetriever()

    async def run(
        self,
        query: str,
        force_web: bool = False,
        local_only: bool = False,
        offline: bool = False,
        top_k_docs: int = 5,
        top_k_passages: int = 6
    ) -> RAGResult:
        query = query.strip()
        print(f"\n{'═'*72}")
        print(f"  LEGAL RAG PIPELINE")
        print(f"  Query : {query}")
        print(f"{'═'*72}\n")

        retrieval_mode: Literal["LOCAL_CORPUS", "QUERY_CRAWL"]
        selected_docs: list[Document] = []
        routing_reason = ""

        # Routing decision
        if force_web:
            retrieval_mode = "QUERY_CRAWL"
            routing_reason = "--force-web requested; running a query-specific crawl."
            print(f"[Routing] {routing_reason}")
            selected_docs = await self.crawl_retriever.retrieve(query, top_k=top_k_docs)

        else:
            print(f"[Routing] Step 1: Evaluating local deduplicated corpus...")
            is_confident, _, _, local_docs, reason = (
                self.local_retriever.evaluate_and_retrieve(
                    query,
                    top_k=top_k_docs,
                    threshold=self.local_threshold,
                    min_coverage=self.min_coverage
                )
            )

            if is_confident:
                retrieval_mode = "LOCAL_CORPUS"
                routing_reason = reason
                selected_docs = local_docs
                print(f"[Routing] ✓ HIT LOCAL DATA: {routing_reason}")
            else:
                print(f"[Routing] ✗ LOCAL MISS / INSUFFICIENT: {reason}")
                if local_only:
                    retrieval_mode = "LOCAL_CORPUS"
                    routing_reason = f"{reason} (--local-only specified; continuing with available local docs)."
                    selected_docs = local_docs
                else:
                    retrieval_mode = "QUERY_CRAWL"
                    routing_reason = f"{reason} -> Falling back to a query-specific crawl."
                    print(f"[Routing] → {routing_reason}")
                    selected_docs = await self.crawl_retriever.retrieve(query, top_k=top_k_docs)

        if not selected_docs:
            return RAGResult(
                query=query,
                retrieval_mode=retrieval_mode,
                reason=routing_reason,
                top_docs=[],
                selected_passages=[],
                summary="No relevant documents could be retrieved from the local corpus or query crawl.",
                generation_method="none"
            )

        print(f"\n[Retrieval] Retrieved {len(selected_docs)} candidate documents.")
        for i, d in enumerate(selected_docs, 1):
            print(f"  [{i}] ({d.source}) BM25: {d.bm25_score:.2f} | {d.title[:60]} | {d.url}")

        # Chunk and Re-rank passages
        print(f"\n[Chunking] Segmenting documents and re-ranking top {top_k_passages} passages...")
        top_passages = PassageChunker.extract_top_passages(
            selected_docs,
            query=query,
            top_k=top_k_passages
        )
        print(f"[Chunking] Extracted {len(top_passages)} high-relevance passages.")
        if not top_passages:
            return RAGResult(
                query=query,
                retrieval_mode=retrieval_mode,
                reason=routing_reason,
                top_docs=selected_docs,
                selected_passages=[],
                summary="Retrieved documents contained no passages matching the query terms; no legal analysis was generated.",
                generation_method="none",
            )

        # Synthesize Summary
        print(f"\n[Synthesis] Generating legal answer and summary...")
        summary, method = await RAGSynthesizer.generate_summary(
            query=query,
            passages=top_passages,
            retrieval_mode=retrieval_mode,
            offline=offline
        )

        return RAGResult(
            query=query,
            retrieval_mode=retrieval_mode,
            reason=routing_reason,
            top_docs=selected_docs,
            selected_passages=top_passages,
            summary=summary,
            generation_method=method
        )


# ── Pretty Output Printer ─────────────────────────────────────────────────────

def print_rag_result(result: RAGResult) -> None:
    SEP = "─" * 74
    print(f"\n{'═'*74}")
    print(f"  RAG SYNTHESIZED LEGAL REPORT")
    print(f"{'═'*74}")
    print(f"  Query          : {result.query}")
    print(f"  Retrieval Mode : [{result.retrieval_mode}]")
    print(f"  Routing Reason : {result.reason}")
    print(f"  Synthesizer    : {result.generation_method}")
    print(SEP)

    print("\n" + result.summary + "\n")

    print(SEP)
    print("  SOURCES & AUTHORITIES RETRIEVED:")
    for i, passage in enumerate(result.selected_passages, 1):
        doc = passage.doc
        court_info = f" ({doc.court})" if doc.court else ""
        print(f"  [Doc {i}] {doc.title}{court_info}")
        print(f"      URL    : {doc.url or 'N/A'}")
        print(f"      Source : {doc.source}  |  Passage BM25: {passage.bm25_score:.2f}")
    print(SEP + "\n")


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Local-First Indian Legal RAG System with Query Crawl Fallback"
    )
    parser.add_argument("query", nargs="*", help="Legal question or query")
    parser.add_argument("--force-web", action="store_true", help="Bypass local corpus and run a query-specific crawl")
    parser.add_argument("--local-only", action="store_true", help="Disable crawling; only search local corpus")
    parser.add_argument("--offline", action="store_true", help="Use built-in extractive legal summarizer without LLM API calls")
    parser.add_argument("--threshold", type=float, default=12.0, help="BM25 confidence threshold for local corpus (default: 12.0)")
    parser.add_argument("--min-coverage", type=float, default=0.50, help="Min term coverage fraction (default: 0.50)")
    parser.add_argument("--top-docs", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--top-passages", type=int, default=6, help="Number of passages for summarizer (default: 6)")

    args = parser.parse_args()

    query_str = " ".join(args.query).strip()
    if not query_str:
        query_str = input("Enter your legal query: ").strip()
    if not query_str:
        sys.exit("No query provided.")

    pipeline = RAGPipeline(
        local_threshold=args.threshold,
        min_coverage=args.min_coverage
    )

    result = asyncio.run(
        pipeline.run(
            query=query_str,
            force_web=args.force_web,
            local_only=args.local_only,
            offline=args.offline,
            top_k_docs=args.top_docs,
            top_k_passages=args.top_passages
        )
    )

    print_rag_result(result)


if __name__ == "__main__":
    main()
