# RAG System for Indian Legal Documents

This project implements a local-first, hybrid Retrieval-Augmented Generation
(RAG) system for Indian legal judgments and statutes. The implementation is in
[rag.py](rag.py). It can find relevant legal authorities, select the most useful
passages, and generate a grounded summary with document citations.

The dashboard integration is implemented in [search_app.py](search_app.py). Select
**include RAG summary** beside a search query to show the RAG-generated panel
above the normal keyword search results.

## End-to-end flow

```text
User question
  -> legal-aware query tokenization
  -> local hybrid document retrieval
  -> confidence evaluation
  -> optional live-crawl fallback
  -> syntax-aware passage chunking
  -> BM25 + SBERT passage re-ranking
  -> grounded synthesis
  -> citation validation
  -> legal summary and source passages
```

The main entry point is `RAGPipeline.run()`:

```text
Query
  -> Hybrid local retrieval
  -> Local corpus confident?
     yes -> use local documents
     no  -> query-specific crawl
  -> Chunk selected documents
  -> Re-rank passages
  -> Generate or extract an answer
```

## 1. Document and passage representation

A document represents one judgment or statute. It contains the title, full text,
court, date, URL, citation list, and detected statutory provisions.

```python
Document(
    doc_id="iks-123",
    title="Justice K.S. Puttaswamy v. Union of India",
    content="Full judgment text...",
    court="Supreme Court",
    date="2017",
    provisions={"art_21"},
)
```

A passage is a smaller section selected from a document:

```python
Passage(
    text="The right to privacy is protected as a fundamental right...",
    bm25_score=18.4,
    dense_score=0.82,
)
```

Passages limit the text sent to a language model and help retrieval operate on
the exact holding or reasoning within a long judgment.

## 2. Legal-aware tokenization

The tokenizer does more than split ordinary words. It converts legal
references into canonical tokens.

| Text | Canonical token |
|---|---|
| Article 21 | `art_21` |
| Section 302 | `sec_302` |
| Order 39 Rule 1 | `ord_39`, `rule_1` |
| Schedule VII | `sch_vii` |

Examples:

```text
right to privacy Article 21 Puttaswamy
```

is represented using meaningful tokens similar to:

```text
right
privacy
art_21
puttaswamy
```

Common words such as `the`, `is`, `of`, `and`, `for`, and `versus` are removed
from the meaningful query-token set. Named case queries are also recognized:

```text
Kesavananda Bharati v. State of Kerala
```

becomes the two case-name groups `kesavananda/bharati` and `state/kerala`.

## 3. Local corpus loading

`LocalRetriever` prefers `data/corpus_dedup.jsonl`. If that file does not exist,
it uses `data/corpus.jsonl`.

For each JSONL record it creates a `Document`, extracts provisions, and builds:

1. A BM25 lexical index.
2. A dense SBERT document-embedding matrix.
3. A statutory-provision index.
4. An in-corpus citation graph.

Corpus loading is lazy: it happens when the first local search is performed.

## 4. BM25 lexical retrieval

BM25 is optimized for exact legal terms:

```text
Article 21
Section 302
Puttaswamy
Kesavananda Bharati
Section 498A
```

BM25 is particularly useful because it avoids confusing:

- Section 302 with Section 303
- Article 21 with Article 19
- closely named cases or parties

The corpus is tokenized into legal-aware word and provision lists, and BM25
scores the documents for the query.

## 5. Dense SBERT retrieval

If `sentence-transformers` is available, the system loads the embedding model
configured by `EMBED_MODEL`. The default is
`BAAI/bge-small-en-v1.5`.

Both queries and documents are converted to normalized vectors:

```text
right to privacy → [0.12, -0.44, 0.83, ...]
```

Cosine similarity is calculated through dot product. Dense retrieval finds
documents that use different words but discuss the same concept:

```text
Query:    "can police search a person's phone?"
Document: "principles governing seizure of digital evidence"
```

The code model-prefixes queries and passages differently where required. BGE
and E5 models receive model-compatible prefixes.

## 6. Embedding cache

Document embeddings are cached in `data/embeddings/<model-name>.npz`.

The cache key is a hash of:

```text
document title + first 1800 characters
```

On a later run:

- unchanged vectors are reused
- new or changed documents are embedded
- the cache is saved back to disk

This avoids embedding the complete legal corpus every time a question is asked.

## 7. Reciprocal Rank Fusion

BM25 and dense retrieval are treated as complementary ranked lists. The system
combines them using Reciprocal Rank Fusion (RRF).

For a document ranked `r`:

```text
RRF contribution = 1 / (60 + r)
```

For example:

| Rank | Contribution |
|---:|---:|
| 1 | 0.01639 |
| 2 | 0.01613 |
| 10 | 0.01429 |

A document appearing near the top in both BM25 and dense retrieval receives a
strong combined score.

BM25 and dense retrieval have equal base weights, while a small final score is
also used for legal-domain boosts.

## 8. Legal-domain boosts

After RRF, the system adds:

- **Provision boost:** shared Article, Section, Rule, or Schedule references.
- **Case-name boost:** both case title sides match a judgment title.
- **Holding boost:** at passage level, phrases such as `held that`,
  `in our view`, `ratio decidendi`, or `appeal is dismissed`.

Example:

```text
Query:    right to privacy Article 21 Puttaswamy
Document: Puttaswamy judgment mentioning Article 21
```

The document receives a provision match and named-case or case-title boost.

## 9. Light citation GraphRAG

The system reads the `cites` field of each document and builds an undirected
citation graph. If document A cites document B, both become neighbors.

If a strongly ranked document cites an in-corpus judgment, that neighbor can
be added only when it has a BM25 hit or dense cosine similarity of at least
`0.50`.

The graph therefore follows relevant citations:

```text
Puttaswamy judgment
  -> Kharak Singh
  -> Gobind
  -> Maneka Gandhi
```

Graph expansion adds at most two citation-neighbor documents by default.

## 10. Local retrieval confidence

The system does not assume the local corpus answered a query. It is confident
when one of these conditions is met.

### Lexical confidence

```text
best BM25 >= 12.0
and term coverage >= 50%
```

### Semantic confidence

```text
best cosine >= 0.60
and term coverage >= 30%
```

Term coverage is the fraction of meaningful query tokens found in at least one
of the top three documents.

The coverage guard prevents a semantically similar but legally unrelated
document from being accepted as authoritative.

## 11. Routing and fallback

The pipeline makes one of these routing decisions:

| Mode | Condition |
|---|---|
| `LOCAL_CORPUS` | Local BM25 or semantic confidence passes |
| `QUERY_CRAWL` | Local corpus is insufficient |
| `LOCAL_CORPUS` | `--local-only` is used |
| `QUERY_CRAWL` | `--force-web` is used |

The query crawler retrieves documents for the original query, writes or updates
`data/corpus.jsonl`, and searches the newly updated corpus.

The dashboard disables live crawling and uses `local_only=True`. This keeps the
web search responsive and makes RAG summaries come from the existing local corpus.

## 12. Syntax-aware chunking

Long judgments are divided into passages of approximately 220 words using:

- blank-line boundaries
- numbered paragraphs
- sentence boundaries
- legal abbreviation awareness
- one-sentence overlap

Abbreviations such as `Mr.`, `Dr.`, `Art.`, `Sec.`, `i.e.`, `e.g.`, and `etc.`
are not treated as a sentence ending by themselves.

For a 500-page judgment, the system does not score only the opening and closing
text. It creates windows and can find relevant reasoning anywhere in the judgment.

## 13. Passage-level max-match retrieval

Every passage is scored against the query using BM25, dense similarity, and
legal boosts.

```text
passage 1: 0.22
passage 2: 0.74
passage 3: 0.91
passage 4: 0.48
```

The top passages are selected. This is called max-match retrieval because the
system keeps the strongest window found anywhere in each long document.

The passage retriever:

- uses the top BM25 passages as the lexical shortlist
- allows dense encoding for up to 500 passages
- applies Reciprocal Rank Fusion
- gives provision and holding bonuses
- removes duplicate or near-duplicate passages
- limits the result to about two passages per document
- returns six passages by default

## 14. Grounded synthesis

The selected passages are converted to a context such as:

```text
[Doc 1] Title: Puttaswamy Judgment
Excerpt a: ...
Excerpt b: ...

[Doc 2] Title: Maneka Gandhi Judgment
Excerpt a: ...
```

The prompt asks the generator to:

1. begin with a direct holding or answer;
2. identify provisions and precedents;
3. summarize ratio decidendi;
4. use only supplied facts;
5. cite claims as `[Doc n]`;
6. state when evidence is incomplete;
7. add an informational and not-legal-advice disclaimer.

Generation temperature is `0.1`, producing a more restrained and factual style.

## 15. Provider fallback

Providers are tried in this order:

1. Google Gemini
2. Anthropic Claude
3. OpenAI
4. Ollama
5. Built-in extractive summarizer

Gemini also tries a comma-separated model list from `GEMINI_MODELS`.

If no API key or Ollama host is available, or all providers fail, the offline
extractive summarizer is used.

## 16. Offline extractive summary

The offline summarizer:

- splits passages into sentences;
- ranks each sentence by query overlap;
- adds SBERT similarity when embeddings are available;
- adds `0.4` to judicial-holding sentences;
- removes near-duplicates;
- returns up to five copied sentences;
- adds a source citation for each sentence.

It is extractive rather than generative. It copies and organizes real source
sentences instead of writing a new paragraph.

## 17. Citation validation

After a generated answer is received, the system searches for cited document
numbers.

It warns if:

- no `[Doc n]` citations are present;
- a `[Doc 9]` citation is present but only `[Doc 1]` and `[Doc 2]` exist.

This validates citation structure. It does not independently prove that the
legal statement is legally correct.

## Worked example

For:

```bash
python rag.py "right to privacy Article 21 Puttaswamy"
```

the system performs the following:

1. extracts `art_21` and `puttaswamy`;
2. scores the corpus with BM25 and dense retrieval;
3. applies provision and case-name boosts;
4. decides whether the local corpus is confident;
5. adds relevant citation neighbors;
6. chunks all selected judgments;
7. re-ranks and de-duplicates passages;
8. sends the top passages to the first available generator;
9. validates `[Doc n]` citations;
10. prints the answer, selected sources, and passage scores.

Example answer structure:

```text
Puttaswamy held that privacy is protected under Article 21 [Doc 1].
The judgment recognized autonomy and decisional privacy [Doc 1].
The supplied passages do not specify the requested remedy [Doc 2].
```

## Dashboard integration

The dashboard in [search_app.py](search_app.py) has a normal feedback-driven
keyword-search flow and an optional RAG panel.

The `/search` response:

1. runs the existing BM25/feedback search;
2. reads the `rag=1` form option;
3. creates or reuses `RAGPipeline`;
4. runs RAG with local-only routing;
5. caches a normalized-query result in memory;
6. renders a `RAG-generated legal summary` panel;
7. continues to render normal ranked search results below it.

The panel displays:

- retrieval mode;
- embedding model;
- synthesizer/provider;
- number of documents;
- number of passages;
- routing reason;
- generated or extractive summary.

RAG results are cached only in application memory. Restarting
`search_app.py` clears the cache.

## CLI examples

One-off query:

```bash
python rag.py "right to privacy Article 21"
```

Interactive mode:

```bash
python rag.py
```

BM25-only mode:

```bash
python rag.py "Section 302 murder" --no-dense
```

Disable citation graph:

```bash
python rag.py "Article 21" --no-graph
```

Force query-specific crawling:

```bash
python rag.py "new digital privacy law" --force-web
```

Disable crawling:

```bash
python rag.py "Article 21" --local-only
```

No external LLM:

```bash
python rag.py "Article 21" --offline
```

## Strengths

- fast local-first search;
- exact matching for provisions and case names;
- semantic matching for paraphrases;
- hybrid BM25 and embedding ranking;
- citation-neighbor expansion;
- long-document max-match retrieval;
- diversified passages;
- grounded generation instructions;
- provider and offline fallbacks;
- document citations.

## Limitations

Retrieval can be strong while the final answer is incomplete or incorrect if:

- the retrieved judgments are partially relevant;
- authorities conflict;
- the crawler has stale or incomplete data;
- source passages omit the answer;
- the generator misinterprets a source;
- citation structure is valid but the factual statement is wrong.

Citation validation checks citation structure, not legal correctness. The
system is a legal research assistant and not a replacement for a lawyer or
direct review of primary legal sources.
