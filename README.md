# Legal IR Pipeline

This command-line project searches for Indian legal documents, retrieves a limited
number of pages from configured sources, and ranks every retrieved document using
local BM25 keyword matching. It does not use an LLM relevance filter.

## What it retrieves

- **Court judgments / case law:** Indian Kanoon search results (direct scraping).
- **Acts and statutes:** India Code's public search API.
- **Additional web results:** Tavily, if `TAVILY_API_KEY` is configured. Tavily is
  optional and its API terms and limits may change.

The project does not currently crawl complete legal archives. Results are limited
to what each source returns for a query and the per-source limits in `pipeline.py`.
It does not specifically search for academic or educational "case studies"; it
focuses on legal texts such as judgments and legislation.

## Other free sources to consider

These are useful official sources. India Code is searched directly; when Tavily
is configured its search can also return pages from these domains:

- [Supreme Court of India](https://www.sci.gov.in/) — official court judgments.
- [eCourts Services](https://ecourts.gov.in/) — court services and case-related
  information, including lower-court material.
- [eGazette of India](https://egazette.gov.in/) — official Gazette notifications.

Their websites may change, impose access limits, or publish documents in formats
such as PDF. Before automating collection, check the site's terms and access
policies; scraping every record is not guaranteed by a search-page scraper.

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python pipeline.py "right to privacy Article 21 India"
```

Run `.venv\Scripts\python pipeline.py` without a query to be prompted interactively.

Use a standard CPython or Anaconda Python (3.10+). The MSYS2/MinGW Python
(`C:\msys64\...\python.exe`) cannot install numpy, which `rank-bm25` needs.

### Caching

Pages downloaded from Indian Kanoon are saved in `cache/` (not committed).
Running the same query again re-uses them, so results are identical and the
site is not hit again. Delete `cache/` to download fresh copies. India Code and
Tavily API results are not cached.

The direct Indian Kanoon and India Code scrapers, and BM25 ranking, need no API
keys. To enable Tavily results, set `TAVILY_API_KEY` in a `.env` file in the
project folder or as an environment variable. Do not commit or share `.env`.

## Building the offline corpus (Phase 1)

```powershell
.venv\Scripts\python crawl.py     # ~2.5 hours, ~540 MB in data/
.venv\Scripts\python dedup.py     # near-duplicate removal, ~minutes
```

`crawl.py` searches Indian Kanoon for each query in `seeds.txt`, then follows
the citations inside those judgments (most-cited first) to collect up to 5,000
judgments and statute sections. It writes `data/corpus.jsonl` and
`data/citations.tsv`. If it is interrupted, or stops because the network is
down, run it again: documents already saved are skipped.

`dedup.py` finds near-duplicates with shingling, MinHash and LSH, and writes
`data/duplicates.tsv` and `data/corpus_dedup.jsonl`. See `PHASE1_CHANGES.md`.

Quick test (~1 minute):
`crawl.py --seed-limit 2 --pages-per-seed 1 --max-docs 40 --out data/test`,
then `dedup.py data/test`.

## Text processing and corpus statistics (Phase 2)

```powershell
.venv\Scripts\python text_processing.py      # demo: tokens, stopwords, stemming
.venv\Scripts\python corpus_stats.py         # Zipf + Heaps, ~10 minutes
.venv\Scripts\python spelling.py demo        # Hamming, LCS, Levenshtein
.venv\Scripts\python spelling.py correct "anticipatary habeus corpas privecy"
.venv\Scripts\python spelling.py case "putaswami"
.venv\Scripts\python spelling.py evaluate    # accuracy on eval/misspellings.tsv
```

`corpus_stats.py` writes `analysis/zipf.png`, `analysis/heaps.png`,
`analysis/corpus_stats.md` and `data/vocabulary.tsv`; `spelling.py` needs the
vocabulary file. The lemmatizer needs WordNet once:
`.venv\Scripts\python -c "import nltk; nltk.download('wordnet')"`.
See `PHASE2_CHANGES.md`.

## Index and Boolean search (Phase 3)

```powershell
.venv\Scripts\python index.py build        # ~80 s → data/index_porter.pkl
.venv\Scripts\python boolean_search.py "anticipatory AND bail AND NOT murder"
.venv\Scripts\python boolean_search.py '"right to privacy" AND surveillance'
.venv\Scripts\python boolean_search.py "bail /5 437"
.venv\Scripts\python compression.py         # → analysis/compression.md
```

See `PHASE3_CHANGES.md`. The project scope, and what was left out and why,
is in `plan.md`.

## How it works

```text
Query
  → Indian Kanoon judgments + India Code legislation (public DSpace search API)
  → optional Tavily results
  → deduplicate retrieved URLs
  → BM25-rank every retrieved document
  → print the top 10
```

All retrieved documents are ranked and returned by `run_pipeline`; only the top
10 are printed by default. The output shows each provider's retrieved count,
top-10 count, and best rank, so you can tell whether a source returned documents
that ranked below the display limit. Change `MAX_FINAL` in `pipeline.py` to print
more.

## Limits

`MAX_PER_SOURCE` in `pipeline.py` is 20. The scraper asks Indian Kanoon for up to
20 results, India Code for up to 10, and Tavily (when enabled) for up to 20.
These are per-query caps, not the total number of documents in those archives.

The old `relevance_filter.py` module remains in the project, but the main pipeline
does not call it. Its Anthropic key is not needed to run `pipeline.py`.
