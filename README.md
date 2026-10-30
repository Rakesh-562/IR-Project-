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
pip install -r requirements.txt
python pipeline.py "right to privacy Article 21 India"
```

Run `python pipeline.py` without a query to be prompted interactively.

The direct Indian Kanoon and India Code scrapers, and BM25 ranking, need no API
keys. To enable Tavily results, set `TAVILY_API_KEY` in a `.env` file in the
project folder or as an environment variable. Do not commit or share `.env`.

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
