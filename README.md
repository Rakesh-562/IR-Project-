# Legal IR Pipeline — Setup & Run Guide

## File Structure

```
relevance_filter.py   ← LLM relevance judge (already built)
legal_scraper.py      ← scrapes real legal sites
pipeline.py           ← connects everything, shows final results
README.md             ← this file
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## API Keys You Need

### Tavily (optional search provider)
- Get a key at https://app.tavily.com
- The scraper loads `TAVILY_API_KEY` from the project `.env` file.

### Anthropic (optional LLM relevance judgments)
- Set `ANTHROPIC_API_KEY` in `.env` if you want LLM-based relevance judgments.
- Without it, the pipeline uses its local BM25 ranking and makes no paid API calls.

## Set Your Keys

Option A — create a `.env` file in the project folder (recommended):
```dotenv
TAVILY_API_KEY=your_tavily_key_here
# Optional: enables LLM relevance judgments
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

The pipeline loads `.env` automatically. Direct Indian Kanoon and India Code
scraping does not require API keys. BM25 fallback also runs locally without an
Anthropic key. Do not commit `.env` or share its contents.

Option B — environment variables:
```powershell
$env:TAVILY_API_KEY="your_tavily_key_here"
# Optional for LLM relevance judgments:
# $env:ANTHROPIC_API_KEY="your_anthropic_key_here"
```

## Run

```bash
# As CLI
python pipeline.py "right to privacy Article 21 India"

# Or interactive
python pipeline.py
# Enter legal search query: bail conditions under CrPC section 437

# More examples
python pipeline.py "fundamental rights violation police custody"
python pipeline.py "property rights women Hindu Succession Act"
python pipeline.py "RTI application rejection appeal procedure"
```

## What the Output Looks Like

```
════════════════════════════════════════════════════════════════════════
  LEGAL SEARCH RESULTS
  Query : right to privacy Article 21 India
════════════════════════════════════════════════════════════════════════

  Filter summary: 18 scraped  →  8 relevant  |  4 partial  |  6 discarded

────────────────────────────────────────────────────────────────────────
  [1]  ✓ RELEVANT   BM25: 4.821
  Justice K.S. Puttaswamy v. Union of India
  https://indiankanoon.org/doc/127517806/
  Source : indiankanoon.org
  Reason : Landmark Supreme Court judgment establishing right to
           privacy as a fundamental right under Article 21.
  Preview: The nine-judge bench unanimously held that the right to
           privacy is a fundamental right protected under Article 21 ...
```

## How It Works (Architecture)

```
User Query
    │
    ▼
legal_scraper.py
    ├── Indian Kanoon website ── court judgments
    ├── India Code website ───── statutes & acts
    └── Tavily Search API ────── optional search results
    │
    ▼ (all scraped docs as ScrapedDoc objects)
    │
relevance_filter.py
    ├── BM25 pre-filter    ──── top-25 candidates only
    └── LLM judge (Claude) ──── relevant / partial / not_relevant
    │
    ▼ (only relevant + partial pass through)
    │
pipeline.py → BM25 ranker
    │
    ▼
Ranked Results
```

## Tuning Parameters

| Parameter        | File         | Default | What it controls                        |
|------------------|--------------|---------|-----------------------------------------|
| MAX_PER_SOURCE   | pipeline.py  | 5       | Docs fetched per source                 |
| BM25_PREFILTER   | pipeline.py  | 25      | Max docs sent to LLM filter             |
| MAX_FINAL        | pipeline.py  | 10      | Results shown to user                   |
| EXCERPT_WORDS    | relevance_filter.py | 400 | Words sent to LLM per doc          |
| MODEL            | relevance_filter.py | claude-haiku-4-5 | LLM model used     |
| SCRAPE_DELAY     | legal_scraper.py | 0.8s | Polite delay between requests        |
