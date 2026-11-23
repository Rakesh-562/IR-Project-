"""
search_app.py — Phase 7: a small search website that learns from clicks.

    python search_app.py            then open http://127.0.0.1:8000

Pages:
    /search?q=...      ranked results (BM25 + chosen feedback)
    /doc?id=...        one document. While it is open, the page measures how
                       long the user stays. On leaving (back button, closing
                       the tab), the browser sends that time to /dwell.
    /dwell             (POST) logs {query, document, rank, seconds} to
                       data/clicks.jsonl via feedback.log_click()

Implicit feedback (see feedback.py): the next search for the same query
treats documents read ≥ 30 s as relevant and documents left in < 10 s as
not relevant, and re-weights the query with Rocchio.

Uses only Python's built-in web server. It listens on 127.0.0.1 (this
computer only).
"""

import argparse
import asyncio
import html
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlencode, urlparse

from feedback import FeedbackSearch, log_click
from index import load
from judge import best_window
from spelling import SpellCorrector
from ranking import Ranker
from rag import RAGPipeline, RAGResult

MAX_WORDS_SHOWN = 20_000          # very long judgments are cut on the page
RAG_TOP_DOCS = 5
RAG_TOP_PASSAGES = 6

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
 body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 16px; line-height: 1.5; }}
 .result {{ margin: 18px 0; }} .meta {{ color: #666; font-size: 0.9em; }} .note {{ color: #555; font-size: 0.9em; }}
 input[type=text] {{ width: 70%; padding: 6px; }} .text {{ white-space: pre-wrap; }}
 .rag-summary {{ border: 1px solid #d8d8d8; border-left: 5px solid #2f6fed; background: #f7faff;
                 border-radius: 6px; padding: 16px; margin: 18px 0 24px; }}
 .rag-summary h2 {{ margin-top: 0; }} .rag-summary .rag-text {{ white-space: pre-wrap; }}
</style></head><body>{body}</body></html>"""

def search_form(query: str = "", options: dict | None = None) -> str:
    options = options or {"clicks": True}
    box = lambda name, label: (f'<label><input type="checkbox" name="{name}" value="1"'
                               f'{" checked" if options.get(name) else ""}> {label}</label> ')
    return (f'<form action="/search"><input type="text" name="q" value="{html.escape(query)}" '
            f'placeholder="e.g. anticipatory bail 438 crpc" autofocus> <button>Search</button><br>'
            + box("clicks", "learn from clicks") + box("thesaurus", "legal thesaurus")
            + box("prf", "pseudo-relevance feedback") + box("rag", "include RAG summary") + "</form>")

def format_rag_summary(result: RAGResult) -> str:
    """Render the RAG result as a safe, concise dashboard panel."""
    summary = html.escape(result.summary).replace("\n", "<br>")
    return (
        '<section class="rag-summary">'
        '<h2>RAG-generated legal summary</h2>'
        f'<div class="meta">Source: {html.escape(result.retrieval_mode)}'
        f' · Embeddings: {html.escape(result.embedding_model)}'
        f' · Synthesizer: {html.escape(result.generation_method)}</div>'
        f'<div class="rag-text">{summary}</div>'
        '<div class="meta">'
        f'{len(result.top_docs)} document(s) retrieved · '
        f'{len(result.selected_passages)} passage(s) used · '
        f'{html.escape(result.reason)}</div>'
        '</section>'
    )

class App:
    def __init__(self):
        print("[app] loading index ...")
        self.search = FeedbackSearch(Ranker(load("porter")))
        print("[app] loading spelling vocabulary ...")
        self.speller = SpellCorrector()
        self._rag_pipeline: RAGPipeline | None = None
        self._rag_cache: dict[str, RAGResult] = {}
        self._rag_lock = threading.Lock()

    def rag_summary(self, query: str) -> RAGResult:
        cache_key = " ".join(query.lower().split())
        with self._rag_lock:
            cached = self._rag_cache.get(cache_key)
            if cached is not None:
                return cached

            if self._rag_pipeline is None:
                print("[app] loading RAG pipeline ...")
                self._rag_pipeline = RAGPipeline()

        result = asyncio.run(
            self._rag_pipeline.run(
                query=query,
                force_web=False,
                local_only=True,
                offline=False,
                top_k_docs=RAG_TOP_DOCS,
                top_k_passages=RAG_TOP_PASSAGES,
            )
        )
        with self._rag_lock:
            self._rag_cache[cache_key] = result
        return result

app: App | None = None

class Handler(BaseHTTPRequestHandler):
    def send_page(self, title: str, body: str, status: int = 200) -> None:
        data = PAGE.format(title=html.escape(title), body=body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/search"):
            self.results_page(params)
        elif url.path == "/doc":
            self.document_page(params)
        else:
            self.send_page("Not found", "<p>Not found.</p>", 404)

    def results_page(self, params: dict) -> None:
        query = params.get("q", "").strip()
        options = {name: params.get(name) == "1" for name in ("clicks", "thesaurus", "prf", "rag")}
        if not query:
            self.send_page("Legal search", "<h1>Legal search</h1>" + search_form())
            return
        fs = app.search
        feedback_options = {
            name: options[name]
            for name in ("clicks", "thesaurus", "prf")
        }

        # Spelling correction (spelling.py), unless the user asked for the query as typed
        typed, correction = query, ""
        if params.get("nocorrect") != "1":
            corrected = app.speller.correct_query(query)
            if corrected != " ".join(query.lower().split()):
                query = corrected
                as_typed = urlencode({"q": typed, "nocorrect": "1",
                                      **{k: "1" for k, on in feedback_options.items() if on}})
                correction = (f'<p>Showing results for <b>{html.escape(corrected)}</b>. '
                              f'<a href="/search?{as_typed}">Search instead for '
                              f'{html.escape(typed)}</a></p>')

        ranked, note = fs.search(query, 10, **feedback_options)
        parts = [search_form(typed, options), correction,
                 f'<p class="note">Feedback used: {html.escape(note)}</p>']
        if options["rag"]:
            parts.append('<p class="note">Generating RAG summary from the local corpus...</p>')
            try:
                rag_result = app.rag_summary(query)
                parts[-1] = format_rag_summary(rag_result)
            except Exception as exc:
                print(f"[app] RAG summary failed: {exc}")
                parts[-1] = (
                    '<section class="rag-summary"><h2>RAG-generated legal summary</h2>'
                    '<p>RAG summary could not be generated. The keyword search results are still available.</p>'
                    '</section>'
                )
        for rank, (doc, score) in enumerate(ranked, 1):
            info = fs.store.text(fs.index.doc_ids[doc])
            words = info["text"].split()[:MAX_WORDS_SHOWN]
            snippet = best_window(" ".join(words), query, 40)
            link = f'/doc?id={quote(info["doc_id"])}&q={quote(query)}&rank={rank}'
            parts.append(
                f'<div class="result"><a href="{link}">{rank}. {html.escape(info["title"])}</a>'
                f'<div class="meta">{html.escape(info["court"])} · score {score:.2f}</div>'
                f'<div>… {html.escape(snippet)} …</div></div>')
        if not ranked:
            parts.append("<p>No documents found.</p>")
        self.send_page(f"{query} — Legal search", "".join(parts))

    def document_page(self, params: dict) -> None:
        fs = app.search
        doc_id, query = params.get("id", ""), params.get("q", "")
        if doc_id not in fs.store.offset:
            self.send_page("Not found", "<p>Unknown document.</p>", 404)
            return
        info = fs.store.text(doc_id)
        words = info["text"].split()
        shown = " ".join(words[:MAX_WORDS_SHOWN])
        cut = (f'<p class="note">Showing the first {MAX_WORDS_SHOWN:,} of {len(words):,} words.</p>'
               if len(words) > MAX_WORDS_SHOWN else "")
        # Dwell time: measured in the browser, sent when the user leaves the page.
        # json.dumps makes the values safe JavaScript; "</" is escaped so the text can't end the script.
        beacon = json.dumps({"q": query, "doc": doc_id, "rank": int(params.get("rank", 0) or 0)})
        beacon = beacon.replace("</", "<\\/")
        body = (f'<p><a href="/search?q={quote(query)}&clicks=1">← back to results</a></p>'
                f'<h2>{html.escape(info["title"])}</h2>'
                f'<p class="meta">{html.escape(info["court"])} · {html.escape(info["date"])} · '
                f'<a href="{html.escape(info["url"])}" target="_blank" rel="noopener">Indian Kanoon</a></p>'
                f'{cut}<div class="text">{html.escape(shown)}</div>'
                f"""<script>
                const opened = Date.now(), click = {beacon};
                window.addEventListener("pagehide", () => {{
                  click.seconds = (Date.now() - opened) / 1000;
                  navigator.sendBeacon("/dwell", JSON.stringify(click));
                }});
                </script>""")
        self.send_page(info["title"], body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/dwell":
            self.send_response(404); self.end_headers(); return
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 10_000)
            click = json.loads(self.rfile.read(length))
            if click["doc"] in app.search.store.offset and click["q"]:
                log_click(click["q"], click["doc"], int(click["rank"]), float(click["seconds"]))
        except (ValueError, KeyError, TypeError):
            pass                                            # ignore malformed beacons
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args) -> None:           # keep the console quiet
        pass


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local legal search website with click feedback.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = App()
    print(f"[app] open http://127.0.0.1:{args.port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
