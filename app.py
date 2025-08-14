"""
Free Multi‑Source Web Search (Streamlit)
=======================================

Synchronous & thread‑safe version.

Why this change?
- Some Streamlit Cloud environments redact async errors and certain async
  HTTP clients can clash with the app event loop. This version removes
  asyncio entirely and uses a ThreadPoolExecutor for safe concurrency.

All sources are still free/open:
- DuckDuckGo Web (`duckduckgo_search`)
- DuckDuckGo Instant Answer API (JSON)
- Wikipedia (MediaWiki API)
- SearXNG (self‑host/public instance, JSON)
- arXiv API (Atom)
- Crossref Works API (JSON)
- Internet Archive Advanced Search (JSON)
- Wikidata Entity Search API (JSON)

Author: You
License: MIT
"""

from __future__ import annotations

# =========================
#          SETUP
# =========================

import json
import html
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import httpx
from bs4 import BeautifulSoup  # (kept in case you want to parse pages later)
from duckduckgo_search import DDGS
import feedparser

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Free Deep Web Search",
    page_icon="🔎",
    layout="wide",
)

# =========================
#        DATA MODEL
# =========================

@dataclass
class SearchResult:
    """Normalized search result structure."""
    title: str
    snippet: str
    url: str
    source: str
    extra: Dict[str, Any] | None = None

# =========================
#      HELPER FUNCTIONS
# =========================

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = httpx.Timeout(12.0, connect=6.0)


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    cut = text.rfind(" ", 0, max_len)
    if cut == -1:
        cut = max_len
    return text[:cut].rstrip() + "…"


def _domain(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", httpx.URL(url).host or "")
    except Exception:
        return ""

# =========================
#       SEARCH CLIENTS
# =========================

def search_ddg_web(query: str, max_results: int = 6) -> List[SearchResult]:
    """DuckDuckGo web search via `duckduckgo_search` lib (no API key)."""
    results: List[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        title=r.get("title") or "Untitled",
                        snippet=_truncate(_clean_html(r.get("body") or "")),
                        url=r.get("href") or "",
                        source="DuckDuckGo",
                    )
                )
    except Exception as e:
        results.append(SearchResult("DuckDuckGo Error", str(e), "", "DuckDuckGo"))
    return results


def search_ddg_instant(query: str) -> List[SearchResult]:
    """DuckDuckGo Instant Answer API (free, no key)."""
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            data = client.get(url, params=params).json()
        results: List[SearchResult] = []
        abstract = data.get("AbstractText") or data.get("Abstract") or ""
        abstract_url = data.get("AbstractURL") or (data.get("Results", [{}])[0].get("FirstURL", "") if data.get("Results") else "")
        heading = data.get("Heading") or "DuckDuckGo Answer"
        if abstract:
            results.append(SearchResult(heading, _truncate(_clean_html(abstract)), abstract_url, "DuckDuckGo Instant Answer"))
        related = data.get("RelatedTopics") or []
        for item in related[:3]:
            if isinstance(item, dict):
                txt = item.get("Text") or ""
                first = item.get("FirstURL") or ""
                if txt:
                    results.append(
                        SearchResult(
                            title=_truncate(_clean_html(txt.split(" - ")[0]), 90),
                            snippet=_truncate(_clean_html(txt), 200),
                            url=first,
                            source="DuckDuckGo Instant Answer",
                        )
                    )
        if not results:
            results.append(SearchResult("No instant answer", "DuckDuckGo did not return an abstract for this query.", "", "DuckDuckGo Instant Answer"))
        return results
    except Exception as e:
        return [SearchResult("DuckDuckGo IA Error", str(e), "", "DuckDuckGo Instant Answer")]


def search_wikipedia(query: str, max_results: int = 5) -> List[SearchResult]:
    """Wikipedia via MediaWiki API (no key)."""
    API = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": str(max_results), "format": "json", "utf8": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            data = client.get(API, params=params).json()
        results: List[SearchResult] = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "Wikipedia Article")
            snippet_html = item.get("snippet", "")
            pageid = item.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
            results.append(SearchResult(title, _truncate(_clean_html(snippet_html)), url, "Wikipedia", extra={"pageid": pageid}))
        if not results:
            results.append(SearchResult("No Wikipedia results", "Try refining your query.", "", "Wikipedia"))
        return results
    except Exception as e:
        return [SearchResult("Wikipedia Error", str(e), "", "Wikipedia")]


def search_searxng(query: str, instance_url: str, max_results: int = 6) -> List[SearchResult]:
    """SearXNG metasearch (JSON). Provide an instance base URL."""
    if not instance_url:
        return [SearchResult("SearXNG not configured", "Provide an instance URL in the sidebar to enable SearXNG.", "", "SearXNG")]
    endpoint = instance_url.rstrip("/") + "/search"
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            data = client.get(endpoint, params=params).json()
        results: List[SearchResult] = []
        for r in (data.get("results") or [])[:max_results]:
            results.append(SearchResult(r.get("title") or "Untitled", _truncate(_clean_html(r.get("content") or "")), r.get("url") or "", "SearXNG", extra={"engine": r.get("engine")}))
        if not results:
            results.append(SearchResult("No SearXNG results", "The instance returned no results.", "", "SearXNG"))
        return results
    except Exception as e:
        return [SearchResult("SearXNG Error", str(e), "", "SearXNG")]


def search_arxiv(query: str, max_results: int = 5) -> List[SearchResult]:
    """arXiv API (Atom feed)."""
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            text = client.get(url, params=params).text
        feed = feedparser.parse(text)
        results: List[SearchResult] = []
        for e in feed.entries:
            link = e.get("link") or (e.links[0].href if e.get("links") else "")
            results.append(SearchResult(e.get("title", "arXiv Paper").strip(), _truncate(_clean_html(e.get("summary", "").strip())), link, "arXiv", extra={"published": e.get("published")}))
        if not results:
            results.append(SearchResult("No arXiv results", "Try a more technical/academic query.", "", "arXiv"))
        return results
    except Exception as e:
        return [SearchResult("arXiv Error", str(e), "", "arXiv")]


def search_crossref(query: str, rows: int = 5) -> List[SearchResult]:
    """Crossref Works API (no key)."""
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            data = client.get(url, params=params).json()
        items = (data.get("message", {}) or {}).get("items", [])
        results: List[SearchResult] = []
        for it in items:
            title = "; ".join(it.get("title", [])).strip() or "Crossref Work"
            url_out = it.get("URL", "")
            abstract = _clean_html(it.get("abstract", ""))
            if not abstract:
                container = "; ".join(it.get("container-title", [])).strip()
                authors = ", ".join([f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", [])][:3])
                parts = [p for p in [container, authors] if p]
                abstract = "; ".join(parts)
            results.append(SearchResult(title, _truncate(abstract or ""), url_out, "Crossref"))
        if not results:
            results.append(SearchResult("No Crossref results", "Try a different query or enable arXiv.", "", "Crossref"))
        return results
    except Exception as e:
        return [SearchResult("Crossref Error", str(e), "", "Crossref")]


def search_internet_archive(query: str, rows: int = 5) -> List[SearchResult]:
    """Internet Archive Advanced Search (no key)."""
    url = "https://archive.org/advancedsearch.php"
    params = {"q": query, "output": "json", "rows": rows}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            data = client.get(url, params=params).json()
        docs = (data.get("response", {}) or {}).get("docs", [])
        results: List[SearchResult] = []
        for d in docs:
            identifier = d.get("identifier", "")
            url_out = f"https://archive.org/details/{identifier}" if identifier else ""
            title = d.get("title") or identifier or "Internet Archive Item"
            desc = d.get("description") or ""
            if isinstance(desc, list):
                desc = " ".join(desc)
            results.append(SearchResult(_truncate(_clean_html(title), 100), _truncate(_clean_html(desc)), url_out, "Internet Archive"))
        if not results:
            results.append(SearchResult("No Archive results", "Try a broader query.", "", "Internet Archive"))
        return results
    except Exception as e:
        return [SearchResult("Internet Archive Error", str(e), "", "Internet Archive")]


def search_wikidata(query: str, limit: int = 5) -> List[SearchResult]:
    """Wikidata entity search API (no key)."""
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "search": query, "language": "en", "format": "json", "limit": limit}
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
            data = client.get(url, params=params).json()
        results: List[SearchResult] = []
        for e in data.get("search", [])[:limit]:
            title = e.get("label") or "Wikidata Entity"
            desc = e.get("description") or ""
            url_out = e.get("url") or (f"https://www.wikidata.org/wiki/{e.get('id')}" if e.get("id") else "")
            results.append(SearchResult(title, _truncate(_clean_html(desc)), url_out, "Wikidata"))
        if not results:
            results.append(SearchResult("No Wikidata results", "Try refining your query.", "", "Wikidata"))
        return results
    except Exception as e:
        return [SearchResult("Wikidata Error", str(e), "", "Wikidata")]

# =========================
#      SUMMARY (LOCAL)
# =========================

STOPWORDS = {"the","is","at","of","on","and","a","to","in","for","by","an","be","as","are","that","with","or","from","this","it","its","we","our","you","your","about","into"}

def extractive_summary(snippets: Iterable[str], max_sentences: int = 5) -> str:
    text = _clean_html(" ".join([s for s in snippets if s]))
    if not text:
        return "Not enough content to summarize."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return _truncate(" ".join(sentences), 1200)
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    freqs: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return _truncate(" ".join(sentences[:max_sentences]), 1200)
    scored = []
    for s in sentences:
        s_words = re.findall(r"[A-Za-z][A-Za-z\-']+", s.lower())
        score = sum(freqs.get(w, 0) for w in s_words) / (len(s_words) + 1)
        scored.append((score, s))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[: max_sentences * 3]
    selected = {s for _, s in top}
    ordered = [s for s in sentences if s in selected][:max_sentences]
    return _truncate(" ".join(ordered), 1200)

# =========================
#        CONTROLLER
# =========================

def run_selected_sources(query: str, sources: List[str], searxng_url: str, max_results: int) -> Dict[str, List[SearchResult]]:
    """Run selected sources concurrently using threads (safe in Streamlit)."""
    fn_map = {
        "DuckDuckGo": lambda: search_ddg_web(query, max_results),
        "DuckDuckGo Instant Answer": lambda: search_ddg_instant(query),
        "Wikipedia": lambda: search_wikipedia(query, max_results),
        "SearXNG": lambda: search_searxng(query, searxng_url, max_results),
        "arXiv": lambda: search_arxiv(query, max_results),
        "Crossref": lambda: search_crossref(query, max_results),
        "Internet Archive": lambda: search_internet_archive(query, max_results),
        "Wikidata": lambda: search_wikidata(query, max_results),
    }

    results: Dict[str, List[SearchResult]] = {s: [] for s in sources}
    futures = {}
    with ThreadPoolExecutor(max_workers=min(len(sources), 8) or 1) as ex:
        for s in sources:
            if s in fn_map:
                futures[ex.submit(fn_map[s])] = s
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                results[src] = fut.result()
            except Exception as e:
                results[src] = [SearchResult(f"{src} Error", str(e), "", src)]
    return results

# =========================
#         STREAMLIT UI
# =========================

st.title("🔎 Free Multi‑Source Web Search")
st.caption("Search multiple free/open sources concurrently. Get organized results with citations.")

with st.sidebar:
    st.header("Settings")
    default_sources = [
        "DuckDuckGo",
        "DuckDuckGo Instant Answer",
        "Wikipedia",
        "SearXNG",
        "arXiv",
        "Crossref",
        "Internet Archive",
        "Wikidata",
    ]
    sources = st.multiselect(
        "Select sources",
        options=default_sources,
        default=["DuckDuckGo", "Wikipedia", "DuckDuckGo Instant Answer", "Wikidata"],
        help="Choose which providers to query in parallel.",
    )

    max_results = st.slider("Max results per source", 1, 10, 5)

    searxng_url = st.text_input(
        "SearXNG instance URL (optional)",
        value="",
        placeholder="e.g. https://searx.be or your own instance",
        help="Provide a SearXNG base URL to enable metasearch.",
    )

    show_summary = st.checkbox("Generate consolidated summary", value=True)
    verbose_errors = st.checkbox("Show verbose errors in UI", value=True, help="Useful on Streamlit Cloud where errors are redacted.")

query = st.text_input(
    "Search topic / question",
    placeholder="e.g. quantum error correction, climate policy 2025, best open data portals",
)

run = st.button("Run search", type="primary")

results_container = st.container()

if run and query.strip():
    with st.spinner("Fetching results from selected sources…"):
        try:
            results_by_source = run_selected_sources(query.strip(), sources, searxng_url.strip(), max_results)
        except Exception as e:
            if verbose_errors:
                st.exception(e)
            else:
                st.error("Search failed. Enable 'Show verbose errors in UI' in the sidebar for details.")
            st.stop()

    with results_container:
        st.subheader("Results by Source")
        all_for_summary: List[str] = []

        for src in sources:
            items = results_by_source.get(src, [])
            with st.expander(f"{src} ({len(items)} results)", expanded=True):
                if not items:
                    st.info("No results.")
                for r in items:
                    col1, col2 = st.columns([0.88, 0.12], vertical_alignment="center")
                    with col1:
                        title = r.title or "Untitled"
                        url = r.url or ""
                        snippet = r.snippet or ""
                        domain = _domain(url)
                        st.markdown(
                            f"**[{title}]({url})**  "+(f"`{domain}`" if domain else "")+"\n"+
                            (snippet if snippet else ""),
                        )
                    with col2:
                        if r.url:
                            st.link_button("Open", r.url, use_container_width=True)
                all_for_summary.extend([it.snippet for it in items if it.snippet])

        if show_summary:
            st.subheader("📝 Consolidated Summary (extractive)")
            st.write(extractive_summary(all_for_summary, max_sentences=6))

        export: Dict[str, List[Dict[str, Any]]] = {src: [asdict(it) for it in results_by_source.get(src, [])] for src in sources}
        st.download_button(
            label="⬇️ Download Results (JSON)",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name="search_results.json",
            mime="application/json",
        )
else:
    st.info("Enter a query and click **Run search** to get started.")
