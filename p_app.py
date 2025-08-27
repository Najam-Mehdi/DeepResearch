# app.py

# =========================
#         SETUP
# =========================

import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import wikipedia
import concurrent.futures

# =========================
#      UTILITY FUNCTIONS
# =========================

def get_duckduckgo_results(query, max_results=5):
    """
    Search DuckDuckGo for the query and return a list of results.
    Each result contains: title, snippet, and URL.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "No Title"),
                    "snippet": r.get("body", "No Snippet"),
                    "url": r.get("href", ""),
                })
    except Exception as e:
        results.append({"title": "Error", "snippet": str(e), "url": ""})
    return results

def get_wikipedia_results(query):
    """
    Search Wikipedia for the query and return summary + page url.
    """
    results = []
    try:
        page = wikipedia.page(query)
        summary = wikipedia.summary(query, sentences=3)
        results.append({
            "title": page.title,
            "snippet": summary,
            "url": page.url
        })
    except wikipedia.DisambiguationError as e:
        # Handle ambiguous topics
        results.append({
            "title": "Wikipedia Disambiguation",
            "snippet": f"Topic is ambiguous. Options: {', '.join(e.options[:5])}",
            "url": ""
        })
    except wikipedia.PageError:
        results.append({
            "title": "Wikipedia Page Not Found",
            "snippet": "No page found for this query.",
            "url": ""
        })
    except Exception as e:
        results.append({
            "title": "Wikipedia Error",
            "snippet": str(e),
            "url": ""
        })
    return results

def get_web_scraped_results(query, max_results=5):
    """
    Perform a simple web scrape of DuckDuckGo HTML search results.
    NOTE: This is just for demonstration, as duckduckgo-search is more robust.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
    results = []
    try:
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select(".result__title")
        snippets = soup.select(".result__snippet")
        for i in range(min(max_results, len(links))):
            a_tag = links[i].find("a", href=True)
            title = a_tag.get_text() if a_tag else "No Title"
            href = a_tag['href'] if a_tag else ""
            snippet = snippets[i].get_text() if i < len(snippets) else ""
            results.append({
                "title": title,
                "snippet": snippet,
                "url": href
            })
    except Exception as e:
        results.append({
            "title": "Web Scraping Error",
            "snippet": str(e),
            "url": ""
        })
    return results

def summarize_results(all_results):
    """
    Create a simple consolidated summary from the fetched results.
    """
    summary = ""
    for source, results in all_results.items():
        if results and isinstance(results, list):
            summary += f"\n\n**{source}**:\n"
            for r in results[:2]:
                summary += f"- {r.get('title', '')}: {r.get('snippet', '')}\n"
    return summary.strip()

# =========================
#      STREAMLIT UI
# =========================

st.set_page_config(page_title="Free Deep Web Search App", layout="centered")
st.title("🔍 Free Deep Web Search App")
st.markdown("""
Enter a topic or question, and this app will search multiple free web sources and present results with citations.
""")

query = st.text_input("Enter a topic, question, or keyword:", "")
search_button = st.button("Search")

if search_button and query.strip():
    st.info(f"Searching for: **{query}** ...")

    # Use threads to fetch concurrently
    with st.spinner("Fetching results from multiple sources..."):
        all_results = {}

        # Use ThreadPoolExecutor for concurrent searches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                "Wikipedia": executor.submit(get_wikipedia_results, query),
                "DuckDuckGo": executor.submit(get_duckduckgo_results, query),
                "Web Scraped Results": executor.submit(get_web_scraped_results, query),
            }
            for source, future in futures.items():
                all_results[source] = future.result()

    # DISPLAY RESULTS
    st.header("🔎 Results by Source")
    for source, results in all_results.items():
        st.subheader(source)
        if results and isinstance(results, list):
            for r in results:
                st.markdown(f"**{r['title']}**  \n"
                            f"{r['snippet']}  \n"
                            f"[Source Link]({r['url']})" if r['url'] else "", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.write("No results found.")

    # SUMMARY SECTION (optional)
    if st.checkbox("Show summary of main insights (auto-generated)"):
        summary = summarize_results(all_results)
        st.markdown("### 📝 Summary")
        st.markdown(summary)

# ============ END ============

st.markdown("""
---
**Instructions:**  
- Enter your search topic and press **Search**.  
- You’ll see grouped results with proper citations.  
- Optionally, click "Show summary..." for a quick overview.  
- All tools and APIs used are 100% free.
""")




#---------------------------------------------------------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------



# app.py — Streamlit Deep Research Tool (free sources only)
# --------------------------------------------------------
# Quickstart (recommended in a fresh virtual env):
#   pip install streamlit duckduckgo-search httpx beautifulsoup4 langdetect
# Then run:
#   streamlit run app.py
#
# Optional (improves extraction/UX):
#   pip install readability-lxml tldextract python-frontmatter
# --------------------------------------------------------

import asyncio
import contextlib
import dataclasses
import html
import io
import json
import math
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, quote

import streamlit as st

# Try to import optional dependencies gracefully
try:
    from duckduckgo_search import DDGS  # duckduckgo-search >= 6.x
except Exception:  # pragma: no cover
    DDGS = None

try:
    import httpx
except Exception as e:  # pragma: no cover
    raise SystemExit("This app requires httpx. Install it with: pip install httpx")

try:
    from bs4 import BeautifulSoup, SoupStrainer
except Exception as e:  # pragma: no cover
    raise SystemExit("This app requires beautifulsoup4. Install it with: pip install beautifulsoup4")

try:
    from langdetect import detect
except Exception:
    detect = None  # We'll fall back to naive heuristics if unavailable

# Optional, improves main-content extraction if available
with contextlib.suppress(Exception):
    from readability import Document  # readability-lxml

with contextlib.suppress(Exception):
    import tldextract

# -----------------------------
# Page config & simple theming
# -----------------------------
st.set_page_config(
    page_title="Deep Research (Free Sources)",
    page_icon="🔎",
    layout="wide",
)

# Subtle modern styling
st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(1000px 600px at 100% -10%, #0ea5e940 0, transparent 60%),
                    radial-gradient(800px 500px at -10% 0%, #22c55e40 0, transparent 50%),
                    linear-gradient(180deg, #0b1220 0%, #0b1220 100%);
        color: #e2e8f0;
      }
      .block-container {max-width: 1200px;}
      h1, h2, h3 { color: #e5e7eb; }
      .card { border-radius: 16px; padding: 14px 16px; background: #111827; border: 1px solid #1f2937; box-shadow: 0 4px 20px rgba(0,0,0,0.25);} 
      .chip { display:inline-block; padding:4px 10px; border-radius:999px; background:#0ea5e9; color:#0b1220; font-weight:600; font-size:12px; margin-right:6px; }
      .muted { color:#94a3b8; }
      a { color:#93c5fd; }
      .source-title { font-weight:700; font-size:15px; }
      .small { font-size:12px; }
      .kbd { background:#0f172a; border:1px solid #1f2937; border-bottom-color:#0ea5e9; padding:2px 6px; border-radius:6px; font-variant-numeric:tabular-nums; }
      .hr { border-top:1px solid #1f2937; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data models
# -----------------------------
@dataclass
class Hit:
    title: str
    url: str
    snippet: str

@dataclass
class Page:
    url: str
    title: str
    text: str
    snippet: str

# -----------------------------
# Utilities
# -----------------------------
STOPWORDS = {
    'a','an','the','and','or','but','if','while','with','to','of','in','on','for','from','by','at','as','is','are','was','were','be','been','it','this','that','these','those','i','you','he','she','they','we','me','him','her','them','my','your','his','their','our','so','than','too','very','can','could','should','would','may','might','will','just','not','no','yes','do','does','did','have','has','had','about','into','over','after','before','between','during','above','below','up','down','out','off','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','s','t','d','ll','m','o','re','ve','y','don','shouldn','now'
}

PAYWALL_DOMAINS = {
    'nytimes.com','wsj.com','ft.com','bloomberg.com','economist.com','washingtonpost.com','foreignaffairs.com'
}

TIMELIMIT_MAP = {
    'Any time': None,
    'Past day': 'd',
    'Past week': 'w',
    'Past month': 'm',
    'Past year': 'y',
}


def domain_of(url: str) -> str:
    try:
        if 'tldextract' in globals() and tldextract:  # precise
            ext = tldextract.extract(url)
            return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        return urlparse(url).netloc.replace('www.', '')
    except Exception:
        return url


def is_probably_english(text: str) -> bool:
    if not text:
        return True
    with contextlib.suppress(Exception):
        if detect:
            lang = detect(text[:1000])
            return lang == 'en'
    # naive fallback: check ascii ratio
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(1, len(text)) > 0.85


async def translate_to_english(text: str, client: httpx.AsyncClient, max_chunk: int = 1800) -> str:
    """Translate text to English using free public services with chunking.
    Tries LibreTranslate instances, then falls back to MyMemory. Returns original text on failure.
    """
    if not text or is_probably_english(text):
        return text

    def _chunks(sentences: List[str], max_len: int) -> List[str]:
        bucket, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) + 1 > max_len:
                if cur:
                    bucket.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            bucket.append(cur)
        return bucket

    sents = sent_split(clean_text(text))
    parts = _chunks(sents if sents else [text], max_chunk)

    async def _try_libre(chunk: str, endpoint: str) -> Optional[str]:
        try:
            r = await client.post(endpoint, json={"q": chunk, "source": "auto", "target": "en", "format": "text"}, timeout=45)
            if r.status_code == 200:
                return r.json().get("translatedText")
        except Exception:
            return None
        return None

    async def _try_mymemory(chunk: str) -> Optional[str]:
        try:
            r = await client.get("https://api.mymemory.translated.net/get", params={"q": chunk, "langpair": "auto|en"}, timeout=45)
            if r.status_code == 200:
                data = r.json()
                return (data.get("responseData") or {}).get("translatedText")
        except Exception:
            return None
        return None

    translated_parts: List[str] = []
    for part in parts:
        out = None
        # Try multiple LibreTranslate community instances
        for ep in ("https://libretranslate.de/translate", "https://translate.astian.org/translate", "https://translate.plausiblet.io/translate"):
            out = await _try_libre(part, ep)
            if out:
                break
        if not out:
            out = await _try_mymemory(part)
        translated_parts.append(out or part)

    return " ".join(translated_parts) or text  # fallback


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sent_split(text: str) -> List[str]:
    # Simple sentence splitter with basic punctuation cues
    text = re.sub(r"([.!?])\s+(?=[A-Z0-9])", r"\1\n", text)
    sents = [s.strip() for s in text.split("\n") if len(s.strip()) > 0]
    # Filter extremely short fragments
    return [s for s in sents if len(s.split()) >= 5]


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower()) if w not in STOPWORDS]


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    denom = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return num / denom if denom else 0.0


def tf_idf_vectors(sents: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    docs = [Counter(tokenize(s)) for s in sents]
    df = Counter()
    for d in docs:
        df.update(d.keys())
    N = len(docs)
    idf = {t: math.log((N + 1) / (1 + df[t])) + 1 for t in df}
    vecs = []
    for d in docs:
        vec = {t: (d[t] / sum(d.values())) * idf[t] for t in d}
        vecs.append(vec)
    return vecs, idf


def mmr_select(sents: List[str], k: int = 8, diversity: float = 0.3) -> List[str]:
    if not sents:
        return []
    vecs, _ = tf_idf_vectors(sents)
    # Query vector approximated as centroid
    q = Counter()
    for v in vecs:
        for t, w in v.items():
            q[t] += w
    q = {t: w / max(1, len(vecs)) for t, w in q.items()}

    chosen = []
    chosen_idx = set()
    while len(chosen) < min(k, len(sents)):
        best_score, best_i = -1e9, None
        for i, v in enumerate(vecs):
            if i in chosen_idx:
                continue
            relevance = cosine(v, q)
            redundancy = max((cosine(v, vecs[j]) for j in chosen_idx), default=0.0)
            score = (1 - diversity) * relevance - diversity * redundancy
            if score > best_score:
                best_score, best_i = score, i
        chosen_idx.add(best_i)
        chosen.append(sents[best_i])
    # Preserve original order for readability
    order = {s: i for i, s in enumerate(sents)}
    chosen.sort(key=lambda s: order.get(s, 0))
    return chosen


def summarize(text: str, max_sentences: int = 8) -> str:
    text = clean_text(text)
    sents = sent_split(text)
    if len(sents) <= max_sentences:
        return textwrap.fill(text, 100)
    picked = mmr_select(sents, k=max_sentences, diversity=0.35)
    return " ".join(picked)


def extract_from_html(html_text: str) -> Tuple[str, str, str]:
    """Return (title, main_text, snippet)."""
    if not html_text:
        return "", "", ""
    title = ""
    try:
        if 'Document' in globals():  # readability available
            doc = Document(html_text)
            title = doc.short_title() or ""
            content_html = doc.summary(html_partial=True)
            soup = BeautifulSoup(content_html, "html.parser")
        else:
            only_html = SoupStrainer(["title", "p", "h1", "h2", "h3", "article", "li"])  
            soup = BeautifulSoup(html_text, "html.parser", parse_only=only_html)
            if soup.title:
                title = soup.title.get_text(strip=True)
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

    # Heuristic: collect paragraphs
    parts = []
    for tag in soup.find_all(["article", "section", "p", "li"]):
        txt = tag.get_text(" ", strip=True)
        if len(txt.split()) >= 6:
            parts.append(txt)
    text = clean_text(" ".join(parts))
    snippet = " ".join(text.split()[:40]) + ("…" if len(text.split()) > 40 else "")
    return title, text, snippet


@st.cache_data(show_spinner=False, ttl=60*60)
def ddg_search(query: str, max_results: int = 8, timelimit_code: Optional[str] = None) -> List[Hit]:
    if DDGS is None:
        st.error(
            "duckduckgo-search is not installed. Run: `pip install duckduckgo-search` and restart the app."
        )
        return []
    hits: List[Hit] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, safesearch="moderate", timelimit=timelimit_code, max_results=max_results):
                url = r.get("href") or r.get("url")
                title = (r.get("title") or r.get("source")) or url
                snippet = r.get("body") or ""
                if not url:
                    continue
                dom = domain_of(url)
                if dom in PAYWALL_DOMAINS:
                    continue
                hits.append(Hit(title=title, url=url, snippet=snippet))
    except Exception as e:
        st.warning(f"Search failed: {e}")
    return hits


async def fetch_html(url: str, client: httpx.AsyncClient) -> str:
    try:
        r = await client.get(url, timeout=20, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36"
        })
        if r.status_code >= 400:
            return ""
        if 'text/html' not in r.headers.get('Content-Type', ''):
            return ""
        return r.text
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=60*60)
def gather_pages(hits: List[Hit], translate: bool = True, english_only: bool = False) -> List[Page]:
    async def _run() -> List[Page]:
        pages: List[Page] = []
        async with httpx.AsyncClient() as client:
            htmls = await asyncio.gather(*[fetch_html(h.url, client) for h in hits])
            for h, html_text in zip(hits, htmls):
                if not html_text:
                    continue
                title, text, snippet = extract_from_html(html_text)
                if not text or len(text.split()) < 80:
                    continue
                # Optionally filter out non-English sources early
                if english_only and not is_probably_english(text):
                    continue
                if translate:
                    text = await translate_to_english(text, client)
                    title = await translate_to_english(title, client)
                    snippet = await translate_to_english(snippet, client)
                pages.append(Page(url=h.url, title=title or h.title, text=text, snippet=snippet or h.snippet))
        return pages

    return asyncio.run(_run())


def make_answer(pages: List[Page], query: str, max_sentences: int = 10) -> str:
    corpus = []
    for p in pages:
        text = p.text
        words = text.split()
        if len(words) > 1200:
            text = " ".join(words[:1200])
        corpus.append(text)
    merged = "\n\n".join(corpus)
    base_summary = summarize(merged, max_sentences=max_sentences)

    q_tokens = tokenize(query)
    if q_tokens:
        sents = sent_split(base_summary)
        scored = [(sum(1 for t in tokenize(s) if t in q_tokens), s) for s in sents]
        scored.sort(key=lambda x: (-x[0], sents.index(x[1])))
        base_summary = " ".join([s for _, s in scored])

    return base_summary


def export_markdown(query: str, answer: str, pages: List[Page]) -> str:
    lines = [f"# Research: {query}", "", "## Summary", "", answer, "", "## Sources"]
    for i, p in enumerate(pages, 1):
        lines += [f"{i}. [{p.title}]({p.url}) — {p.snippet}"]
    md = "\n".join(lines)
    return md


# -----------------------------
# UI
# -----------------------------

def header_ui():
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown("<h1>🔎 Deep Research (Free)</h1>", unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Ask anything. I\'ll search multiple free sources, show quick previews, and synthesize a concise answer in English with citations.</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<div style='text-align:right'><span class='chip'>Free sources only</span> <span class='chip'>No API keys</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


def sidebar_ui():
    st.sidebar.header("Settings")
    max_sources = st.sidebar.slider("Max sources to use", 3, 15, 8)
    timelimit = st.sidebar.selectbox("Time range", list(TIMELIMIT_MAP.keys()), index=0)
    translate = st.sidebar.toggle("Force English translation", value=True, help="Uses LibreTranslate/MyMemory (free public instances).")
    prefer_english = st.sidebar.toggle("Prefer English sources only", value=False)
    summary_len = st.sidebar.select_slider("Summary length (sentences)", options=[6,8,10,12,14], value=10)
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: refine your query with specific entities, time ranges, or filetypes (e.g., 'site:who.int vaccination 2023').")
    return max_sources, timelimit, translate, summary_len, prefer_english


header_ui()
max_sources, timelimit_label, do_translate, summary_len, prefer_english = sidebar_ui()

query = st.text_input("What do you want to research?", placeholder="e.g., How do heat pumps compare to gas boilers for home heating in Europe?", help="Enter a topic, question, or comparison.")

col_a, col_b = st.columns([0.2, 0.8])
with col_a:
    go = st.button("Research", type="primary", use_container_width=True)
with col_b:
    st.caption("The app will search the web, fetch top sources, show previews, and synthesize an answer.")

if go and query.strip():
    with st.spinner("Searching the web..."):
        hits = ddg_search(query.strip(), max_results=max_sources, timelimit_code=TIMELIMIT_MAP[timelimit_label])

    if not hits:
        st.warning("No results found or search library missing. Please adjust your query or install requirements.")
        st.stop()

    with st.spinner("Fetching and analyzing sources…"):
        pages = gather_pages(hits, translate=do_translate, english_only=prefer_english)

    if not pages:
        st.warning("Couldn't extract enough content from the sources. Try a different query or time range.")
        st.stop()

    # Glimpses
    st.subheader("Source previews")
    for i, p in enumerate(pages, 1):
        dom = domain_of(p.url)
        with st.container(border=True):
            st.markdown(f"<div class='small muted'>Source {i} · {html.escape(dom)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='source-title'>{html.escape(p.title or dom)}</div>", unsafe_allow_html=True)
            st.write(p.snippet)
            with st.expander("Show more"):
                st.write(textwrap.shorten(p.text, width=1500, placeholder="…"))
            st.markdown(f"[Open source]({p.url})")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Summary
    with st.spinner("Synthesizing the answer…"):
        answer = make_answer(pages, query, max_sentences=summary_len)
        # Ensure final answer is English when requested
        if do_translate and not is_probably_english(answer):
            async def _translate_answer():
                async with httpx.AsyncClient() as client:
                    return await translate_to_english(answer, client)
            try:
                answer = asyncio.run(_translate_answer())
            except RuntimeError:
                # Fallback if event loop already running
                answer = answer

    st.subheader("Answer (English)")
    st.write(answer)

    # Citations
    st.subheader("Citations")
    for i, p in enumerate(pages, 1):
        st.markdown(f"{i}. [{p.title}]({p.url})")

    # Export
    md = export_markdown(query, answer, pages)
    st.download_button("Download report (Markdown)", data=md.encode("utf-8"), file_name="research_report.md", mime="text/markdown")

else:
    st.caption("Enter a query and click **Research** to begin.")

# Footer / About
with st.expander("About this app"):
    st.markdown(
        """
        - **Free sources only**: uses DuckDuckGo search and filters common paywalls.
        - **No API keys needed**: optional translation via public LibreTranslate or MyMemory.
        - **Summarization**: custom extractive summarizer (TF‑IDF + MMR) for fast, local synthesis.
        - **Notes**: Some sites block scraping or serve non-HTML; such pages are skipped.
        - **Tip**: Turn on *Prefer English sources only* if auto-translation is slow or rate-limited.
         for fast, local synthesis.
        - **Notes**: Some sites block scraping or serve non-HTML; such pages are skipped.
        """
    )
