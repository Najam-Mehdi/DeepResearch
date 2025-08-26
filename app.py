"""
Free Multi‑Source Web Search (Streamlit)
=======================================

A fully free, API‑key‑optional research app that queries multiple open sources
concurrently and returns organized, cited results — with an optional
extractive summary.

Sources implemented (all free/open):
- DuckDuckGo Web (via `duckduckgo_search` library)
- DuckDuckGo Instant Answer API (JSON)
- Wikipedia (MediaWiki API)
- SearXNG (self‑hosted or public instance, JSON endpoint)
- arXiv API (Atom)
- Crossref Works API (JSON)
- Internet Archive Advanced Search (JSON)
- Wikidata Entity Search API (JSON)

Notes
-----
- No paid subscriptions required. All endpoints are free. SearXNG works best
  with your own instance URL (public instances may rate limit).
- Summarization uses a simple extractive method (no model, no external API).
- Results are grouped by source and downloadable as JSON.

Author: You
License: MIT
"""

# =========================
#          SETUP
# =========================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from functools import lru_cache
import html
import json
import re
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st
import httpx
from bs4 import BeautifulSoup
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
    """Normalized search result structure.

    Attributes
    ----------
    title : str
        Human‑readable title of the result.
    snippet : str
        Short summary/snippet (plain text preferred).
    url : str
        Canonical URL for the result.
    source : str
        Name of the provider (e.g., "DuckDuckGo", "Wikipedia").
    extra : dict
        Provider‑specific fields (optional).
    """

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
    """Remove HTML tags/entities; return trimmed plain text."""
    if not text:
        return ""
    # Unescape HTML entities first
    text = html.unescape(text)
    # Quick strip of tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate to a reasonable snippet length without breaking words."""
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

async def search_ddg_web(query: str, max_results: int = 6) -> List[SearchResult]:
    """DuckDuckGo web search via `duckduckgo_search` lib (no API key).

    Returns top results with title/body/url.
    """
    def run_ddg() -> List[SearchResult]:
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
            results.append(
                SearchResult(
                    title="DuckDuckGo Error",
                    snippet=str(e),
                    url="",
                    source="DuckDuckGo",
                )
            )
        return results

    return await asyncio.to_thread(run_ddg)


async def search_ddg_instant(query: str) -> List[SearchResult]:
    """DuckDuckGo Instant Answer API (free, no key).

    Docs: https://api.duckduckgo.com/?q=QUERY&format=json
    """
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
            results: List[SearchResult] = []

            # Primary abstract
            abstract = data.get("AbstractText") or data.get("Abstract") or ""
            abstract_url = data.get("AbstractURL") or data.get("Results", [{}])[0].get("FirstURL", "")
            heading = data.get("Heading") or "DuckDuckGo Answer"
            if abstract:
                results.append(
                    SearchResult(
                        title=heading,
                        snippet=_truncate(_clean_html(abstract)),
                        url=abstract_url or "",
                        source="DuckDuckGo Instant Answer",
                    )
                )

            # Related topics (a few top items)
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
                results.append(
                    SearchResult(
                        title="No instant answer",
                        snippet="DuckDuckGo did not return an abstract for this query.",
                        url="",
                        source="DuckDuckGo Instant Answer",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="DuckDuckGo IA Error",
                    snippet=str(e),
                    url="",
                    source="DuckDuckGo Instant Answer",
                )
            ]


async def search_wikipedia(query: str, max_results: int = 5) -> List[SearchResult]:
    """Wikipedia via MediaWiki API (no key).

    We do a search then map to page URLs using pageid.
    """
    API = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(max_results),
        "format": "json",
        "utf8": 1,
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(API, params=params)
            data = resp.json()
            results: List[SearchResult] = []
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "Wikipedia Article")
                snippet_html = item.get("snippet", "")
                pageid = item.get("pageid")
                url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
                results.append(
                    SearchResult(
                        title=title,
                        snippet=_truncate(_clean_html(snippet_html)),
                        url=url,
                        source="Wikipedia",
                        extra={"pageid": pageid},
                    )
                )
            if not results:
                results.append(
                    SearchResult(
                        title="No Wikipedia results",
                        snippet="Try refining your query.",
                        url="",
                        source="Wikipedia",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="Wikipedia Error",
                    snippet=str(e),
                    url="",
                    source="Wikipedia",
                )
            ]


async def search_searxng(query: str, instance_url: str, max_results: int = 6) -> List[SearchResult]:
    """SearXNG metasearch (JSON). Provide an instance base URL, e.g.,
    https://searxng.example.org

    If the instance is missing or unreachable, we return a single error result.
    """
    if not instance_url:
        return [
            SearchResult(
                title="SearXNG not configured",
                snippet="Provide an instance URL in the sidebar to enable SearXNG.",
                url="",
                source="SearXNG",
            )
        ]

    endpoint = instance_url.rstrip("/") + "/search"
    params = {"q": query, "format": "json", "language": "en", "safesearch": 1}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(endpoint, params=params)
            data = resp.json()
            results: List[SearchResult] = []
            for r in (data.get("results") or [])[:max_results]:
                results.append(
                    SearchResult(
                        title=r.get("title") or "Untitled",
                        snippet=_truncate(_clean_html(r.get("content") or "")),
                        url=r.get("url") or "",
                        source="SearXNG",
                        extra={"engine": r.get("engine")},
                    )
                )
            if not results:
                results.append(
                    SearchResult(
                        title="No SearXNG results",
                        snippet="The instance returned no results.",
                        url="",
                        source="SearXNG",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="SearXNG Error",
                    snippet=str(e),
                    url="",
                    source="SearXNG",
                )
            ]


async def search_arxiv(query: str, max_results: int = 5) -> List[SearchResult]:
    """arXiv API (Atom feed). No key required.

    Docs: https://info.arxiv.org/help/api/
    """
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(url, params=params)
            feed = feedparser.parse(resp.text)
            results: List[SearchResult] = []
            for e in feed.entries:
                link = e.get("link") or (e.links[0].href if e.get("links") else "")
                results.append(
                    SearchResult(
                        title=e.get("title", "arXiv Paper").strip(),
                        snippet=_truncate(_clean_html(e.get("summary", "").strip())),
                        url=link,
                        source="arXiv",
                        extra={"published": e.get("published")},
                    )
                )
            if not results:
                results.append(
                    SearchResult(
                        title="No arXiv results",
                        snippet="Try a more technical/academic query.",
                        url="",
                        source="arXiv",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="arXiv Error",
                    snippet=str(e),
                    url="",
                    source="arXiv",
                )
            ]


async def search_crossref(query: str, rows: int = 5) -> List[SearchResult]:
    """Crossref Works API (no key).

    Docs: https://api.crossref.org/works
    """
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
            items = (data.get("message", {}) or {}).get("items", [])
            results: List[SearchResult] = []
            for it in items:
                title = "; ".join(it.get("title", [])).strip() or "Crossref Work"
                url_out = it.get("URL", "")
                abstract = _clean_html(it.get("abstract", ""))
                # Fallback to container-title + author if no abstract
                if not abstract:
                    container = "; ".join(it.get("container-title", [])).strip()
                    authors = ", ".join(
                        [f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", [])][:3]
                    )
                    parts = [p for p in [container, authors] if p]
                    abstract = "; ".join(parts)
                results.append(
                    SearchResult(
                        title=title,
                        snippet=_truncate(abstract or ""),
                        url=url_out,
                        source="Crossref",
                    )
                )
            if not results:
                results.append(
                    SearchResult(
                        title="No Crossref results",
                        snippet="Try a different query or enable arXiv.",
                        url="",
                        source="Crossref",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="Crossref Error",
                    snippet=str(e),
                    url="",
                    source="Crossref",
                )
            ]


async def search_internet_archive(query: str, rows: int = 5) -> List[SearchResult]:
    """Internet Archive Advanced Search (no key).

    Docs: https://archive.org/advancedsearch.php
    """
    url = "https://archive.org/advancedsearch.php"
    params = {"q": query, "output": "json", "rows": rows}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
            docs = (data.get("response", {}) or {}).get("docs", [])
            results: List[SearchResult] = []
            for d in docs:
                identifier = d.get("identifier", "")
                url_out = f"https://archive.org/details/{identifier}" if identifier else ""
                title = d.get("title") or identifier or "Internet Archive Item"
                desc = d.get("description") or ""
                if isinstance(desc, list):
                    desc = " ".join(desc)
                results.append(
                    SearchResult(
                        title=_truncate(_clean_html(title), 100),
                        snippet=_truncate(_clean_html(desc)),
                        url=url_out,
                        source="Internet Archive",
                    )
                )
            if not results:
                results.append(
                    SearchResult(
                        title="No Archive results",
                        snippet="Try a broader query.",
                        url="",
                        source="Internet Archive",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="Internet Archive Error",
                    snippet=str(e),
                    url="",
                    source="Internet Archive",
                )
            ]


async def search_wikidata(query: str, limit: int = 5) -> List[SearchResult]:
    """Wikidata entity search API (no key).

    Docs: https://www.wikidata.org/w/api.php?action=help&modules=wbsearchentities
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": limit,
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
            results: List[SearchResult] = []
            for e in data.get("search", [])[:limit]:
                title = e.get("label") or "Wikidata Entity"
                desc = e.get("description") or ""
                url_out = e.get("url") or (f"https://www.wikidata.org/wiki/{e.get('id')}" if e.get("id") else "")
                results.append(
                    SearchResult(
                        title=title,
                        snippet=_truncate(_clean_html(desc)),
                        url=url_out,
                        source="Wikidata",
                    )
                )
            if not results:
                results.append(
                    SearchResult(
                        title="No Wikidata results",
                        snippet="Try refining your query.",
                        url="",
                        source="Wikidata",
                    )
                )
            return results
        except Exception as e:
            return [
                SearchResult(
                    title="Wikidata Error",
                    snippet=str(e),
                    url="",
                    source="Wikidata",
                )
            ]


# =========================
#      SUMMARY (LOCAL)
# =========================

STOPWORDS = {
    # Short, embedded stopword list (English)
    "the","is","at","of","on","and","a","to","in","for","by","an","be","as","are",
    "that","with","or","from","this","it","its","we","our","you","your","about","into",
}


def extractive_summary(snippets: Iterable[str], max_sentences: int = 5) -> str:
    """Very simple frequency‑based extractive summary.

    Parameters
    ----------
    snippets : Iterable[str]
        Collection of strings to summarize (e.g., result snippets).
    max_sentences : int
        Max number of sentences to return.

    Returns
    -------
    str
        Summary text (<= max_sentences sentences) or a helpful message.
    """
    text = " ".join([s for s in snippets if s])
    text = _clean_html(text)
    if not text:
        return "Not enough content to summarize."

    # Split into sentences (very rough)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return _truncate(" ".join(sentences), 1200)

    # Score words
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    freqs: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1

    if not freqs:
        return _truncate(" ".join(sentences[:max_sentences]), 1200)

    # Score sentences by word frequency
    sent_scores: List[tuple[float, str]] = []
    for s in sentences:
        s_words = re.findall(r"[A-Za-z][A-Za-z\-']+", s.lower())
        score = sum(freqs.get(w, 0) for w in s_words) / (len(s_words) + 1)
        sent_scores.append((score, s))

    # Pick top sentences, preserve original order
    top = sorted(sent_scores, key=lambda x: x[0], reverse=True)[: max_sentences * 3]
    selected = {s for _, s in top}
    ordered = [s for s in sentences if s in selected][:max_sentences]
    return _truncate(" ".join(ordered), 1200)


# =========================
#        CONTROLLERS
# =========================

async def run_selected_sources(
    query: str,
    sources: List[str],
    searxng_url: str,
    max_results: int,
) -> Dict[str, List[SearchResult]]:
    """Run all selected sources concurrently and return grouped results."""
    tasks = []
    for src in sources:
        if src == "DuckDuckGo":
            tasks.append(search_ddg_web(query, max_results))
        elif src == "DuckDuckGo Instant Answer":
            tasks.append(search_ddg_instant(query))
        elif src == "Wikipedia":
            tasks.append(search_wikipedia(query, max_results))
        elif src == "SearXNG":
            tasks.append(search_searxng(query, searxng_url, max_results))
        elif src == "arXiv":
            tasks.append(search_arxiv(query, max_results))
        elif src == "Crossref":
            tasks.append(search_crossref(query, max_results))
        elif src == "Internet Archive":
            tasks.append(search_internet_archive(query, max_results))
        elif src == "Wikidata":
            tasks.append(search_wikidata(query, max_results))

    results_grouped: Dict[str, List[SearchResult]] = {s: [] for s in sources}

    for src, coro in zip(sources, asyncio.as_completed(tasks)):
        # NOTE: We need to await in the order of completion, then map back.
        pass

    # The above zip(as_completed(...)) loses mapping; use gather instead.
    gathered = await asyncio.gather(*tasks)
    for src, res in zip(sources, gathered):
        results_grouped[src] = res

    return results_grouped


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

query = st.text_input(
    "Search topic / question",
    placeholder="e.g. quantum error correction, climate policy 2025, best open data portals",
)

run = st.button("Run search", type="primary")

# Container for results
results_container = st.container()

if run and query.strip():
    with st.spinner("Fetching results from selected sources…"):
        results_by_source = asyncio.run(run_selected_sources(query.strip(), sources, searxng_url.strip(), max_results))

    # ---------------------
    # Display: Results
    # ---------------------
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
                # Collect snippets for summary
                all_for_summary.extend([it.snippet for it in items if it.snippet])

        # ---------------------
        # Display: Summary
        # ---------------------
        if show_summary:
            st.subheader("📝 Consolidated Summary (extractive)")
            st.write(extractive_summary(all_for_summary, max_sentences=6))

        # ---------------------
        # Download JSON
        # ---------------------
        export: Dict[str, List[Dict[str, Any]]] = {
            src: [asdict(it) for it in results_by_source.get(src, [])] for src in sources
        }
        st.download_button(
            label="⬇️ Download Results (JSON)",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name="search_results.json",
            mime="application/json",
        )

else:
    st.info("Enter a query and click **Run search** to get started.")

# =========================
#            END
# =========================
