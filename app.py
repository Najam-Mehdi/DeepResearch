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
from urllib.parse import urlparse

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


async def translate_to_english(text: str, client: httpx.AsyncClient) -> str:
    """Translate text to English using LibreTranslate public instances (no key)."""
    if not text or is_probably_english(text):
        return text
    payload = {"q": text, "source": "auto", "target": "en", "format": "text"}
    # Try a couple of common public instances
    for endpoint in ("https://libretranslate.de/translate", "https://translate.astian.org/translate"):
        try:
            r = await client.post(endpoint, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data.get("translatedText", text)
        except Exception:
            continue
    return text  # fallback


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
def gather_pages(hits: List[Hit], translate: bool = True) -> List[Page]:
    async def _run() -> List[Page]:
        pages: List[Page] = []
        async with httpx.AsyncClient() as client:
            htmls = await asyncio.gather(*[fetch_html(h.url, client) for h in hits])
            for h, html_text in zip(hits, htmls):
                if not html_text:
                    continue
                title, text, snippet = extract_from_html(html_text)
                if not text or len(text.split()) < 80:
                    # Skip very short contents
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
        # Light normalization
        text = p.text
        # Trim extremely long pages per-source to avoid skew
        words = text.split()
        if len(words) > 1200:
            text = " ".join(words[:1200])
        corpus.append(text)
    merged = "\n\n".join(corpus)
    base_summary = summarize(merged, max_sentences=max_sentences)

    # Add a short, focused finalization step based on the query terms
    q_tokens = tokenize(query)
    if q_tokens:
        # Slightly bias towards sentences containing query terms
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
    translate = st.sidebar.toggle("Force English translation", value=True, help="Uses LibreTranslate (free public instances).")
    summary_len = st.sidebar.select_slider("Summary length (sentences)", options=[6,8,10,12,14], value=10)
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: refine your query with specific entities, time ranges, or filetypes (e.g., 'site:who.int vaccination 2023').")
    return max_sources, timelimit, translate, summary_len


header_ui()
max_sources, timelimit_label, do_translate, summary_len = sidebar_ui()

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
        pages = gather_pages(hits, translate=do_translate)

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
        - **No API keys needed**: optional translation via public LibreTranslate instances.
        - **Summarization**: custom extractive summarizer (TF‑IDF + MMR) for fast, local synthesis.
        - **Notes**: Some sites block scraping or serve non-HTML; such pages are skipped.
        """
    )
