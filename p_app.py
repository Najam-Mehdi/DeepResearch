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
