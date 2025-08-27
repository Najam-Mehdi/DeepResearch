# app.py
# A Streamlit web application for the Deep Research Tool.

import streamlit as st
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import re
import time

# --- NLTK Data Download ---
# This is a one-time setup for NLTK data.
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        st.info("Downloading necessary NLTK data for the first run...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        st.success("NLTK data downloaded successfully!")

download_nltk_data()

# --- Core Research Functions ---

def search_google(query, num_results=10):
    """
    Performs a Google search and returns URLs.
    """
    st.write(f"Searching for: '{query}'...")
    try:
        # Add a small delay to avoid being blocked
        time.sleep(2)
        return list(search(query, num_results=num_results, lang='en'))
    except Exception as e:
        st.error(f"An error occurred during search: {e}")
        st.warning("Google may have temporarily blocked this IP due to too many requests. Please try again later.")
        return []

def scrape_content(url, progress_bar):
    """
    Scrapes textual content from a URL.
    """
    progress_bar.write(f"Scraping: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,-]', '', text)
        return text
    except requests.exceptions.RequestException as e:
        progress_bar.warning(f"Could not scrape {url}: {e}")
        return None
    except Exception as e:
        progress_bar.error(f"An error occurred while processing {url}: {e}")
        return None

def summarize_text(text, num_sentences=5):
    """
    Generates a summary of the given text using frequency-based analysis.
    """
    if not text:
        return ""
        
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq = FreqDist(filtered_words)
    
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if len(sentence.split(' ')) < 35: # Penalize very long sentences
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = freq[word]
                    else:
                        sentence_scores[sentence] += freq[word]

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return ' '.join(summary_sentences)

# --- Streamlit App UI ---

st.set_page_config(page_title="Deep Research Tool", layout="wide")
st.title("🤖 Deep Research Tool")
st.markdown("Enter a query to get a detailed report with citations from online sources.")

query = st.text_input("Enter your research query:", placeholder="e.g., latest advancements in AI")

if st.button("Start Research"):
    if not query:
        st.warning("Please enter a query to start the research.")
    else:
        # --- Step 1: Searching ---
        with st.spinner("Step 1: Searching for online sources..."):
            urls = search_google(query)
        
        if not urls:
            st.error("No search results found. Please try a different query.")
        else:
            st.success(f"Found {len(urls)} potential sources.")
            
            # --- Step 2: Scraping and Processing ---
            scraped_sources = {}
            progress_container = st.expander("Show Scraping Progress", expanded=True)
            
            with st.spinner("Step 2: Scraping and summarizing content..."):
                for i, url in enumerate(urls):
                    content = scrape_content(url, progress_container)
                    if content:
                        scraped_sources[url] = content
            
            if not scraped_sources:
                st.error("Could not retrieve content from any of the search results. This might be due to anti-scraping measures on the websites.")
            else:
                # --- Step 3: Generating Report ---
                with st.spinner("Step 3: Compiling the research report..."):
                    st.header("Research Report")
                    st.subheader(f"Query: {query}")
                    
                    report_content = ""
                    citations = []
                    
                    st.subheader("Summary of Findings")
                    for i, (url, content) in enumerate(scraped_sources.items()):
                        summary = summarize_text(content, num_sentences=3)
                        if summary:
                            st.markdown(f"> {summary} **[{i+1}]**")
                            citations.append(f"[{i+1}] {url}")
                    
                    st.subheader("Sources and Citations")
                    for citation in citations:
                        st.markdown(f"- {citation}")
                
                st.success("Report generation complete!")
                st.balloons()
