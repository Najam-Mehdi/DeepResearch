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
import time
import random
from transformers import pipeline

# --- NLTK Data Download ---
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.info("Downloading necessary NLTK data for the first run...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        st.success("NLTK data downloaded successfully!")

download_nltk_data()

# --- Transformer Summarization Model ---
@st.cache_resource
def load_summarization_model():
    """Loads a pre-trained summarization model from Hugging Face."""
    st.info("Loading text summarization model...")
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        st.success("Summarization model loaded successfully!")
        return summarizer
    except Exception as e:
        st.error(f"Failed to load summarization model: {e}")
        st.warning("Falling back to NLTK-based summarizer.")
        return None

summarizer_pipeline = load_summarization_model()

# --- Core Research Functions ---
def search_google(query, num_results):
    """
    Performs a Google search and returns URLs.
    """
    st.write(f"Searching for: '{query}'...")
    try:
        ## Add a random delay to avoid being blocked
        #time.sleep(time.uniform(1.5, 3.0))
        # Corrected line to use random.uniform()
        time.sleep(random.uniform(1.5, 3.0))       
        return list(search(query, num_results=num_results, lang='en'))
    except Exception as e:
        st.error(f"An error occurred during search: {e}")
        st.warning("Google may have temporarily blocked this IP due to too many requests. Please try again later.")
        return []

def scrape_content(url, progress_bar):
    """
    Scrapes textual content from a URL with robust error handling.
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

        text = soup.get_text(separator=' ', strip=True)
        # Simplified and robust text cleaning
        text = re.sub(r'\s+', ' ', text)
        return text
    except requests.exceptions.RequestException as e:
        progress_bar.warning(f"Could not scrape {url}: {e}")
        return None
    except Exception as e:
        progress_bar.error(f"An error occurred while processing {url}: {e}")
        return None

def summarize_text_transformer(text, max_length=150, min_length=40):
    """
    Generates a summary using a Hugging Face Transformer model.
    """
    if not text:
        return ""
    try:
        summary = summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization model failed: {e}. Falling back to NLTK.")
        return summarize_text_nltk(text)

def summarize_text_nltk(text, num_sentences=5):
    """
    Generates a summary using frequency-based analysis (NLTK fallback).
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
                if len(sentence.split(' ')) < 35:
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

# User controls
query = st.text_input("Enter your research query:", placeholder="e.g., latest advancements in AI")
num_sources = st.slider("Number of sources to search:", 5, 20, 10)
if st.button("Start Research"):
    if not query:
        st.warning("Please enter a query to start the research.")
    else:
        # --- Step 1: Searching ---
        with st.spinner("Step 1: Searching for online sources..."):
            urls = search_google(query, num_results=num_sources)
        
        if not urls:
            st.error("No search results found. Please try a different query.")
        else:
            st.success(f"Found {len(urls)} potential sources.")
            
            # --- Step 2: Scraping and Processing ---
            scraped_sources = {}
            report_text = f"Research Report for: {query}\n\n"
            citations = []
            
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            
            st.subheader("Progress:")
            
            for i, url in enumerate(urls):
                with st.empty():
                    progress_text = f"Scraping and summarizing source {i+1} of {len(urls)}: {url}"
                    st.write(progress_text)
                    
                content = scrape_content(url, st.container())
                if content:
                    summary = summarize_text_transformer(content) if summarizer_pipeline else summarize_text_nltk(content)
                    if summary:
                        scraped_sources[url] = summary
                        citations.append(f"[{len(scraped_sources)}] {url}")
                        report_text += f"Summary [{len(scraped_sources)}]:\n{summary}\n\n"
                progress_bar.progress((i + 1) / len(urls))
            
            progress_container.empty()
            st.success("All available sources processed!")
            
            if not scraped_sources:
                st.error("Could not retrieve content from any of the search results. This might be due to anti-scraping measures on the websites.")
            else:
                # --- Step 3: Generating and Displaying Report ---
                st.header("Research Report")
                st.subheader(f"Query: {query}")
                
                for i, (url, summary) in enumerate(scraped_sources.items()):
                    st.markdown(f"> {summary} **[{i+1}]**")
                
                st.subheader("Sources and Citations")
                for citation in citations:
                    st.markdown(f"- {citation}")
                    
                # Add a download button
                report_text += "--- Sources ---\n" + '\n'.join(citations)
                st.download_button(
                    label="Download Report as Text",
                    data=report_text,
                    file_name=f"{query.replace(' ', '_')}_report.txt",
                    mime="text/plain"
                )
                
                st.success("Report generation complete!")
                st.balloons()
