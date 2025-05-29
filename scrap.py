import streamlit as st
import requests
import google.generativeai as genai

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyCExbTXGgui6VDZqGf5iId-RWOwmgxhwL0"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# SerpAPI key
SERP_API_KEY = "ebebc7f891a20ef1e7baffb155f26de799ddabb9c93219108480d89075e4ae53"

# Search Function
def search_google(query, api_key, num_results=5):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    results = response.json()
    extracted = []
    for r in results.get("organic_results", []):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        text = f"{title}. {snippet}. Link: {link}"
        extracted.append(text)
    return extracted

# Generate Gemini Response
def get_gemini_summary(clean_texts):
    if not clean_texts:
        return "No relevant search results found."

    prompt = (
        "You are an academic assistant tasked with generating a concise and informative summary based on search results.\n\n"
        "The goal is to extract key professional information about the individual mentioned in the search query. Your summary should include:\n"
        "- Full name and current designation\n"
        "- Institution/department affiliation\n"
        "- Academic or research background (e.g., Ph.D., research areas)\n"
        "- Links to official profiles or Google Scholar if available\n\n"
        "Present the answer in a clean, well-structured format using bullet points or short paragraphs where helpful. Do not include raw URLs unless linking to official academic profiles.\n\n"
        "Here are the search results:\n\n"
    )

    prompt += "\n\n".join(clean_texts)
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Academic Info Extractor")
query = st.text_input("Enter your search query (e.g., Dr Kunwar Pal NIT Jalandhar):")

if st.button("Search and Summarize"):
    if not query:
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Searching and generating summary..."):
            results = search_google(f"{query} site:nitj.ac.in", SERP_API_KEY)
            summary = get_gemini_summary(results)
            st.subheader("Generated Summary")
            st.write(summary)

            st.subheader("Raw Search Results")
            for r in results:
                st.markdown(f"- {r}")
