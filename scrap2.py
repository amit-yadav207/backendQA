import streamlit as st
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from utils import extract_page_text

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Search Function using SerpAPI
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
def get_gemini_summary(contexts, question):
    if not contexts:
        return "No relevant search results found."

    context = "\n\n".join(contexts)

    prompt = f"""
You are an academic assistant tasked with answering questions based on the retrieved documents. The context is related to NIT Jalandhar.

Your response should be clear, accurate, and structured. Use bullet points or short paragraphs where appropriate.

Instructions:
- Analyze the user's question carefully.
- Use only the information from the context to answer.
- If the answer cannot be found in the documents, respond with "The answer is not available in the provided context."

User question:
{question}

Relevant documents:
{context}

Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📚 NITJ Academic Info Assistant")
query = st.text_input("Enter your search query (e.g., Dr Kunwar Pal NIT Jalandhar):")

if st.button("Search and Summarize"):
    if not query:
        st.warning("Please enter a query first.")
    else:
        with st.spinner("🔍 Searching and generating summary..."):
            results = search_google(f"{query} site:nitj.ac.in", SERP_API_KEY)
            detailed_contexts = []

            st.subheader("🔎 Raw Search Results + Extracted Content")
            for r in results:
                title_snippet, link = r.split("Link: ")
                st.markdown(f"**{title_snippet.strip()}**")
                st.markdown(f"🔗 {link.strip()}")

                content = extract_page_text(link.strip())
                st.markdown(f"📄 Page Extract: {content}")
                detailed_contexts.append(content)
                st.markdown("---")

            # Summarize
            summary = get_gemini_summary(detailed_contexts, query)
            st.subheader("📄 Generated Summary")
            st.write(summary)
