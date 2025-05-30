# import streamlit as st
# import requests
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv


# # Initialize Gemini API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model = genai.GenerativeModel("gemini-1.5-flash")

# # SerpAPI key

# SERP_API_KEY =  os.getenv("SERP_API_KEY")

# # Search Function
# def search_google(query, api_key, num_results=5):
#     url = "https://serpapi.com/search"
#     params = {
#         "engine": "google",
#         "q": query,
#         "api_key": api_key,
#         "num": num_results
#     }
#     response = requests.get(url, params=params)
#     if response.status_code != 200:
#         return []

#     results = response.json()
#     extracted = []
#     for r in results.get("organic_results", []):
#         title = r.get("title", "")
#         snippet = r.get("snippet", "")
#         link = r.get("link", "")
#         text = f"{title}. {snippet}. Link: {link}"
#         extracted.append(text)
#     return extracted

# # Generate Gemini Response
# def get_gemini_summary(clean_texts, question):
#     if not clean_texts:
#         return "No relevant search results found."

#     context = "\n\n".join(clean_texts)

#     prompt = f"""
#             You are an academic assistant tasked with answering questions based on the retrieved documents. The context is related to NIT Jalandhar.

#             Your response should be clear, accurate, and structured. Use bullet points or short paragraphs where appropriate.

#             Instructions:
#             - Analyze the user's question carefully.
#             - Use only the information from the context to answer.
#             - If the answer cannot be found in the documents, respond with "The answer is not available in the provided context."

#             User question:
#             {question}

#             Relevant documents:
#             {context}

#             Answer:
#             """
#     response = model.generate_content(prompt)
#     return response.text

# # Streamlit UI
# st.title("Academic Info Extractor")
# query = st.text_input("Enter your search query (e.g., Dr Kunwar Pal NIT Jalandhar):")

# if st.button("Search and Summarize"):
#     if not query:
#         st.warning("Please enter a query first.")
#     else:
#         with st.spinner("Searching and generating summary..."):
#             results = search_google(f"{query} site:nitj.ac.in", SERP_API_KEY)
#             summary = get_gemini_summary(results, query)
#             st.subheader("Generated Summary")
#             st.write(summary)

#             st.subheader("Raw Search Results")
#             for r in results:
#                 st.markdown(f"- {r}")




import streamlit as st
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import mimetypes
import fitz  # PyMuPDF
from utils import extract_page_text



# Load API keys from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Helper to extract page content
def extract_page_text(url, max_chars=1000):
    try:
        if url.endswith(".pdf") or "pdf" in mimetypes.guess_type(url)[0]:
            # Fetch and read PDF
            response = requests.get(url)
            with open("temp.pdf", "wb") as f:
                f.write(response.content)

            doc = fitz.open("temp.pdf")
            text = ""
            for page in doc:
                text += page.get_text()
                if len(text) >= max_chars:
                    break
            doc.close()
            return text[:max_chars] if text else "(Could not extract readable text from PDF)"
        
        # Fallback to HTML
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = ' '.join(soup.stripped_strings)
        return text[:max_chars]
    
    except Exception as e:
        return f"(Failed to fetch content from {url}: {e})"
# Google search using SerpAPI
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
        full_text = extract_page_text(link)
        text = f"{title}\nSnippet: {snippet}\nLink: {link}\nPage Extract: {full_text}"
        extracted.append(text)
    return extracted

# Gemini summarizer
def get_gemini_summary(clean_texts, question):
    if not clean_texts:
        return "No relevant search results found."

    context = "\n\n".join(clean_texts)
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
st.title("🎓 NITJ Academic Info Extractor")
query = st.text_input("Enter your search query (e.g., Dr Kunwar Pal NIT Jalandhar):")

if st.button("Search and Summarize"):
    if not query:
        st.warning("Please enter a query first.")
    else:
        with st.spinner("🔍 Searching and generating summary..."):
            results = search_google(f"{query} site:nitj.ac.in", SERP_API_KEY)
            summary = get_gemini_summary(results, query)
            
            st.subheader("📄 Generated Summary")
            st.write(summary)

            st.subheader("🔎 Raw Search Results + Extracted Content")
            for r in results:
                st.markdown(f"---\n{r}")
