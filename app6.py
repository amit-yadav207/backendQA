import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Remove after recreating environment
import re
import socket
import os.path as ospath
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from logging.handlers import RotatingFileHandler
from typing import List
import google.generativeai as genai
import uvicorn
import speech_recognition as sr
import tempfile
import json
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
import mimetypes
import fitz  # PyMuPDF
from langchain.schema import Document
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options



# Configure logging
logger = logging.getLogger('NITJChatbot')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('chatbot.log', maxBytes=1_000_000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Starting NITJ AI Chatbot backend")

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
logger.debug("Google API key configured")

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Utility Functions ---------------------------

def clean_text(text):
    logger.debug("Cleaning text input")
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    logger.debug("Text cleaned successfully")
    return cleaned


def check_connectivity():
    logger.info("Checking internet connectivity")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logger.info("Internet connectivity confirmed")
        return True
    except OSError as e:
        logger.error(f"Connectivity check failed: {e}")
        return False

def faiss_index_exists():
    logger.debug("Checking if FAISS index exists")
    exists = ospath.exists("faiss_index") and ospath.isfile(ospath.join("faiss_index", "index.faiss"))
    logger.debug(f"FAISS index exists: {exists}")
    return exists

# --------------------------- PDF Processing ----------------------------

def get_pdf_text(pdf_docs: List[UploadFile]):
    logger.info("Extracting text from %d PDF documents", len(pdf_docs))
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
            logger.debug("Text extracted from PDF: %s", pdf.filename)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf.filename}: {e}")
    cleaned_text = clean_text(text)
    logger.info("PDF text extraction completed, length: %d characters", len(cleaned_text))
    return cleaned_text

def get_text_chunks(text):
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=600)
    chunks = text_splitter.split_text(text)
    logger.debug("Text split into %d chunks", len(chunks))
    return chunks

def get_vector_store(text_chunks):
    logger.info("Creating FAISS vector store with %d text chunks", len(text_chunks))
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logger.info("FAISS vector store created and saved to 'faiss_index'")
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise


# --------------------------- Fall Back Function ----------------------------

# def extract_clean_text_from_html(html):
#     soup = BeautifulSoup(html, "html.parser")

#     # Try to extract meaningful sections if present
#     main_content = soup.find("div", class_="container") or soup.body

#     if main_content:
#         text = main_content.get_text(separator=" ", strip=True)
#     else:
#         text = soup.get_text(separator=" ", strip=True)

#     # Remove extra whitespace and shorten long outputs
#     return " ".join(text.split())



def extract_content_with_selenium(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(10)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    finally:
        driver.quit()

def extract_content_from_url(url):
    logger.info(f"Attempting to extract content from URL: {url}")
    try:
        if url.endswith(".pdf"):
            logger.info("URL points to a PDF. Downloading...")
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("PDF downloaded successfully.")
                with open("temp.pdf", "wb") as f:
                    f.write(response.content)
                try:
                    logger.info("Attempting to extract text directly from PDF...")
                    text = extract_text("temp.pdf")
                    logger.info("Text extraction from PDF succeeded.")
                    return text
                except Exception as e:
                    logger.warning(f"Direct PDF text extraction failed: {e}")
                    logger.info("Falling back to OCR extraction using pdf2image and pytesseract...")

                    from pdf2image import convert_from_path
                    from PIL import Image
                    import pytesseract

                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    images = convert_from_path("temp.pdf", poppler_path=r"C:\poppler-24.08.0\Library\bin")

                    text = ""
                    for i, img in enumerate(images):
                        img_path = f"page_{i}.png"
                        img.save(img_path, "PNG")
                        logger.info(f"Performing OCR on {img_path}...")
                        ocr_text = pytesseract.image_to_string(Image.open(img_path))
                        text += ocr_text
                    logger.info("OCR-based PDF content extraction complete.")
                    return text
            else:
                logger.warning(f"Failed to download PDF. Status code: {response.status_code}")
        else:
            logger.info("URL points to HTML content. Parsing with BeautifulSoup...")
            html = requests.get(url, timeout=10).text
            text = extract_clean_text_from_html(html)
            # text=extract_content_with_selenium(url)
            logger.info(f"HTML content extracted successfully. and Text is : {text[:100]}...")
            logger.debug(f"Combined context length: {len(text)} characters")
            return clean_text
        
    except Exception as e:
        logger.error(f"Error extracting from URL {url}: {e}")
        return None


def search_google(query, api_key, num_results=2):
    logger.info(f"Searching Google via SerpAPI for query: {query}")
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"{query} site:nitj.ac.in",
        "api_key": api_key,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        logger.debug(f"SerpAPI request sent with params: {params}")
        if response.status_code != 200:
            logger.error(f"SerpAPI request failed with status code: {response.status_code}")
            return []

        results = response.json()
        extracted = []
        for r in results.get("organic_results", []):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            logger.info(f"Found result - Title: {title}, Link: {link}")
            extracted.append({"title": title, "snippet": snippet, "link": link})

        logger.info(f"Total results extracted from SerpAPI: {len(extracted)}")
        return extracted
    except Exception as e:
        logger.error(f"Error in SerpAPI search: {e}")
        return []


from langchain.schema import Document

def gemini_answer(context, question):
    logger.info("Generating answer using Gemini from fallback context...")
    logger.debug(f"Context passed to Gemini (first 500 chars): {context[:500]}")
    logger.debug(f"Question passed to Gemini: {question}")

    chain = get_conversational_chain()
    logger.info("Invoking conversational chain in fallback...")

    try:
        # Wrap context string in a list of Document objects
        context_docs = [Document(page_content=context)]

        # Pass context_docs instead of plain string
        response = chain.invoke({"context": context_docs, "question": question})

        response_text = response.strip()
        logger.info("Gemini response generated successfully from fallback.")
        logger.debug(f"Gemini fallback response (first 100 chars): {response_text[:100]}")
        return response_text

    except Exception as e:
        logger.error(f"Error during Gemini fallback generation: {e}")
        raise



async def fallback_response(question):
    logger.info("Entering fallback response flow...")
    api_key = os.getenv("SERP_API_KEY")
    logger.debug(f"SERP_API_KEY loaded: {'Yes' if api_key else 'No'}")

    search_results = search_google(question, api_key)
    if not search_results:
        logger.warning("Fallback: No search results found.")
        return {"response": "Could not find relevant results even through fallback search."}

    texts = []
    for result in search_results:
        link = result.get("link")
        logger.info(f"Attempting to extract content from fallback link: {link}")
        if link:
            content = extract_content_from_url(link)
            if content:
                texts.append(content)
            else:
                logger.warning(f"No content extracted from URL: {link}")
        else:
            logger.warning("Result does not contain a link.")

    if not texts:
        logger.warning("Fallback attempted but no readable content found.")
        return {"response": "Fallback attempted but no readable content was found."}

    combined_context = "\n".join(texts)[:10000]
    logger.info("Combined context from fallback sources ready for Gemini.")
    logger.debug(f"Combined context length: {len(combined_context)} characters")

    try:
        answer = gemini_answer(combined_context, question)
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error generating Gemini response in fallback: {e}")
        return {"response": "Could not generate a response even after fallback."}

































# --------------------------- Chat Functionality ----------------------------

def get_conversational_chain():
    logger.debug("Initializing conversational chain")
    prompt_template = """
    You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

    Guidelines:
    - Use  the given context to answer. If the answer is not found, first check the prompt i have given you, if still not found any appropriate answer then respond: "Answer is not available in the context."
    - Never fabricate or assume facts not in the context.
    - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
    - If a question involves steps, procedures, or processes, use clear bullet points.
    - For definitions or factual queries: provide concise, formal answers.
    - For lists: use bullet points.
    - For how-to or process questions: step-by-step format.
    - For comparisons: use tables or summaries.
    - Do NOT search externally; use only the provided context.
    - If the context is insufficient, say so clearly.
    - If counting not provided directly then count and give answer.
    - Use **bold headings**, bullet points, and numbered lists where applicable.
    - Structure your answers for readability.
    - Use tables if comparisons are involved.
    - when asked list then use bullet points to answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt,
        )
        logger.debug("Conversational chain initialized successfully")
        return chain
    except Exception as e:
        logger.error(f"Error initializing conversational chain: {e}")
        raise

def load_local_llm():
    logger.info("Loading local LLM (microsoft/phi-2)")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.float32,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                device_map="cpu"
            )
        logger.info("Local LLM loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading local LLM: {e}")
        raise

def local_llm_response(context, question):
    logger.info("Generating local LLM response for question: %s", question[:50])
    try:
        tokenizer, model = load_local_llm()
        prompt = """
        You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

        Guidelines:
        - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
        - Never fabricate or assume facts not in the context.
        - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
        - If a question involves steps, procedures, or processes, use clear bullet points.
        - For definitions or factual queries: provide concise, formal answers.
        - For lists: use bullet points.
        - For how-to or process questions: step-by-step format.
        - For comparisons: use tables or summaries.
        - Do NOT search externally; use only the provided context.
        - If the context is insufficient, say so clearly.
        - If counting not provided directly then count and give answer.
        - Use **bold headings**, bullet points, and numbered lists where applicable.
        - Structure your answers for readability.
        - Use tables if comparisons are involved.
        - when asked list then use bullet points to answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        inputs = tokenizer(prompt.format(context=context, question=question), return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.5,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded.split("Answer:")[-1].strip()
        logger.debug("Local LLM response generated: %s", answer[:50])
        return answer
    except Exception as e:
        logger.error(f"Error generating local LLM response: {e}")
        raise

def load_vector_store():
    logger.info("Loading FAISS vector store")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {e}")
        raise

# --------------------------- API Endpoints ----------------------------

@app.get("/api/connectivity")
async def get_connectivity():
    return {"online": check_connectivity()}

@app.get("/api/faiss-exists")
async def check_faiss():
    return {"exists": faiss_index_exists()}

@app.post("/api/process-pdfs")
async def process_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")
    try:
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        for i, chunk in enumerate(text_chunks[:5]):
            logger.debug(f"Chunk {i+1}: {chunk}...")

        get_vector_store(text_chunks)
        return {"message": "Documents processed successfully"}
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        raise HTTPException(status_code=500, detail="Failed to process documents")

@app.post("/api/ask")
async def ask_question(data: dict):
    user_question = data.get("question")
    offline_mode = data.get("offlineMode", False)  # Get offlineMode from frontend
    logger.info("Received question: %s, offlineMode: %s", user_question[:50], offline_mode)
    
    if not user_question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        vector_store = load_vector_store()
        retrieved_docs = vector_store.similarity_search(user_question, k=5)
        logger.debug("Retrieved %d documents for question", len(retrieved_docs))
        logger.debug("Retrieved documents content: %s", [doc.page_content[:200] for doc in retrieved_docs])
        
        if not retrieved_docs:
            response_text = "Answer is not available in the context or ask question in different way."
            logger.info("No relevant documents found for question")
            return await fallback_response(user_question)
        else:
            if offline_mode:
                context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                response_text = local_llm_response(context_text, user_question)
                logger.debug("Offline response generated: %s", response_text[:50])
            else:
                try:
                    chain = get_conversational_chain()
                    logger.info("Invoking conversational chain")
                    response = chain.invoke({"context": retrieved_docs, "question": user_question})
                    response_text = response.strip()
                    logger.debug("Online response generated: %s", response_text[:50])
                     # Check if Gemini returns unhelpful response
                    if "Answer is not available" in response_text or len(response_text) < 20:
                        return await fallback_response(user_question)

                except Exception as e:
                    logger.error(f"Error invoking chain: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to invoke chain: {str(e)}")
        return await fallback_response(user_question)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error processing user question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.post("/api/voice-to-text")
async def voice_to_text(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Initialize recognizer
        recognizer = sr.Recognizer()

        # Load audio file
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        # Transcribe
        text = recognizer.recognize_google(audio_data)

        logger.info("Voice successfully converted to text: %s", text[:50])
        return {"text": text}
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return {"error": "Speech recognition could not understand audio"}
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service: {e}")
        return {"error": "Google Speech API unavailable"}
    except Exception as e:
        logger.error(f"Voice to text conversion failed: {e}")
        return {"error": "Failed to convert voice to text"}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/feedback")
async def save_feedback(feedback: dict):
    try:
        # Initialize feedback file if it doesn't exist
        feedback_file = "feedback.json"
        if not os.path.exists(feedback_file):
            with open(feedback_file, "w") as f:
                json.dump([], f)
            logger.debug("Initialized empty feedback.json file")

        # Read existing feedback
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)

        # Append new feedback
        feedback_data.append({
            "question": feedback["question"],
            "response": feedback["response"],
            "feedback": feedback["feedback"],
            "timestamp": feedback["timestamp"],
        })

        # Write back to file
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

        logger.info(f"Saved {feedback['feedback']} feedback for question: {feedback['question'][:50]} and response: {feedback['response'][:50]}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")






# --------------------------- Health Check Endpoint ----------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "NITJ Chatbot backend is running"}

# --------------------------- App Runner ----------------------------

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))  # Get port from env, default to 8000
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
