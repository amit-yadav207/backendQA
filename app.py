# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Remove after recreating environment
# import re
# import socket
# import os.path as ospath
# from dotenv import load_dotenv
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import logging
# from logging.handlers import RotatingFileHandler
# from typing import List
# import google.generativeai as genai
# import uvicorn

# # Configure logging
# logger = logging.getLogger('NITJChatbot')
# logger.setLevel(logging.DEBUG)
# handler = RotatingFileHandler('chatbot.log', maxBytes=1_000_000, backupCount=3)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# logger.info("Starting NITJ AI Chatbot backend")

# # Load API key from .env file
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# logger.debug("Google API key configured")

# app = FastAPI()

# # Add CORS middleware to allow frontend requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Vite default port
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --------------------------- Utility Functions ---------------------------

# def clean_text(text):
#     logger.debug("Cleaning text input")
#     cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
#     logger.debug("Text cleaned successfully")
#     return cleaned

# def check_connectivity():
#     logger.info("Checking internet connectivity")
#     try:
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         logger.info("Internet connectivity confirmed")
#         return True
#     except OSError as e:
#         logger.error(f"Connectivity check failed: {e}")
#         return False

# def faiss_index_exists():
#     logger.debug("Checking if FAISS index exists")
#     exists = ospath.exists("faiss_index") and ospath.isfile(ospath.join("faiss_index", "index.faiss"))
#     logger.debug(f"FAISS index exists: {exists}")
#     return exists

# # --------------------------- PDF Processing ----------------------------

# def get_pdf_text(pdf_docs: List[UploadFile]):
#     logger.info("Extracting text from %d PDF documents", len(pdf_docs))
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf.file)
#             for page in pdf_reader.pages:
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text + "\n"
#             logger.debug("Text extracted from PDF: %s", pdf.filename)
#         except Exception as e:
#             logger.error(f"Error extracting text from PDF {pdf.filename}: {e}")
#     cleaned_text = clean_text(text)
#     logger.info("PDF text extraction completed, length: %d characters", len(cleaned_text))
#     return cleaned_text

# def get_text_chunks(text):
#     logger.info("Splitting text into chunks")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
#     chunks = text_splitter.split_text(text)
#     logger.debug("Text split into %d chunks", len(chunks))
#     return chunks

# def get_vector_store(text_chunks):
#     logger.info("Creating FAISS vector store with %d text chunks", len(text_chunks))
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#         logger.info("FAISS vector store created and saved to 'faiss_index'")
#     except Exception as e:
#         logger.error(f"Error creating FAISS vector store: {e}")
#         raise

# # --------------------------- Chat Functionality ----------------------------

# def get_conversational_chain():
#     logger.debug("Initializing conversational chain")
#     prompt_template = """
#     You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

#     Guidelines:
#     - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
#     - Never fabricate or assume facts not in the context.
#     - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
#     - If a question involves steps, procedures, or processes, use clear bullet points.
#     - For definitions or factual queries: provide concise, formal answers.
#     - For lists: use bullet points.
#     - For how-to or process questions: step-by-step format.
#     - For comparisons: use tables or summaries.
#     - Do NOT search externally; use only the provided context.
#     - If the context is insufficient, say so clearly.
#     - If counting not provided directly then count and give answer.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     try:
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#         prompt = ChatPromptTemplate.from_template(prompt_template)
#         chain = create_stuff_documents_chain(
#             llm=model,
#             prompt=prompt,
#         )
#         logger.debug("Conversational chain initialized successfully")
#         return chain
#     except Exception as e:
#         logger.error(f"Error initializing conversational chain: {e}")
#         raise

# def load_local_llm():
#     logger.info("Loading local LLM (microsoft/phi-2)")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             "microsoft/phi-2",
#             torch_dtype=torch.float32,
#             device_map="auto"
#         )
#         logger.info("Local LLM loaded successfully")
#         return tokenizer, model
#     except Exception as e:
#         logger.error(f"Error loading local LLM: {e}")
#         raise

# def local_llm_response(context, question):
#     logger.info("Generating local LLM response for question: %s", question[:50])
#     try:
#         tokenizer, model = load_local_llm()
#         prompt = """
#         You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

#         Guidelines:
#         - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
#         - Never fabricate or assume facts not in the context.
#         - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
#         - If a question involves steps, procedures, or processes, use clear bullet points.
#         - For definitions or factual queries: provide concise, formal answers.
#         - For lists: use bullet points.
#         - For how-to or process questions: step-by-step format.
#         - For comparisons: use tables or summaries.
#         - Do NOT search externally; use only the provided context.
#         - If the context is insufficient, say so clearly.
#         - If counting not provided directly then count and give answer.

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer:
#         """
#         inputs = tokenizer(prompt.format(context=context, question=question), return_tensors="pt", truncation=True, max_length=2048).to(model.device)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             temperature=0.5,
#             do_sample=True,
#             top_p=0.95,
#             pad_token_id=tokenizer.eos_token_id
#         )
#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         answer = decoded.split("Answer:")[-1].strip()
#         logger.debug("Local LLM response generated: %s", answer[:50])
#         return answer
#     except Exception as e:
#         logger.error(f"Error generating local LLM response: {e}")
#         raise

# def load_vector_store():
#     logger.info("Loading FAISS vector store")
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         logger.info("FAISS vector store loaded successfully")
#         return vector_store
#     except Exception as e:
#         logger.error(f"Error loading FAISS vector store: {e}")
#         raise

# # --------------------------- API Endpoints ----------------------------

# @app.get("/api/connectivity")
# async def get_connectivity():
#     return {"online": check_connectivity()}

# @app.get("/api/faiss-exists")
# async def check_faiss():
#     return {"exists": faiss_index_exists()}

# @app.post("/api/process-pdfs")
# async def process_pdfs(files: List[UploadFile] = File(...)):
#     if not files:
#         raise HTTPException(status_code=400, detail="No PDF files uploaded")
#     try:
#         raw_text = get_pdf_text(files)
#         text_chunks = get_text_chunks(raw_text)
#         get_vector_store(text_chunks)
#         return {"message": "Documents processed successfully"}
#     except Exception as e:
#         logger.error(f"Error processing PDFs: {e}")
#         raise HTTPException(status_code=500, detail="Failed to process documents")

# @app.post("/api/ask")
# async def ask_question(data: dict):
#     user_question = data.get("question")
#     logger.info(user_question)
#     if not user_question:
#         raise HTTPException(status_code=400, detail="Question is required")
    
#     try:
#         vector_store = load_vector_store()
#         retrieved_docs = vector_store.similarity_search(user_question, k=3)
#         logger.debug("Retrieved %d documents for question", len(retrieved_docs))
        
#         if not retrieved_docs:
#             response_text = "Answer is not available in the context or ask question in different way."
#             logger.info("No relevant documents found for question")
#         else:
#             if check_connectivity():
#                 chain = get_conversational_chain()
#                 logger.info("Got conversational Chain")
#                 response = chain.invoke({"context": retrieved_docs, "question": user_question})
#                 logger.debug("response: ",response)
#                 response_text = response
#                 logger.debug("Online response generated: %s", response_text[:50])
#             else:
#                 context_text = "\n".join([doc.page_content for doc in retrieved_docs])
#                 response_text = local_llm_response(context_text, user_question)
#                 logger.debug("Offline response generated: %s", response_text[:50])
        
#         return {"response": response_text}
#     except Exception as e:
#         logger.error(f"Error processing user question: {e}")
#         raise HTTPException(status_code=500, detail="Failed to process question")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



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
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import torch
import logging
from logging.handlers import RotatingFileHandler
from typing import List
import google.generativeai as genai
import uvicorn

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
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Utility Functions ---------------------------

def clean_text(text):
    logger.debug("Cleaning text input")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
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

# --------------------------- Chat Functionality ----------------------------

def get_conversational_chain():
    logger.debug("Initializing conversational chain")
    prompt_template = """
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
        retrieved_docs = vector_store.similarity_search(user_question, k=3)
        logger.debug("Retrieved %d documents for question", len(retrieved_docs))
        
        if not retrieved_docs:
            response_text = "Answer is not available in the context or ask question in different way."
            logger.info("No relevant documents found for question")
        else:
            # # Ensure retrieved_docs are in Document format
            # docs = [
            #     Document(page_content=doc.page_content)
            #     if not isinstance(doc, Document)
            #     else doc
            #     for doc in retrieved_docs
            # ]
            
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
                except Exception as e:
                    logger.error(f"Error invoking chain: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to invoke chain: {str(e)}")
        
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error processing user question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)