import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
import mimetypes
import requests
import tempfile

def extract_page_text(url, max_chars=1000):
    try:
        # Guess content type using extension or HTTP response headers
        content_type = mimetypes.guess_type(url)[0] or ""
        
        # --- Handle PDFs ---
        if url.endswith(".pdf") or "pdf" in content_type:
            response = requests.get(url)
            if response.status_code != 200:
                return "(PDF download failed)"
            pdf_bytes = response.content

            # Write bytes to a temporary PDF file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Extract text using PyMuPDF
            doc = fitz.open(tmp_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()

            # If no text found, fallback to OCR
            if full_text.strip():
                return full_text[:max_chars]
            else:
                images = convert_from_bytes(pdf_bytes)
                ocr_text = ""
                for img in images[:2]:  # OCR first 2 pages only
                    ocr_text += pytesseract.image_to_string(img)
                return ocr_text[:max_chars] if ocr_text.strip() else "(Could not OCR PDF)"
        
        # --- Handle Web Pages ---
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return "(Webpage download failed)"

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Join all visible text
        visible_text = ' '.join(soup.stripped_strings)
        return visible_text[:max_chars] if visible_text else "(No readable text found on webpage)"

    except Exception as e:
        return f"(Failed to extract text: {str(e)})"
