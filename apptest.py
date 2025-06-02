import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from PIL import Image
import pytesseract
print(pytesseract.image_to_string(Image.open('page_0.png')))
print(pytesseract.image_to_string(Image.open('page_1.png')))
