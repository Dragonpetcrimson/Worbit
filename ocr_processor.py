# ocr_processor.py
from typing import List, Dict
from PIL import Image
import pytesseract
import os

def extract_ocr_data(image_paths: List[str], min_length: int = 15) -> List[Dict]:
    """
    Performs OCR on each image. Skips blank or too-short results.
    Returns a list of {'file': filename, 'text': ocr_text}
    """
    ocr_results = []

    for image_path in image_paths:
        try:
            text = pytesseract.image_to_string(Image.open(image_path))

            # Clean and validate
            cleaned_text = text.strip()
            if len(cleaned_text) >= min_length:
                ocr_results.append({
                    'file': os.path.basename(image_path),
                    'text': cleaned_text
                })

        except Exception as e:
            print(f"❌ OCR failed on {image_path}: {e}")
            continue

    return ocr_results
