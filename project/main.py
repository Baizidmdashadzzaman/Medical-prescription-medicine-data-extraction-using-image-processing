import cv2
import numpy as np
import pytesseract
import re
import easyocr  # Better for handwriting than Tesseract
import spacy
from PIL import Image

# Initialize
nlp = spacy.load("en_core_web_sm")
reader = easyocr.Reader(['en'])  # Requires CUDA for GPU acceleration


def preprocess_image(image_path):
    """Clean up the prescription image"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew (simplified)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC)

    return rotated


def extract_text(image):
    """OCR using EasyOCR for better handwriting support"""
    results = reader.readtext(image, paragraph=True)
    full_text = " ".join([res[1] for res in results])
    return full_text


def extract_medical_info(text):
    """Extract medicines and dosages using NLP and regex"""
    doc = nlp(text)

    medicine_info = {
        "medicines": [],
        "dosages": [],
        "frequencies": []
    }

    # Regex patterns
    dosage_pattern = r'\d+\s*(mg|g|ml|tablet|tab|times|hr|hour|day)'
    frequency_pattern = r'(once|twice|thrice|daily|every\s+\d+\s+hours)'

    # Find medicines (crude approach - consider medical NER models)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:  # Weak heuristic
            medicine_info["medicines"].append(ent.text)

    # Find dosages
    medicine_info["dosages"] = re.findall(dosage_pattern, text, re.I)

    # Find frequencies
    medicine_info["frequencies"] = re.findall(frequency_pattern, text, re.I)

    return medicine_info


def process_prescription(image_path):
    """Full processing pipeline"""
    preprocessed = preprocess_image(image_path)
    text = extract_text(preprocessed)
    info = extract_medical_info(text)

    return {
        "raw_text": text,
        "medical_info": info
    }


if __name__ == "__main__":
    # Test with a sample image
    result = process_prescription("data-4.jpg")

    print("Extracted Text:")
    print(result["raw_text"])
    print("\nMedical Information:")
    print(f"Medicines: {result['medical_info']['medicines']}")
    print(f"Dosages: {result['medical_info']['dosages']}")
    print(f"Frequencies: {result['medical_info']['frequencies']}")