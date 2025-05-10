from PIL import Image
import pytesseract
import re

# Set the path to Tesseract executable (update based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# Load the image
image = Image.open("data-4.jpg")

# Extract text using OCR
text = pytesseract.image_to_string(image)

# Process the extracted text to find medicine names
medicine_names = []
for line in text.split('\n'):
    # Match lines with time followed by medicine names (e.g., "7:15. Neebion")
    match = re.search(r'\d+:\d+\.\s+([A-Za-zèé]+)', line)
    if match:
        name = match.group(1).strip()
        medicine_names.append(name)

print("Extracted Medicine Names:", medicine_names)