from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the handwritten image
image = Image.open("1.jpg").convert("RGB")

# Load processor and model from Hugging Face
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Preprocess the image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate predicted text
with torch.no_grad():
    generated_ids = model.generate(pixel_values)

# Decode to text
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Output result
print("\n✅ Detected Handwritten Text:\n")
print(text)
