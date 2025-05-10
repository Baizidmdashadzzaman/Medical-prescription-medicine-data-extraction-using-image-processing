import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("data-4.jpg")
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# MSER object
mser = cv2.MSER_create()

# Detect regions
regions, _ = mser.detectRegions(gray)

# Draw MSER regions
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
mser_img = original.copy()
cv2.polylines(mser_img, hulls, 1, (0, 255, 0))

# Find bounding boxes from hulls
boxes = [cv2.boundingRect(hull) for hull in hulls]

# Filter and group right-side boxes (assumes medicine section is on the right)
right_boxes = [box for box in boxes if box[0] > gray.shape[1] // 2 and box[2] > 30 and box[3] > 20]

# Get union of right-side boxes to create one bounding region
if right_boxes:
    x_vals = [x for x, y, w, h in right_boxes]
    y_vals = [y for x, y, w, h in right_boxes]
    x_ends = [x + w for x, y, w, h in right_boxes]
    y_ends = [y + h for x, y, w, h in right_boxes]

    x_min, y_min = min(x_vals), min(y_vals)
    x_max, y_max = max(x_ends), max(y_ends)

    medicine_section = original[y_min:y_max, x_min:x_max]
else:
    medicine_section = None

# Show result
plt.figure(figsize=(14, 8))

# Image with MSER regions
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(mser_img, cv2.COLOR_BGR2RGB))
plt.title("MSER Detected Text Regions")
plt.axis('off')

# Extracted section
if medicine_section is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(medicine_section, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Medicine Section")
    plt.axis('off')
else:
    print("Still couldn't detect medicine section.")

plt.tight_layout()
plt.show()

