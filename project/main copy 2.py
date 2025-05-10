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

# Draw all bounding boxes on the image
for x, y, w, h in boxes:
    cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Edge detection using Canny
edges = cv2.Canny(gray, 50, 150)

# Line detection using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
line_img = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Document edges using contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

# Show results
plt.figure(figsize=(16, 10))

# Image with MSER regions
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(mser_img, cv2.COLOR_BGR2RGB))
plt.title("MSER Detected Text Regions")
plt.axis('off')

# Image with bounding boxes
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("All Bounding Boxes")
plt.axis('off')

# Edge-detected image
plt.subplot(2, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

# Image with detected lines
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.title("Hough Line Detection")
plt.axis('off')

# Image with documented edges
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title("Documented Edges (Contours)")
plt.axis('off')

plt.tight_layout()
plt.show()
