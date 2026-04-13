"""
Name: Manish Pandey
Roll No: 2301010302
Course: Image Processing & Computer Vision
Unit: Mini Project
Assignment Title: Object Representation and Feature Extraction for Traffic Images
Date:
"""

import cv2
import numpy as np
import os

print("🚦 Feature-Based Traffic Monitoring System")

# Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# -----------------------------
# Task 1: Load Image
# -----------------------------
image_path = "traffic.jpg"  # change this
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/original.png", img)

print("✅ Image loaded")

# -----------------------------
# Task 1: Edge Detection
# -----------------------------

# Sobel Edge Detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobel_x, sobel_y)

# Normalize for saving
sobel = np.uint8(sobel)

# Canny Edge Detection
canny = cv2.Canny(gray, 100, 200)

cv2.imwrite("outputs/sobel.png", sobel)
cv2.imwrite("outputs/canny.png", canny)

print("✅ Edge detection done")

# -----------------------------
# Task 2: Contours & Bounding Boxes
# -----------------------------
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()

print("\n📦 Object Measurements:")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area > 500:  # filter noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        print(f"Object {i+1}: Area = {area:.2f}, Perimeter = {perimeter:.2f}")

cv2.imwrite("outputs/contours.png", contour_img)

print("✅ Contours and bounding boxes drawn")

# -----------------------------
# Task 3: Feature Extraction (ORB)
# -----------------------------
orb = cv2.ORB_create(nfeatures=500)

keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

cv2.imwrite("outputs/orb_features.png", feature_img)

print(f"✅ ORB Features detected: {len(keypoints)} keypoints")

# -----------------------------
# Task 4: Analysis
# -----------------------------
print("\n🧠 Analysis:")

print("1. Sobel detects gradients but produces thicker edges.")
print("2. Canny provides thin, well-defined edges → better for object detection.")
print("3. Contours help identify object boundaries like vehicles.")
print("4. Bounding boxes help track and count vehicles.")
print("5. ORB detects keypoints efficiently and is fast for real-time systems.")

print("\n🚗 Traffic Monitoring Insight:")
print("• Edge detection → lane & boundary detection")
print("• Contours → vehicle detection")
print("• Features → object tracking and recognition")
