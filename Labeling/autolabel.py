import cv2
import numpy as np
from matplotlib import pyplot as plt

#Makes the image more clear and less affected by noise
def enhance_contrast(image):
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(bilateral_filtered)

#Detects the edges of the image to identify the ink
#The Canny edge detection algorithm is used, which detects a wide range of edges in the image
def detect_crackle(image):
    enhanced_image = enhance_contrast(image)
    crackle = cv2.Canny(enhanced_image, 50, 150)
    return crackle


def detect_ink(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    enhanced = enhance_contrast(image)
    edges = cv2.Canny(enhanced, 100, 200)
    return edges

input_image_path = r'C:\Users\Mafe\Desktop\Segment8png\25.png'
output_image_path = r'C:\Users\Mafe\Desktop\processed_image.png'

detected_ink = detect_ink(input_image_path)
cv2.imwrite(output_image_path, detected_ink)

# Visualization for manual review
original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
processed_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

#The morphology operation is applied to the detected ink to fill in the gaps and make the ink more visible
def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return morphed

detected_ink = detect_ink(input_image_path)
morphed_ink = apply_morphology(detected_ink)
cv2.imwrite(output_image_path, morphed_ink)

#Visualization for manual review
processed_image_morphed = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

#Combines the detected ink and the morphed ink to create a single image with the ink highlighted
def combine_results(edges, morphed):
    combined = cv2.bitwise_or(edges, morphed)
    return combined

edges = detect_ink(input_image_path)
morphed = apply_morphology(edges)
combined_results = combine_results(edges, morphed)
cv2.imwrite(output_image_path, combined_results)

#Visualization for manual review
processed_image_combined = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(original_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Imagen Combinada")
plt.imshow(processed_image_combined, cmap='gray')
plt.show()

#Overlay the processed image on the original image for visual comparison
def overlay_images(original, processed, alpha=0.6):
    # Convert images to color if they are in grayscale
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Add color to the processed image to make it stand out (e.g., red)
    processed_colored = cv2.applyColorMap(processed, cv2.COLORMAP_JET)
    
    # Combine the images
    overlay = cv2.addWeighted(original, 1 - alpha, processed_colored, alpha, 0)
    return overlay


original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
processed_image_combined = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

#Applies the overlay function to the images
overlay_image = overlay_images(original_image, processed_image_combined)


overlay_image_path = r'C:\Users\Mafe\Desktop\overlay_image.png'
cv2.imwrite(overlay_image_path, overlay_image)


plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Superpuesta")
plt.show()
