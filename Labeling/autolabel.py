import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

# Mejora el contraste y reduce el ruido en la imagen
def enhance_contrast(image):
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(bilateral_filtered)

# Aplica técnicas de augmentación adicionales
def augment_image(image):
    # Aumentar brillo y contraste
    alpha = 2.0 # Contraste
    beta = 50   # Brillo
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Desenfoque Gaussiano
    blurred = cv2.GaussianBlur(adjusted, (3, 3), 0)
    
    # Detección de bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

# Recorta la imagen para enfocar mejor los rastros de tinta
def crop_image(image, crop_size=(300, 300)):
    height, width = image.shape
    start_x = random.randint(0, width - crop_size[0])
    start_y = random.randint(0, height - crop_size[1])
    end_x = start_x + crop_size[0]
    end_y = start_y + crop_size[1]
    return image[start_y:end_y, start_x:end_x]

# Redimensiona la imagen a las dimensiones especificadas
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

# Aplica operaciones morfológicas para hacer los rastros de tinta más visibles
def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    morphed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return morphed

def detect_ink(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    enhanced = enhance_contrast(image)
    augmented = augment_image(enhanced)
    cropped = crop_image(augmented)
    morphed = apply_morphology(cropped)
    resized = resize_image(morphed)
    return resized

input_image_path = r'C:\Users\Mafe\Desktop\Carga2_PNG\25.png'
output_image_path = r'C:\Users\Mafe\Desktop\processed_image.png'

detected_ink = detect_ink(input_image_path)
cv2.imwrite(output_image_path, detected_ink)

# Visualización para revisión manual
original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
processed_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

# Combina los resultados detectados de tinta y la imagen morfológica para crear una sola imagen
def combine_results(edges, morphed):
    combined = cv2.bitwise_or(edges, morphed)
    return combined

edges = detect_ink(input_image_path)
morphed = apply_morphology(edges)
combined_results = combine_results(edges, morphed)
cv2.imwrite(output_image_path, combined_results)

# Visualización para revisión manual
processed_image_combined = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(original_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Imagen Combinada")
plt.imshow(processed_image_combined, cmap='gray')
plt.show()

# Superpone la imagen procesada sobre la imagen original para comparación visual
def overlay_images(original, processed, alpha=0.6):
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Redimensiona la imagen original para que coincida con la imagen procesada
    original_resized = resize_image(original, processed.shape[:2])
    
    processed_colored = cv2.applyColorMap(processed, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_resized, 1 - alpha, processed_colored, alpha, 0)
    return overlay

overlay_image = overlay_images(original_image, processed_image_combined)

overlay_image_path = r'C:\Users\Mafe\Desktop\overlay_image.png'
cv2.imwrite(overlay_image_path, overlay_image)

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.title("Imagen Superpuesta")
plt.show()
