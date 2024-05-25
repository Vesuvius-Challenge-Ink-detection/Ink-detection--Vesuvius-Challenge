import cv2
import numpy as np
from matplotlib import pyplot as plt


def crop_image(image, start_row, start_col, end_row, end_col):
    return image[start_row:end_row, start_col:end_col]

def enhance_contrast(image):
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(bilateral_filtered)


def detect_ink(image):
    enhanced = enhance_contrast(image)
    edges = cv2.Canny(enhanced, 50, 150)
    return edges


def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return morphed


def combine_results(edges, morphed):
    combined = cv2.bitwise_or(edges, morphed)
    return combined


def thicken_edges(edges, thickness=3):
    kernel = np.ones((thickness, thickness), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    return thick_edges

def overlay_images_with_annotations(original, processed, alpha=0.6):
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    blue_background = np.zeros_like(original)
    blue_background[:, :] = [255, 0, 0]  

    red_lines = np.zeros_like(original)
    red_lines[processed > 0] = [0, 0, 255]  

    overlay = cv2.addWeighted(blue_background, alpha, red_lines, 1 - alpha, 0)
    overlay = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)
    
    return overlay

input_image_path = r'C:\Users\Mafe\Desktop\Segment4png\00.png'
original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen en la ruta: {input_image_path}")

start_row, start_col, end_row, end_col = 2000, 400, 2500, 1400 
cropped_image = crop_image(original_image, start_row, start_col, end_row, end_col)

edges = detect_ink(cropped_image)
morphed = apply_morphology(edges)
combined_results = combine_results(edges, morphed)

thick_edges = thicken_edges(combined_results)

overlay_image = overlay_images_with_annotations(cropped_image, thick_edges)

overlay_image_path = r'C:\Users\Mafe\Desktop\overlay_image_with_thick_red_lines.png'
cv2.imwrite(overlay_image_path, overlay_image)

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.title("Detección de Tinta con Líneas Rojas y Fondo Azul")
plt.show()
