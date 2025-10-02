# Fill-to-Outline Processing

import cv2
import numpy as np
from PIL import Image
import os
from utils.plot_utils import plot_images

# Processing parameters
BINARY_THRESHOLD = 150
THICKNESS_THRESHOLD = 5  # Higher threshold to detect truly thick fills
MIN_FILL_AREA = 500  # Minimum area for filled regions
OUTLINE_WIDTH = 2

def convert_fills_to_outlines(binary_img):
    """
    Convert thick filled regions to outline borders using local detection.
    
    Key improvement: Process only locally thick regions, not entire connected components.
    This prevents thin lines connected to thick fills from being removed.
    """
    # Step 1: Detect thick regions using morphological erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THICKNESS_THRESHOLD, THICKNESS_THRESHOLD))
    eroded = cv2.erode(binary_img, kernel)
    
    # Step 2: Find connected components in the eroded image (only truly thick fills survive)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded)
    
    result_img = binary_img.copy()
    processed_count = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Only process large filled regions
        if area < MIN_FILL_AREA:
            continue
        
        # Step 3: Get the component mask from eroded image
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Step 4: Dilate back to recover the original fill boundary
        dilated_mask = cv2.dilate(component_mask, kernel)
        
        # Step 5: Extract this region from original binary image
        original_fill = cv2.bitwise_and(binary_img, dilated_mask)
        
        # Step 6: Create outline from this fill
        outline_mask = create_outline_from_fill(original_fill)
        
        # Step 7: Remove the fill and add outline (only in this local region)
        result_img = cv2.bitwise_and(result_img, cv2.bitwise_not(dilated_mask))
        result_img = cv2.bitwise_or(result_img, outline_mask)
        processed_count += 1
    
    print(f"Processed {processed_count} thick filled regions")
    return result_img

def binarize_image(img):
    """Binarize image (inverted: white lines on black background)"""
    _, binary = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    print(f"Binary threshold: {BINARY_THRESHOLD}")
    print(f"White pixels: {np.sum(binary > 0)}")
    return binary

def create_outline_from_fill(fill_mask):
    """Generate outline from fill region"""
    contours, _ = cv2.findContours(fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline_mask = np.zeros_like(fill_mask)

    for contour in contours:
        cv2.drawContours(outline_mask, [contour], -1, 255, thickness=OUTLINE_WIDTH)

    return outline_mask

def process_fill_to_outline():
    """Process fill-to-outline conversion"""
    input_path = os.path.join(TARGET_OUTPUT_PATH, f"{TARGET_NAME}_qwen_edit.png")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return None

    print(f"Loading image: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    print(f"Original image shape: {img.shape}")

    # Binarize
    binary = binarize_image(img)

    # Convert fills to outlines
    outline_img = convert_fills_to_outlines(binary)

    # Save results
    binary_path = os.path.join(TARGET_OUTPUT_PATH, f"{TARGET_NAME}_binary.png")
    outline_path = os.path.join(TARGET_OUTPUT_PATH, f"{TARGET_NAME}_outline.png")

    cv2.imwrite(binary_path, binary)
    cv2.imwrite(outline_path, outline_img)

    print(f"Binary image saved: {binary_path}")
    print(f"Outline image saved: {outline_path}")

    return binary, outline_img

# Execute fill-to-outline processing
print("=== Fill-to-Outline Processing ===")
binary_result, outline_result = process_fill_to_outline()

if outline_result is not None:
    print("✅ Fill-to-outline processing completed!")
    
    # Load images for visualization
    original_path = os.path.join(TARGET_OUTPUT_PATH, f"{TARGET_NAME}_qwen_edit.png")
    
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is not None:
        # Convert to PIL for plot_images
        original_pil = Image.fromarray(original_img).convert('RGB')
        binary_pil = Image.fromarray(binary_result).convert('RGB')
        outline_pil = Image.fromarray(outline_result).convert('RGB')
        
        plot_images([
            (original_pil, "Original"),
            (binary_pil, "Binary"),
            (outline_pil, "Outline")
        ])
    else:
        print("❌ Cannot load images for visualization")
else:
    print("❌ Fill-to-outline processing failed!")

