import cairosvg
import io
import os
from PIL import Image
import matplotlib.pyplot as plt


def read_svg_file(svg_path):
    with open(svg_path, 'r', encoding='utf-8') as f:
        return f.read()


def svg_code_to_pil_image(svg_code, width=None, height=None, dpi=300, background_color='white'):
    png_bytes = cairosvg.svg2png(
        bytestring=svg_code.encode('utf-8'),
        output_width=width,
        output_height=height,
        dpi=dpi
    )
    
    pil_image = Image.open(io.BytesIO(png_bytes))
    
    if pil_image.mode in ('RGBA', 'LA'):
        if background_color == 'white':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
        elif background_color == 'black':
            background = Image.new('RGB', pil_image.size, (0, 0, 0))
        else:
            background = Image.new('RGB', pil_image.size, background_color)
        
        if pil_image.mode == 'RGBA':
            background.paste(pil_image, mask=pil_image.split()[-1])
        else:
            background.paste(pil_image)
        
        return background
    else:
        return pil_image.convert("RGB")


def save_pil_image(pil_image, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    pil_image.save(full_path)
    return full_path


def create_comparison_plot(original_image, edited_image, prompt, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(edited_image)
    axes[1].set_title("Qwen Edit")
    axes[1].axis('off')
    
    fig.text(0.5, 0.02, f"Edit Prompt: {prompt}",
             fontsize=10, ha='center', wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    return save_path


# ============================================================================
# Line Thinning Functions (追加的新功能)
# ============================================================================

import cv2
import numpy as np


def get_skeleton(binary_img):
    """
    Extract skeleton (centerline) of binary image using morphological thinning.
    This preserves connectivity while reducing thickness to 1 pixel.
    
    Args:
        binary_img: Binary image (white lines on black background)
    
    Returns:
        Skeleton image
    """
    # Convert to binary format required by morphologyEx
    img_binary = binary_img.copy()
    img_binary[img_binary > 0] = 1
    
    # Morphological skeleton
    skeleton = np.zeros_like(img_binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        # Erode the image
        eroded = cv2.erode(img_binary, kernel)
        
        # Dilate the eroded image
        temp = cv2.dilate(eroded, kernel)
        
        # Subtract the dilated image from the original
        temp = cv2.subtract(img_binary, temp)
        
        # Bitwise OR with the skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)
        
        # Update the image for next iteration
        img_binary = eroded.copy()
        
        # Stop when image is completely eroded
        if cv2.countNonZero(img_binary) == 0:
            break
    
    # Convert back to 0-255 range
    skeleton = skeleton * 255
    
    return skeleton


def count_connected_components(binary_img):
    """Count number of connected components in binary image"""
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_img)
    return num_labels - 1  # Subtract background


def adaptive_line_thinning(binary_img, max_iterations=3, preserve_connectivity=True):
    """
    Adaptively thin lines: reduce thickness of thick lines while preserving thin lines.
    
    This method uses controlled erosion that stops when a line would break.
    
    Args:
        binary_img: Binary image (white lines on black background)
        max_iterations: Maximum erosion iterations (controls how much thinning)
        preserve_connectivity: If True, stop erosion if connectivity would be lost
    
    Returns:
        Thinned binary image
    """
    result = binary_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    for i in range(max_iterations):
        # Try erosion
        eroded = cv2.erode(result, kernel, iterations=1)
        
        if preserve_connectivity:
            # Check if erosion breaks connectivity
            original_components = count_connected_components(result)
            eroded_components = count_connected_components(eroded)
            
            # If erosion increases component count, it broke some connections
            if eroded_components > original_components:
                print(f"Stopped at iteration {i}: erosion would break connectivity")
                break
            
            # Check if any line completely disappeared
            original_nonzero = cv2.countNonZero(result)
            eroded_nonzero = cv2.countNonZero(eroded)
            
            # If we lost more than 50% of pixels, stop (too aggressive)
            if eroded_nonzero < original_nonzero * 0.5:
                print(f"Stopped at iteration {i}: too much pixel loss")
                break
        
        result = eroded
        print(f"Iteration {i+1}: kept {cv2.countNonZero(result)} pixels")
    
    return result


def smart_line_thinning(binary_img, target_width=2, preserve_threshold=0.5):
    """
    Smart line thinning using distance transform and skeleton.
    
    Strategy:
    1. Calculate distance transform (measures line thickness at each point)
    2. Extract skeleton (centerline)
    3. Rebuild lines from skeleton with uniform target width
    4. For originally thin lines (width < target), preserve original
    
    Args:
        binary_img: Binary image (white lines on black background)
        target_width: Target line width in pixels (larger = less aggressive thinning)
        preserve_threshold: Threshold ratio for preserving original thickness
                          (0.0-1.0, higher = preserve more original lines)
    
    Returns:
        Thinned binary image with more uniform thickness
    """
    # Distance transform: measures distance from each white pixel to nearest edge
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # Extract skeleton
    skeleton = get_skeleton(binary_img)
    
    # Rebuild lines from skeleton with target width
    half_width = target_width // 2
    kernel_size = target_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate skeleton to target width
    uniform_lines = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Identify originally thin lines (max distance < threshold)
    # For these areas, preserve original
    thin_mask = np.zeros_like(binary_img, dtype=bool)
    
    # Get connected components of original image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    
    # Calculate threshold based on preserve_threshold parameter
    # preserve_threshold: 0.5 means preserve lines with max_dist < target_width/2
    # preserve_threshold: 1.0 means preserve lines with max_dist < target_width
    thickness_threshold = target_width * preserve_threshold
    
    for i in range(1, num_labels):
        component_mask = (labels == i)
        max_dist = np.max(dist_transform[component_mask])
        
        # If max distance < thickness_threshold, preserve original
        if max_dist < thickness_threshold:
            thin_mask[component_mask] = True
    
    # Combine: use uniform width for thick lines, preserve original for thin lines
    result = uniform_lines.copy()
    result[thin_mask] = binary_img[thin_mask]
    
    return result
