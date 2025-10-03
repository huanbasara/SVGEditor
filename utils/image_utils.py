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
# Image Processing Functions
# ============================================================================

import cv2
import numpy as np


def morphological_smooth(binary_img, kernel_size=3, iterations=1):
    """
    Morphological closing operation: connect breaks, smooth contours.
    
    Process: Dilation → Erosion
    - Fills small gaps (< kernel_size)
    - Smooths jagged edges
    - Preserves overall thickness
    
    Args:
        binary_img: Binary image (white lines on black background)
        kernel_size: Size of morphological kernel (3, 5, 7 recommended)
        iterations: Number of closing iterations (1-2 recommended)
    
    Returns:
        Smoothed binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    print(f"Morphological smoothing: kernel={kernel_size}x{kernel_size}, iterations={iterations}")
    print(f"White pixels: {cv2.countNonZero(binary_img)} → {cv2.countNonZero(result)}")
    
    return result


def uniform_line_thickness(binary_img, target_width=3, preserve_threshold=0.5):
    """
    Unify line thickness using skeleton-based reconstruction.
    
    Strategy:
    1. Extract skeleton (1-pixel centerline)
    2. Rebuild lines from skeleton with uniform target_width
    3. Preserve originally thin lines (width < target_width * preserve_threshold)
    
    This provides smooth, uniform thickness while preserving fine details.
    
    Args:
        binary_img: Binary image (white lines on black background)
        target_width: Target line width in pixels (3-5 recommended)
        preserve_threshold: Ratio for preserving thin lines (0.0-1.0)
                          0.5 = preserve lines thinner than target_width * 0.5
    
    Returns:
        Uniformly thickened binary image
    """
    # Distance transform: measures thickness at each point
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # Extract skeleton (1-pixel centerline)
    skeleton = get_skeleton(binary_img)
    
    # Rebuild lines from skeleton with target width
    kernel_size = target_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    uniform_lines = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Identify originally thin lines to preserve
    thin_mask = np.zeros_like(binary_img, dtype=bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    
    thickness_threshold = target_width * preserve_threshold
    preserved_count = 0
    rebuilt_count = 0
    
    for i in range(1, num_labels):
        component_mask = (labels == i)
        max_dist = np.max(dist_transform[component_mask])
        
        # If max distance < thickness_threshold, preserve original
        if max_dist < thickness_threshold:
            thin_mask[component_mask] = True
            preserved_count += 1
        else:
            rebuilt_count += 1
    
    # Combine: use uniform width for thick lines, preserve original for thin lines
    result = uniform_lines.copy()
    result[thin_mask] = binary_img[thin_mask]
    
    print(f"Uniform line thickness: target_width={target_width}px, preserve_threshold={preserve_threshold}")
    print(f"Preserved {preserved_count} thin lines, rebuilt {rebuilt_count} thick lines")
    print(f"White pixels: {cv2.countNonZero(binary_img)} → {cv2.countNonZero(result)}")
    
    return result


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


