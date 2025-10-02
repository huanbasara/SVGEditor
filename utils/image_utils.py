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


def uniform_line_thickness(binary_img, target_thickness=2, tolerance=0.5, 
                          skeleton_pixel_tolerance=0.7, skeleton_endpoint_tolerance=1.5):
    """
    Make all lines uniform thickness: erode thick lines, dilate thin lines.
    
    Args:
        binary_img: Binary image (white lines on black background)
        target_thickness: Target radius for all lines (pixels)
        tolerance: Thickness tolerance range (lines within ±tolerance are left unchanged)
        skeleton_pixel_tolerance: Skeleton pixel preservation ratio (0.7 = allow 30% loss)
        skeleton_endpoint_tolerance: Skeleton endpoint increase ratio (1.5 = allow 50% increase)
    
    Returns:
        Uniformly thinned/thickened binary image
    """
    # Store skeleton tolerances for use in helper functions
    global _skeleton_pixel_tol, _skeleton_endpoint_tol
    _skeleton_pixel_tol = skeleton_pixel_tolerance
    _skeleton_endpoint_tol = skeleton_endpoint_tolerance
    
    # Get all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    
    result = np.zeros_like(binary_img)
    
    print(f"Processing {num_labels - 1} connected components...")
    print(f"Target thickness: {target_thickness}px, Tolerance: ±{tolerance}px")
    print(f"Skeleton preservation: pixel≥{skeleton_pixel_tolerance*100:.0f}%, endpoint≤{(skeleton_endpoint_tolerance-1)*100:.0f}% increase")
    
    eroded_count = 0
    dilated_count = 0
    unchanged_count = 0
    
    for i in range(1, num_labels):
        # Extract this component
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Measure current thickness
        dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        max_radius = dist_transform.max()
        
        # Determine action
        if max_radius > target_thickness + tolerance:
            # Too thick → erode
            processed, success = erode_component_safely(component_mask, max_radius, target_thickness)
            if success:
                eroded_count += 1
            else:
                unchanged_count += 1
        elif max_radius < target_thickness - tolerance:
            # Too thin → dilate
            processed, success = dilate_component_safely(component_mask, max_radius, target_thickness)
            if success:
                dilated_count += 1
            else:
                unchanged_count += 1
        else:
            # Within tolerance → keep original
            processed = component_mask
            unchanged_count += 1
        
        # Add to result
        result = cv2.bitwise_or(result, processed)
    
    print(f"Eroded {eroded_count}, Dilated {dilated_count}, Unchanged {unchanged_count}")
    print(f"White pixels: {cv2.countNonZero(binary_img)} → {cv2.countNonZero(result)}")
    
    return result


def erode_component_safely(component, current_radius, target_radius):
    """
    Erode component to target radius with structural integrity check.
    
    Returns:
        (processed_component, success)
    """
    # Calculate erosion amount
    erosion_amount = int(round(current_radius - target_radius))
    erosion_amount = max(1, erosion_amount)  # At least 1
    
    # Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(component, kernel, iterations=erosion_amount)
    
    # Check structural integrity
    if cv2.countNonZero(eroded) < 5:
        return component, False
    
    if not is_structure_preserved(component, eroded):
        return component, False
    
    return eroded, True


def dilate_component_safely(component, current_radius, target_radius):
    """
    Dilate component to target radius with structural integrity check.
    
    Returns:
        (processed_component, success)
    """
    # Calculate dilation amount
    dilation_amount = int(round(target_radius - current_radius))
    dilation_amount = max(1, min(dilation_amount, 2))  # Cap at 2 to avoid over-dilation
    
    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(component, kernel, iterations=dilation_amount)
    
    # For dilation, we mainly check if it's not too aggressive
    original_pixels = cv2.countNonZero(component)
    dilated_pixels = cv2.countNonZero(dilated)
    
    if dilated_pixels > original_pixels * 3:
        # Too aggressive, use less dilation
        return cv2.dilate(component, kernel, iterations=1), True
    
    return dilated, True





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


def gentle_line_thinning(binary_img, erosion_iterations=1, dilation_iterations=0):
    """
    Gentle line thinning using erosion (and optional dilation).
    
    This method does simple morphological operations without skeleton extraction.
    Suitable for: making thick lines slightly thinner while keeping fills as fills.
    
    Args:
        binary_img: Binary image (white lines on black background)
        erosion_iterations: Number of erosion iterations (1-2 recommended)
        dilation_iterations: Number of dilation iterations after erosion (0 = no dilation)
    
    Returns:
        Thinned binary image
    """
    result = binary_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Erosion: make lines thinner
    if erosion_iterations > 0:
        result = cv2.erode(result, kernel, iterations=erosion_iterations)
        print(f"Applied {erosion_iterations} erosion iteration(s)")
        print(f"White pixels after erosion: {cv2.countNonZero(result)}")
    
    # Dilation: expand lines back slightly (optional)
    if dilation_iterations > 0:
        result = cv2.dilate(result, kernel, iterations=dilation_iterations)
        print(f"Applied {dilation_iterations} dilation iteration(s)")
        print(f"White pixels after dilation: {cv2.countNonZero(result)}")
    
    return result


def selective_line_thinning(binary_img, erosion_iterations=1):
    """
    Selective line thinning: erode thick lines, preserve thin lines that would break.
    
    Uses skeleton comparison to detect structural changes (not just pixel count).
    
    Args:
        binary_img: Binary image (white lines on black background)
        erosion_iterations: Number of erosion iterations to try (1-2 recommended)
    
    Returns:
        Selectively thinned binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Get all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    
    result = np.zeros_like(binary_img)
    preserved_count = 0
    eroded_count = 0
    
    print(f"Processing {num_labels - 1} connected components...")
    
    for i in range(1, num_labels):
        # Extract this component
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Try eroding this component
        eroded_component = cv2.erode(component_mask, kernel, iterations=erosion_iterations)
        
        # Check structural integrity using skeleton
        if is_structure_preserved(component_mask, eroded_component):
            # Structure intact → use eroded version
            result = cv2.bitwise_or(result, eroded_component)
            eroded_count += 1
        else:
            # Structure damaged → preserve original
            result = cv2.bitwise_or(result, component_mask)
            preserved_count += 1
    
    print(f"Preserved {preserved_count} components, eroded {eroded_count} components")
    print(f"White pixels: {cv2.countNonZero(binary_img)} → {cv2.countNonZero(result)}")
    
    return result


def is_structure_preserved(original, eroded):
    """
    Check if erosion preserves the structural integrity.
    
    Uses skeleton comparison: if the skeleton structure remains similar,
    then erosion is safe.
    
    Uses global tolerance settings: _skeleton_pixel_tol and _skeleton_endpoint_tol
    
    Args:
        original: Original component mask
        eroded: Eroded component mask
    
    Returns:
        True if structure is preserved, False if damaged
    """
    # Get tolerance values (use defaults if not set)
    pixel_tol = globals().get('_skeleton_pixel_tol', 0.7)
    endpoint_tol = globals().get('_skeleton_endpoint_tol', 1.5)
    
    # Quick checks first
    eroded_pixels = cv2.countNonZero(eroded)
    if eroded_pixels == 0:
        return False  # Component completely eroded
    
    original_count = count_connected_components(original)
    eroded_count = count_connected_components(eroded)
    if eroded_count > original_count:
        return False  # Component split into multiple parts
    
    # Skeleton comparison for structural integrity
    original_skeleton = cv2.ximgproc.thinning(original)
    eroded_skeleton = cv2.ximgproc.thinning(eroded)
    
    original_skeleton_pixels = cv2.countNonZero(original_skeleton)
    eroded_skeleton_pixels = cv2.countNonZero(eroded_skeleton)
    
    # Check skeleton pixel preservation
    if eroded_skeleton_pixels < original_skeleton_pixels * pixel_tol:
        return False
    
    # Check if skeleton endpoints increased (indicates structure break)
    original_endpoints = count_skeleton_endpoints(original_skeleton)
    eroded_endpoints = count_skeleton_endpoints(eroded_skeleton)
    
    if eroded_endpoints > original_endpoints * endpoint_tol:
        return False  # Too many new endpoints = structure broken
    
    return True


def count_skeleton_endpoints(skeleton):
    """Count endpoints in a skeleton (pixels with only 1 neighbor)"""
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve: center pixel = 10, each neighbor = 1
    filtered = cv2.filter2D(skeleton // 255, -1, kernel)
    
    # Endpoint: value = 11 (10 + 1 neighbor)
    endpoints = np.sum(filtered == 11)
    
    return endpoints


def adaptive_local_thinning(binary_img, erosion_strength=1, thin_threshold=3):
    """
    Adaptive local thinning: gently erode thick parts, preserve thin parts.
    
    Uses distance transform to detect local thickness:
    - Thin areas (width < thin_threshold): preserve original
    - Thick areas (width >= thin_threshold): erode by erosion_strength pixels
    
    Args:
        binary_img: Binary image (white lines on black background)
        erosion_strength: How many pixels to erode from thick areas (1-2 recommended)
        thin_threshold: Thickness threshold to distinguish thin from thick (pixels)
    
    Returns:
        Gently thinned binary image
    """
    # Get all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    
    result = np.zeros_like(binary_img)
    
    print(f"Processing {num_labels - 1} connected components...")
    print(f"Erosion strength: {erosion_strength}px, Thin threshold: {thin_threshold}px")
    
    preserved_count = 0
    eroded_count = 0
    max_radii = []
    
    for i in range(1, num_labels):
        # Extract this component
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Apply gentle local thinning
        thinned_component, was_eroded, max_radius = thin_component_gently(component_mask, erosion_strength, thin_threshold)
        
        max_radii.append(max_radius)
        
        if was_eroded:
            eroded_count += 1
        else:
            preserved_count += 1
        
        # Add to result
        result = cv2.bitwise_or(result, thinned_component)
    
    # Show radius distribution
    if max_radii:
        max_radii_sorted = sorted(max_radii, reverse=True)
        print(f"Top 10 component max radii: {[f'{r:.1f}' for r in max_radii_sorted[:10]]}")
        print(f"Min radius: {min(max_radii):.1f}, Max radius: {max(max_radii):.1f}")
    
    print(f"Preserved {preserved_count} thin components, eroded {eroded_count} thick components")
    print(f"White pixels: {cv2.countNonZero(binary_img)} → {cv2.countNonZero(result)}")
    
    return result


def thin_component_gently(component, erosion_strength, thin_threshold):
    """
    Gently thin a component: only erode thick areas, preserve thin areas.
    
    Key insight: Distance transform = distance to nearest edge
    - Large value = thick area (far from edge)
    - Small value = thin area (close to edge)
    
    Strategy:
    1. Compute distance transform (each pixel's distance to edge)
    2. If max_distance < thin_threshold → preserve entire component (it's thin)
    3. If max_distance >= thin_threshold → erode by removing outer layers
    
    Args:
        component: Single component mask
        erosion_strength: How many pixels to erode (remove from edges)
        thin_threshold: Thickness threshold (if max radius < this, preserve)
    
    Returns:
        (thinned_component, was_eroded, max_radius): Tuple of result, erosion status, and max radius
    """
    # Distance transform: each pixel's distance to nearest edge
    dist_transform = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    
    # Max distance = "radius" of thickest part
    max_radius = dist_transform.max()
    
    # If entire component is thin, preserve it
    if max_radius < thin_threshold:
        return component, False, max_radius
    
    # Component is thick, erode it by removing outer pixels
    # Calculate effective erosion: don't erode more than what exists
    effective_erosion = min(erosion_strength, max_radius - 1.0)
    
    if effective_erosion < 0.5:
        # Too small to erode meaningfully
        return component, False, max_radius
    
    # Keep only pixels that are > effective_erosion away from edge
    # This removes effective_erosion pixels from the boundary
    result = (dist_transform > effective_erosion).astype(np.uint8) * 255
    
    # Safety check: if result is empty, return original
    result_pixels = cv2.countNonZero(result)
    
    if result_pixels < 5:
        # Would completely destroy the component, preserve original
        return component, False, max_radius
    
    # Structural integrity check: verify skeleton is preserved
    if not is_structure_preserved(component, result):
        # Erosion damaged structure, return original
        return component, False, max_radius
    
    # Successfully eroded
    return result, True, max_radius


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
