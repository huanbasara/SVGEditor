"""
SAM (Segment Anything Model) processor for image segmentation
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def initialize_sam(checkpoint_path: str, config: Dict, model_type: str = "vit_h", device: str = "cuda") -> SamAutomaticMaskGenerator:
    """
    Initialize SAM model with given checkpoint and configuration
    
    Args:
        checkpoint_path: Path to SAM model checkpoint (.pth file)
        config: Configuration dictionary for SamAutomaticMaskGenerator
        model_type: Model type (default: "vit_h")
        device: Device to run the model on (default: "cuda")
    
    Returns:
        SamAutomaticMaskGenerator instance
    """
    print(f"🔄 Loading SAM model from: {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, **config)
    print("✅ SAM model loaded successfully")
    return mask_generator


def process_image_with_sam(image_path: str, sam_config: Dict, model_path: str) -> List[Dict]:
    """
    Process an image with SAM to generate segmentation masks
    
    Args:
        image_path: Path to input image
        sam_config: Configuration for SAM model including model parameters
        model_path: Path to SAM model checkpoint
    
    Returns:
        List of mask dictionaries containing segmentation results
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAM model not found: {model_path}")
    
    # SAM configuration
    sam_model_config = {
        "points_per_side": sam_config.get("points_per_side", 32),
        "pred_iou_thresh": sam_config.get("pred_iou_thresh", 0.88),
        "stability_score_thresh": sam_config.get("stability_score_thresh", 0.95),
        "crop_n_layers": sam_config.get("crop_n_layers", 1),
        "crop_n_points_downscale_factor": sam_config.get("crop_n_points_downscale_factor", 2),
        "min_mask_region_area": sam_config.get("min_mask_region_area", 100),
    }
    
    # Initialize SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator = initialize_sam(model_path, sam_model_config, device=device)
    
    # Load and process image
    print(f"🖼️ Processing image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    print("🎭 Generating masks...")
    masks = mask_generator.generate(image_rgb)
    print(f"✅ Generated {len(masks)} masks")
    
    return masks, image_rgb


def save_masks_to_directory(masks: List[Dict], image: np.ndarray, output_dir: str, base_filename: str = "mask") -> None:
    """
    Save individual masks to specified directory
    
    Args:
        masks: List of mask dictionaries from SAM
        image: Original image as numpy array
        output_dir: Directory to save mask images
        base_filename: Base filename for mask files (default: "mask")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean existing mask files (idempotent operation)
    mask_files = list(output_path.glob(f"{base_filename}_*.png"))
    for mask_file in mask_files:
        mask_file.unlink()
    
    if not masks:
        print("⚠️ No masks to save")
        return
    
    # Sort masks by area (largest first)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    print(f"💾 Saving {len(sorted_masks)} masks to: {output_dir}")
    
    # Save combined view with all masks
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    
    for i, mask_data in enumerate(sorted_masks):
        # Create colored mask overlay
        color = np.concatenate([np.random.random(3), [0.6]])
        mask_overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
        mask_overlay[mask_data['segmentation']] = color
        ax.imshow(mask_overlay)
        
        # Save individual mask
        mask_fig, mask_ax = plt.subplots(figsize=(8, 8))
        mask_ax.imshow(mask_overlay)
        mask_ax.axis('off')
        
        # Add mask statistics
        stats_text = (
            f"Area: {mask_data['area']:.0f}\n"
            f"IoU: {mask_data['predicted_iou']:.3f}\n"
            f"Stability: {mask_data['stability_score']:.3f}"
        )
        mask_ax.text(10, 40, stats_text, fontsize=10, color='white', 
                     bbox=dict(facecolor='black', alpha=0.7))
        
        individual_mask_path = output_path / f"{base_filename}_{i:03d}.png"
        plt.savefig(individual_mask_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(mask_fig)
    
    # Save combined mask view
    ax.axis('off')
    combined_mask_path = output_path / f"{base_filename}_combined.png"
    plt.savefig(combined_mask_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✅ Saved {len(sorted_masks)} individual masks + 1 combined view")
    print(f"📁 Files saved to: {output_dir}")


def get_default_sam_config(project_dir: str) -> Dict:
    """
    Get default SAM configuration
    
    Args:
        project_dir: Base project directory path
    
    Returns:
        Dictionary with default SAM configuration
    """
    return {
        "PROJECT_DIR": project_dir,
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100,
    }
