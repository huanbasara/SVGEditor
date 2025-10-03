import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_images(images_and_titles, bottom_text=None):
    """
    Display images in a horizontal row with titles.
    
    Args:
        images_and_titles: List of tuples containing (image, title)
        bottom_text: Optional text to display below all images
    """
    n_images = len(images_and_titles)
    
    if n_images == 0:
        return
    
    # Adjust figure height if bottom text is provided
    fig_height = 6 if bottom_text is None else 7
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, fig_height))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(images_and_titles):
        # Auto-detect grayscale images and use gray colormap
        if isinstance(image, Image.Image) and image.mode == 'L':
            axes[i].imshow(image, cmap='gray')
        elif isinstance(image, np.ndarray) and len(image.shape) == 2:
            axes[i].imshow(image, cmap='gray')
        else:
            axes[i].imshow(image)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Add bottom text if provided
    if bottom_text:
        fig.text(0.5, 0.02, bottom_text, ha='center', va='bottom', 
                 fontsize=10, wrap=True, style='italic', color='gray')
        plt.subplots_adjust(bottom=0.12)
    
    plt.show()