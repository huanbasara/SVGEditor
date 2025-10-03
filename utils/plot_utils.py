import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_images(images_and_titles):
    """
    Display images in a horizontal row with titles.
    
    Args:
        images_and_titles: List of tuples containing (image, title)
    """
    n_images = len(images_and_titles)
    
    if n_images == 0:
        return
    
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
    
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
    plt.show()
