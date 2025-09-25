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
