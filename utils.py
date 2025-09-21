"""
SVG处理工具函数
"""

import cairosvg
import io
import os
from PIL import Image
import numpy as np


def read_svg_file(svg_path):
    """从SVG文件路径读取SVG代码"""
    with open(svg_path, 'r', encoding='utf-8') as f:
        return f.read()


def svg_code_to_pil_image(svg_code, width=None, height=None, dpi=300):
    """将SVG代码转换为PIL Image对象"""
    png_bytes = cairosvg.svg2png(
        bytestring=svg_code.encode('utf-8'),
        output_width=width,
        output_height=height,
        dpi=dpi
    )
    return Image.open(io.BytesIO(png_bytes))


def save_pil_image(pil_image, output_path, filename):
    """保存PIL图像到指定路径"""
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    pil_image.save(full_path)
    return full_path

def get_prompt(target):
    files_to_captions = {
        "apple": "A red apple with green leaves and stems",
        "bonsai": "A bonsai tree in a pot",
        "daisy": "A daisy flower",
        "icecream": "An ice cream cone with three scoops",
        "lighthouse": "A lighthouse by the sea",
        "penguin": "A penguin standing on ice",
    }
    return files_to_captions[os.path.basename(target)]

def get_edit_prompts(target):
    # 获取原始prompt
    original_prompt = get_prompt(target)
    
    # 获取编辑提示词
    files_to_edit_prompts = {
        "apple": [
            "remove the leaves",
            "add more leaves"
        ],
        "bonsai": [
            "make the branches longer",
            "make the green circles smaller"
        ],
        "daisy": [
            "make the center of the flower smaller",
            "add more leaves"
        ],
        "icecream": [
            "make the cone longer",
            "make the scoops smaller"
        ],
        "lighthouse": [
            "lower the sea level",
            "make the lighthouse wider"
        ],
        "penguin": [
            "make the penguin fatter"
        ]
    }
    
    # 拼接原始prompt和编辑prompt
    edit_prompts = files_to_edit_prompts[os.path.basename(target)]
    combined_prompts = []
    
    for edit_prompt in edit_prompts:
        combined_prompt = f"{original_prompt}, {edit_prompt}"
        combined_prompts.append(combined_prompt)
    
    return combined_prompts