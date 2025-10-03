import os

# ========== Prompt Configuration ==========
EDIT_INSTRUCTIONS = {
    "apple": "Remove the green leaves from the apple",
    "bonsai": "Make the branches longer",
    "daisy": "Make the center of the daisy flower smaller", 
    "icecream": "Make the cone longer",
    "lighthouse": "Lower the sea level",
    "penguin": "Make the penguin fatter",
    "153_B": (
        "Transform this girl's long ponytail into a chic, shoulder-length layered cut. "
        "Remove all long flowing hair. "
        "Ensure clean, continuous lines without artifacts."
    ),
}

STYLE_REQUIREMENTS = (
    "Maintain the exact same minimalist black line art style on pure white background. "
    "Convert all lines to uniform deep black color with consistent thickness. "
    "Keep all unmodified parts with their original line structure, position, and curvature. "
    "Ensure all lines are clear, smooth, and of the same color depth and weight. "
    "Avoid any variations in line weight or opacity within the new image."
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, "
    "colored lines, grayscale variations, rough edges"
)


def get_edit_instruction(target):
    """Get edit instruction for target"""
    target_name = os.path.basename(target)
    return EDIT_INSTRUCTIONS.get(target_name, "")


def get_full_prompt(target):
    """Get complete prompt with style requirements"""
    instruction = get_edit_instruction(target)
    return f"{instruction} {STYLE_REQUIREMENTS}"


def get_negative_prompt():
    """Get negative prompt"""
    return NEGATIVE_PROMPT


# Legacy compatibility
def get_prompt(target):
    """Legacy function - returns full prompt with style requirements"""
    return get_full_prompt(target)