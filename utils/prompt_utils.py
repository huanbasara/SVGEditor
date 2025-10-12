import os

# ========== Prompt Configuration ==========
EDIT_INSTRUCTIONS = {
    # Legacy examples
    "apple": "Remove the green leaves from the apple",
    "bonsai": "Make the branches longer",
    "daisy": "Make the center of the daisy flower smaller", 
    "icecream": "Make the cone longer",
    "lighthouse": "Lower the sea level",
    "penguin": "Make the penguin fatter",
    
    # Batch targets for vectorization workflow
    "41_A": "Transform the armor into casual dress clothing",
    "41_B": "Transform the armor into casual dress clothing",
    "65_A": "Remove the scarf and reveal the neck underneath",
    "65_B": "Remove the scarf and reveal the neck underneath",
    "153_A": "Change long ponytail to short layered hair",
    "153_B": "Change long ponytail to short layered hair",
    "254_A": "Transform the braided pigtails into straight long hair",
    "254_B": "Transform the braided pigtails into straight long hair",
    "710_A": "Change angry expression to happy expression",
    "710_B": "Change angry expression to happy expression",
    "1061_A": "Transform conflict scene into hugging scene",
    "1061_B": "Transform conflict scene into hugging scene",
}

STYLE_REQUIREMENTS = (
    "Maintain black line art style on white background. "
    "Keep unmodified parts with original line structure. "
    "No color fills, no shading, only clean outlines."
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, "
    "colored lines, grayscale variations, rough edges, "
    "solid color fill, filled shapes, black fill, color blocks, shading, gradient, "
    "painting style, illustration style, anime coloring, cel shading"
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