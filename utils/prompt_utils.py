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
    "41_A": (
        "Transform the armor into casual dress clothing. "
        "Replace the metallic armor pieces with soft fabric dress. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "41_B": (
        "Transform the armor into casual dress clothing. "
        "Replace the metallic armor pieces with soft fabric dress. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "65_A": (
        "Remove the scarf and reveal the neck underneath. "
        "Show the clear neck line and collar area. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "65_B": (
        "Remove the scarf and reveal the neck underneath. "
        "Show the clear neck line and collar area. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "153_A": (
        "Transform this girl's long ponytail into a chic, shoulder-length layered cut. "
        "Remove all long flowing hair. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "153_B": (
        "Transform this girl's long ponytail into a chic, shoulder-length layered cut. "
        "Remove all long flowing hair. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "254_A": (
        "Transform the braided pigtails into straight long hair. "
        "Replace the braided style with smooth, flowing straight hair. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "254_B": (
        "Transform the braided pigtails into straight long hair. "
        "Replace the braided style with smooth, flowing straight hair. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "710_A": (
        "Change the angry facial expression to a happy expression. "
        "Transform frowning mouth to smiling, relaxed eyebrows. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "710_B": (
        "Change the angry facial expression to a happy expression. "
        "Transform frowning mouth to smiling, relaxed eyebrows. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "1061_A": (
        "Transform the conflict scene into a hugging scene. "
        "Change the fighting poses to embracing poses. "
        "Ensure clean, continuous lines without artifacts."
    ),
    "1061_B": (
        "Transform the conflict scene into a hugging scene. "
        "Change the fighting poses to embracing poses. "
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