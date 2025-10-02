import os


def get_prompt(target):
    """Generate complete edit prompt for target SVG"""
    target_name = os.path.basename(target)
    
    # Edit instructions for each target
    edit_instructions = {
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
    
    edit_instruction = edit_instructions[target_name]
    
    # Updated style requirements
    style_requirements = (
        "Maintain the exact same minimalist black line art style on pure white background. "
        "Convert all lines to uniform deep black color with consistent thickness. "
        "Keep all unmodified parts with their original line structure, position, and curvature. "
        "Ensure all lines are clear, smooth, and of the same color depth and weight. "
        "Avoid any variations in line weight or opacity within the new image."
    )
    
    prompt = f"{edit_instruction} {style_requirements}"
    
    return prompt


def get_negative_prompt():
    """Generate negative prompt for consistent line art quality"""
    return (
        "inconsistent line thickness, varying line weight, "
        "gradient lines, non-uniform line opacity, "
        "thick and thin lines mixed, uneven line density, "
        "color variations in lines, non-black lines, "
        "colored lines, gradient effects, "
        "white background noise, speckles, dots, "
        "artifacts, jagged edges, broken lines, "
        "discontinuous lines, blurry lines, "
        "low quality, distorted, messy"
    )