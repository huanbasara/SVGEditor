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
            "Transform this girl's long ponytail into a short bob haircut. "
            "Make the hair end at chin level, remove all the long flowing hair. "
            "Keep everything else identical: same pose, same pointing gesture, "
            "same facial expression, same outfit."
        ),
    }
    
    edit_instruction = edit_instructions[target_name]
    
    # Optimized template for SVG-friendly editing
    style_requirements = (
        "Maintain the exact same minimalist black line art style on pure white background. "
        "Keep all unmodified parts unchanged. "
        "Use uniform black lines with consistent thickness throughout the entire image. "
        "Ensure all lines are clear, sharp, and of the same color depth. "
        "Avoid any variations in line weight or opacity."
    )
    
    prompt = f"{edit_instruction} {style_requirements}"
    
    return prompt
