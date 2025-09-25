import os


def get_prompt(target):
    """Generate complete edit prompt for target SVG"""
    target_name = os.path.basename(target)
    
    # Original descriptions for each target
    original_descriptions = {
        "apple": "A red apple with green leaves and stems",
        "bonsai": "A bonsai tree in a pot", 
        "daisy": "A daisy flower",
        "icecream": "An ice cream cone with three scoops",
        "lighthouse": "A lighthouse by the sea",
        "penguin": "A penguin standing on ice",
    }
    
    # Edit instructions for each target
    edit_instructions = {
        "apple": "Remove the green leaves from the apple",
        "bonsai": "Make the branches longer",
        "daisy": "Make the center of the daisy flower smaller", 
        "icecream": "Make the cone longer",
        "lighthouse": "Lower the sea level",
        "penguin": "Make the penguin fatter",
    }
    
    original_description = original_descriptions[target_name]
    edit_instruction = edit_instructions[target_name]
    
    # Template for complete edit prompt
    prompt = f"Original picture is {original_description}. Edit instruction is {edit_instruction}. Do not change other parts of the picture, follow the drawing style of the original picture."
    
    return prompt
