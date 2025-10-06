"""
Simple OpenAI image editing utility
"""

import os
import requests
from openai import OpenAI

def edit_image_with_openai(image_path: str, instruction: str, api_key: str, model: str = "dall-e-2", size: str = "256x256") -> str:
    """
    Edit image using OpenAI API
    
    Args:
        image_path: Input image path
        instruction: Edit instruction
        api_key: OpenAI API key
        model: Model name (default: dall-e-2)
        size: Image size (default: 256x256)
        
    Returns:
        Edited image URL
    """
    try:
        client = OpenAI(api_key=api_key)
        
        with open(image_path, "rb") as image_file:
            response = client.images.edit(
                image=image_file,
                prompt=instruction,
                model=model,
                size=size,
                n=1
            )
        
        if response.data:
            return response.data[0].url
        else:
            return ""
            
    except Exception as e:
        return ""

def download_image(url: str, save_path: str) -> bool:
    """
    Download image from URL
    
    Args:
        url: Image URL
        save_path: Save path
        
    Returns:
        Success status
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        return False
    
