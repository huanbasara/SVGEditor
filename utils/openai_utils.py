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
        
        print(f"API Response: {response}")
        print(f"Response data: {response.data}")
        print(f"Response data length: {len(response.data) if response.data else 0}")
        
        if response.data and len(response.data) > 0:
            url = response.data[0].url
            print(f"Image URL: {url}")
            return url
        else:
            print("No data in response")
            return ""
            
    except Exception as e:
        print(f"Exception in edit_image_with_openai: {e}")
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
    
