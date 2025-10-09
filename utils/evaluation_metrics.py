"""
Diffusion Model Evaluation Metrics
Provides comprehensive evaluation for image editing results
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import lpips


class DiffusionEvaluator:
    """Comprehensive evaluator for diffusion model image editing"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Weights configuration
        self.weights = {
            'edit_compliance': 0.50,
            'style_consistency': 0.30,
            'structural_plausibility': 0.10,
            'aesthetic_quality': 0.10
        }
    
    def evaluate(self, img_before, img_after, text_source, text_target, edit_prompt):
        """
        Evaluate image editing results
        
        Args:
            img_before: Path or PIL Image (before editing)
            img_after: Path or PIL Image (after editing)
            text_source: Source attribute text (e.g., "long ponytail hair")
            text_target: Target attribute text (e.g., "short layered hair")
            edit_prompt: Full edit instruction
            
        Returns:
            dict: Comprehensive evaluation scores
        """
        # Load images
        img_before = self._load_image(img_before)
        img_after = self._load_image(img_after)
        
        # 1. Edit Compliance
        directional_clip = self._directional_clip(img_before, img_after, text_source, text_target)
        clip_score = self._clip_score(img_after, edit_prompt)
        edit_compliance = 0.8 * directional_clip + 0.2 * clip_score
        
        # 2. Style Consistency
        clip_style = self._clip_style_similarity(img_after)
        edge_sim = self._edge_similarity(img_before, img_after)
        style_consistency = 0.33 * clip_style + 0.67 * edge_sim
        
        # 3. Structural Plausibility
        structural_score = self._lpips_score(img_before, img_after)
        
        # 4. Aesthetic Quality
        aesthetic_score = self._aesthetic_score(img_after)
        
        # Total score
        total_score = (
            self.weights['edit_compliance'] * edit_compliance +
            self.weights['style_consistency'] * style_consistency +
            self.weights['structural_plausibility'] * structural_score +
            self.weights['aesthetic_quality'] * aesthetic_score
        )
        
        return {
            'edit_compliance': float(edit_compliance),
            'directional_clip': float(directional_clip),
            'clip_score': float(clip_score),
            'style_consistency': float(style_consistency),
            'clip_style': float(clip_style),
            'edge_similarity': float(edge_sim),
            'structural_plausibility': float(structural_score),
            'aesthetic_quality': float(aesthetic_score),
            'total_score': float(total_score)
        }
    
    def _load_image(self, img):
        """Load image from path or return PIL Image"""
        if isinstance(img, str):
            return Image.open(img).convert('RGB')
        return img.convert('RGB')
    
    def _get_clip_embeddings(self, images=None, texts=None):
        """Get CLIP embeddings for images or texts"""
        with torch.no_grad():
            if images is not None:
                inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.clip_model.get_image_features(**inputs)
                return F.normalize(image_features, dim=-1)
            
            if texts is not None:
                inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.clip_model.get_text_features(**inputs)
                return F.normalize(text_features, dim=-1)
    
    def _directional_clip(self, img_before, img_after, text_source, text_target):
        """Compute Directional CLIP score"""
        img_before_emb = self._get_clip_embeddings(images=[img_before])
        img_after_emb = self._get_clip_embeddings(images=[img_after])
        text_source_emb = self._get_clip_embeddings(texts=[text_source])
        text_target_emb = self._get_clip_embeddings(texts=[text_target])
        
        img_direction = img_after_emb - img_before_emb
        text_direction = text_target_emb - text_source_emb
        
        similarity = F.cosine_similarity(img_direction, text_direction).item()
        return (similarity + 1) / 2  # Normalize to [0, 1]
    
    def _clip_score(self, img_after, edit_prompt):
        """Compute CLIP score between edited image and prompt"""
        img_emb = self._get_clip_embeddings(images=[img_after])
        text_emb = self._get_clip_embeddings(texts=[edit_prompt])
        similarity = F.cosine_similarity(img_emb, text_emb).item()
        return (similarity + 1) / 2  # Normalize to [0, 1]
    
    def _clip_style_similarity(self, img_after):
        """Compute CLIP style similarity (line art)"""
        style_description = "black and white line art sketch"
        img_emb = self._get_clip_embeddings(images=[img_after])
        style_emb = self._get_clip_embeddings(texts=[style_description])
        similarity = F.cosine_similarity(img_emb, style_emb).item()
        return (similarity + 1) / 2  # Normalize to [0, 1]
    
    def _edge_similarity(self, img_before, img_after):
        """Compute edge structure similarity using Canny + SSIM"""
        # Convert to grayscale numpy arrays
        img_before_np = np.array(img_before.convert('L'))
        img_after_np = np.array(img_after.convert('L'))
        
        # Canny edge detection
        edges_before = cv2.Canny(img_before_np, 100, 200)
        edges_after = cv2.Canny(img_after_np, 100, 200)
        
        # Compute SSIM on edge maps
        similarity = ssim(edges_before, edges_after)
        return max(0.0, similarity)  # Ensure non-negative
    
    def _lpips_score(self, img_before, img_after):
        """Compute LPIPS perceptual similarity"""
        # Resize to 256x256 for LPIPS
        img_before_resized = img_before.resize((256, 256), Image.Resampling.LANCZOS)
        img_after_resized = img_after.resize((256, 256), Image.Resampling.LANCZOS)
        
        # Convert to tensors [-1, 1]
        img_before_tensor = torch.from_numpy(np.array(img_before_resized)).permute(2, 0, 1).float() / 127.5 - 1
        img_after_tensor = torch.from_numpy(np.array(img_after_resized)).permute(2, 0, 1).float() / 127.5 - 1
        
        img_before_tensor = img_before_tensor.unsqueeze(0).to(self.device)
        img_after_tensor = img_after_tensor.unsqueeze(0).to(self.device)
        
        # Compute LPIPS distance
        with torch.no_grad():
            distance = self.lpips_model(img_before_tensor, img_after_tensor).item()
        
        # Convert to similarity score [0, 1]
        similarity = 1 / (1 + distance)
        return similarity
    
    def _aesthetic_score(self, img_after):
        """Compute aesthetic quality score using CLIP"""
        # Simple aesthetic scoring using CLIP
        positive_prompts = [
            "high quality artwork",
            "professional illustration",
            "clean and detailed drawing"
        ]
        negative_prompts = [
            "low quality",
            "blurry and distorted",
            "messy sketch"
        ]
        
        img_emb = self._get_clip_embeddings(images=[img_after])
        
        # Compute similarity with positive and negative prompts
        pos_emb = self._get_clip_embeddings(texts=positive_prompts)
        neg_emb = self._get_clip_embeddings(texts=negative_prompts)
        
        pos_sim = F.cosine_similarity(img_emb, pos_emb.mean(dim=0, keepdim=True)).item()
        neg_sim = F.cosine_similarity(img_emb, neg_emb.mean(dim=0, keepdim=True)).item()
        
        # Combine scores
        aesthetic_score = (pos_sim - neg_sim + 2) / 4  # Normalize to [0, 1]
        return max(0.0, min(1.0, aesthetic_score))


def evaluate_single_image(img_before_path, img_after_path, text_source, text_target, edit_prompt, device='cuda'):
    """
    Convenience function for single image evaluation
    
    Args:
        img_before_path: Path to original image
        img_after_path: Path to edited image
        text_source: Source attribute description
        text_target: Target attribute description
        edit_prompt: Full edit instruction
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        dict: Evaluation scores
    """
    evaluator = DiffusionEvaluator(device=device)
    return evaluator.evaluate(img_before_path, img_after_path, text_source, text_target, edit_prompt)

