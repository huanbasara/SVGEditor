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
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', image_size=256):
        self.device = device
        self.image_size = image_size
        
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
        # Auto-detect image sizes and use the smaller one for consistent processing
        if isinstance(img_before, str):
            before_img = Image.open(img_before)
        else:
            before_img = img_before
            
        if isinstance(img_after, str):
            after_img = Image.open(img_after)
        else:
            after_img = img_after
        
        # Get image sizes and use the smaller dimension
        before_size = min(before_img.size)
        after_size = min(after_img.size)
        target_size = min(before_size, after_size)
        
        # Load and resize images to the smaller size
        img_before = self._load_image(img_before, target_size=target_size)
        img_after = self._load_image(img_after, target_size=target_size)
        
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
    
    def _load_image(self, img, target_size=None):
        """Load image from path or return PIL Image"""
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        else:
            img = img.convert('RGB')
        
        # If target_size is provided, resize to that size
        if target_size is not None:
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        elif self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        return img
    
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
        # Images are already resized to consistent size in evaluate()
        # Convert to tensors [-1, 1]
        img_before_tensor = torch.from_numpy(np.array(img_before)).permute(2, 0, 1).float() / 127.5 - 1
        img_after_tensor = torch.from_numpy(np.array(img_after)).permute(2, 0, 1).float() / 127.5 - 1
        
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


def evaluate_single_image(img_before_path, img_after_path, text_source, text_target, edit_prompt, device='cuda', image_size=256):
    """
    Convenience function for single image evaluation
    
    Args:
        img_before_path: Path to original image
        img_after_path: Path to edited image
        text_source: Source attribute description
        text_target: Target attribute description
        edit_prompt: Full edit instruction
        device: Device to run on ('cuda' or 'cpu')
        image_size: Image size for processing (default: 256)
    
    Returns:
        dict: Evaluation scores
    """
    evaluator = DiffusionEvaluator(device=device, image_size=image_size)
    return evaluator.evaluate(img_before_path, img_after_path, text_source, text_target, edit_prompt)


class VectorizationVisualEvaluator:
    """
    Visual quality evaluator for vectorization process
    
    Evaluates the visual quality of vectorized images (skeleton or SVG rendered)
    against the original image, focusing on:
    1. Structural similarity (SSIM)
    2. Aesthetic quality (CLIP-based)
    3. Line quality (continuity & uniformity)
    4. Clarity (Laplacian variance)
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load CLIP model for aesthetic evaluation
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Weights for overall score
        self.weights = {
            'structure_similarity': 0.40,  # SSIM - most important
            'aesthetic_quality': 0.30,     # Visual beauty
            'line_quality': 0.20,          # Line continuity & uniformity
            'clarity': 0.10               # Contrast & sharpness
        }
    
    def evaluate(self, original_img, vectorized_img):
        """
        Evaluate vectorization visual quality
        
        Args:
            original_img: Original image (path or PIL Image or numpy array)
            vectorized_img: Vectorized image (path or PIL Image or numpy array)
            
        Returns:
            dict: Comprehensive visual quality scores
        """
        # Load images
        original = self._load_image(original_img)
        vectorized = self._load_image(vectorized_img)
        
        # Ensure same size
        if original.shape != vectorized.shape:
            vectorized = cv2.resize(vectorized, (original.shape[1], original.shape[0]))
        
        # 1. Structural Similarity (SSIM)
        ssim_score = self._compute_ssim(original, vectorized)
        
        # 2. Aesthetic Quality (CLIP-based)
        aesthetic_score = self._compute_aesthetic_quality(vectorized)
        
        # 3. Line Quality
        line_quality_score = self._compute_line_quality(vectorized)
        
        # 4. Clarity (Laplacian variance)
        clarity_score = self._compute_clarity(vectorized)
        
        # Overall score (weighted average)
        overall_score = (
            self.weights['structure_similarity'] * ssim_score +
            self.weights['aesthetic_quality'] * aesthetic_score +
            self.weights['line_quality'] * line_quality_score +
            self.weights['clarity'] * clarity_score
        )
        
        return {
            'structure_similarity': float(ssim_score),
            'aesthetic_quality': float(aesthetic_score),
            'line_quality': float(line_quality_score),
            'clarity': float(clarity_score),
            'overall_score': float(overall_score),
            'quality_level': self._get_quality_level(overall_score)
        }
    
    def _load_image(self, img):
        """Load image from various formats to grayscale numpy array"""
        if isinstance(img, str):
            # Load from path
            img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                raise ValueError(f"Failed to load image from {img}")
            return img_array
        elif isinstance(img, Image.Image):
            # Convert PIL Image to numpy
            return np.array(img.convert('L'))
        elif isinstance(img, np.ndarray):
            # Already numpy array
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
    
    def _compute_ssim(self, original, vectorized):
        """
        Compute Structural Similarity Index (SSIM)
        
        Measures structural similarity between original and vectorized images
        Range: [-1, 1], higher is better
        """
        similarity = ssim(original, vectorized)
        # Normalize to [0, 1]
        return max(0.0, (similarity + 1) / 2)
    
    def _compute_aesthetic_quality(self, vectorized_img):
        """
        Compute aesthetic quality using CLIP
        
        Evaluates how well the image matches aesthetic quality descriptions
        """
        # Convert to PIL RGB for CLIP
        if len(vectorized_img.shape) == 2:
            pil_img = Image.fromarray(vectorized_img).convert('RGB')
        else:
            pil_img = Image.fromarray(vectorized_img)
        
        # Aesthetic prompts
        positive_prompts = [
            "high quality line art",
            "clean professional drawing",
            "smooth elegant illustration"
        ]
        negative_prompts = [
            "low quality sketch",
            "messy rough drawing",
            "blurry distorted image"
        ]
        
        with torch.no_grad():
            # Get image embedding
            img_inputs = self.clip_processor(images=[pil_img], return_tensors="pt", padding=True)
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
            img_features = self.clip_model.get_image_features(**img_inputs)
            img_emb = F.normalize(img_features, dim=-1)
            
            # Get positive prompt embeddings
            pos_inputs = self.clip_processor(text=positive_prompts, return_tensors="pt", padding=True)
            pos_inputs = {k: v.to(self.device) for k, v in pos_inputs.items()}
            pos_features = self.clip_model.get_text_features(**pos_inputs)
            pos_emb = F.normalize(pos_features, dim=-1)
            
            # Get negative prompt embeddings
            neg_inputs = self.clip_processor(text=negative_prompts, return_tensors="pt", padding=True)
            neg_inputs = {k: v.to(self.device) for k, v in neg_inputs.items()}
            neg_features = self.clip_model.get_text_features(**neg_inputs)
            neg_emb = F.normalize(neg_features, dim=-1)
            
            # Compute similarities
            pos_sim = F.cosine_similarity(img_emb, pos_emb.mean(dim=0, keepdim=True)).item()
            neg_sim = F.cosine_similarity(img_emb, neg_emb.mean(dim=0, keepdim=True)).item()
        
        # Combine scores: emphasize positive, penalize negative
        aesthetic_score = (pos_sim - neg_sim + 2) / 4  # Normalize to [0, 1]
        return max(0.0, min(1.0, aesthetic_score))
    
    def _compute_line_quality(self, vectorized_img):
        """
        Compute line quality: continuity and uniformity
        
        Evaluates:
        1. Line continuity (fewer endpoints = more continuous)
        2. Line width uniformity (lower variance = more uniform)
        """
        from skimage.morphology import skeletonize
        
        # Binarize image
        _, binary = cv2.threshold(vectorized_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Extract skeleton
        skeleton = skeletonize(binary > 0)
        
        # 1. Continuity: count endpoints
        endpoints = self._find_skeleton_endpoints(skeleton)
        n_endpoints = len(endpoints)
        
        # Normalize: fewer endpoints = better continuity
        # Assume reasonable range: 0-100 endpoints
        continuity_score = 1.0 / (1.0 + n_endpoints / 20.0)
        
        # 2. Uniformity: measure line width variance
        if np.sum(binary > 0) > 0:
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            line_widths = dist_transform[binary > 0]
            width_variance = np.var(line_widths)
            
            # Normalize: lower variance = better uniformity
            uniformity_score = 1.0 / (1.0 + width_variance / 10.0)
        else:
            uniformity_score = 0.0
        
        # Combine scores
        line_quality = (continuity_score + uniformity_score) / 2
        
        return line_quality
    
    def _find_skeleton_endpoints(self, skeleton):
        """Find skeleton endpoints (pixels with only one neighbor)"""
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = np.where((skeleton == 1) & (neighbor_count == 11))
        
        return list(zip(endpoints[0], endpoints[1]))
    
    def _compute_clarity(self, vectorized_img):
        """
        Compute image clarity using Laplacian variance
        
        Higher variance = sharper edges = better clarity
        """
        # Compute Laplacian
        laplacian = cv2.Laplacian(vectorized_img, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Normalize using sigmoid
        # Typical range: 0-1000 for line art
        clarity_score = 1.0 / (1.0 + np.exp(-laplacian_var / 100))
        
        return clarity_score
    
    def _get_quality_level(self, overall_score):
        """Get quality level description from overall score"""
        if overall_score >= 0.8:
            return "Excellent"
        elif overall_score >= 0.65:
            return "Good"
        elif overall_score >= 0.5:
            return "Acceptable"
        else:
            return "Needs Improvement"


def evaluate_vectorization_visual(original_img, vectorized_img, device='cuda'):
    """
    Convenience function for vectorization visual quality evaluation
    
    Args:
        original_img: Original image (path or PIL Image or numpy array)
        vectorized_img: Vectorized image (path or PIL Image or numpy array)
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        dict: Visual quality evaluation scores
    """
    evaluator = VectorizationVisualEvaluator(device=device)
    return evaluator.evaluate(original_img, vectorized_img)

