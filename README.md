# SVG Diffusion - Diffusion Models for SVG Editing

A research project exploring the application of diffusion models for editing and manipulating SVG graphics, with a focus on anime/cartoon style transformations.

## 🎯 Project Goals

This project aims to:
- Apply diffusion models to SVG-based graphics editing
- Explore structural editing capabilities (e.g., making heads bigger, adjusting proportions)
- Maintain anime/cartoon artistic styles during transformations
- Combine multiple diffusion models for optimal results

## 📋 Research Approach

Based on our initial research, we're focusing on:

1. **InstructPix2Pix** - For structural editing with instruction-following capabilities
2. **Anime-specific models** (like aamXLAnimeMix) - For high-quality anime style generation
3. **Hybrid approach** - Using InstructPix2Pix for structure + Anime models for style

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd SVGEditor

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### Usage

The main research notebook is `SvgDiffusion.ipynb`. This notebook contains:
- Model loading and initialization
- Image preprocessing for SVG inputs
- Diffusion model inference pipelines
- Style transfer experiments
- Results visualization and comparison

## 📁 Project Structure

```
SVGEditor/
├── notebooks/           # Jupyter notebooks for experiments
│   └── SvgDiffusion.ipynb
├── src/                # Source code modules
│   ├── __init__.py
│   ├── models/         # Model loading and inference
│   ├── utils/          # Utility functions
│   └── data/           # Data processing utilities
├── data/               # Sample data and outputs
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
└── README.md          # This file
```

## 🔬 Research Models

### Primary Models
- **InstructPix2Pix**: Instruction-following image editing
- **aamXLAnimeMix_v10**: High-quality anime style generation
- **Stable Diffusion XL**: Base model for various tasks

### Experimental Approaches
- Prompt-to-Prompt (P2P) for fine-grained control
- BLIP-Diffusion for subject-driven editing
- Imagic for complex non-rigid transformations

## 📊 Expected Workflow

1. **Input**: SVG graphics (converted to raster)
2. **Structure Editing**: Use InstructPix2Pix with anime style prompts
3. **Style Enhancement**: Apply anime models for final polish
4. **Output**: High-quality edited graphics maintaining artistic style

## 🤝 Contributing

This is a research project. Feel free to:
- Experiment with different model combinations
- Add new diffusion model implementations
- Improve preprocessing and postprocessing pipelines
- Share results and findings

## 📄 License

MIT License - see LICENSE file for details.

## 🔧 SVG Optimization Pipeline

### SVG Simplification Algorithm

The `simplify()` function in `svglib` performs different operations depending on the input SVG type:

#### 1. For Line-based SVG (L commands)
When the SVG consists of straight lines (like skeleton extraction output):

**Step 1: RDP Algorithm (Ramer-Douglas-Peucker)**
- Reduces the number of points while preserving the overall shape
- Controlled by `tolerance` parameter (pixels)
- Higher tolerance = more aggressive simplification = fewer points

**Step 2: Bezier Curve Fitting**
- Converts simplified polylines into smooth Bezier curves (C commands)
- Uses least-squares fitting based on [Paper.js PathFitter](https://github.com/paperjs/paper.js/blob/develop/src/path/PathFitter.js)
- Controlled by `epsilon` parameter (fitting precision)
- Lower epsilon = more accurate curve fitting

**Parameters:**
```python
svg.simplify(
    tolerance=2.0,      # RDP simplification tolerance (pixels)
    epsilon=0.1,        # Bezier fitting precision
    force_smooth=True   # Force smooth curve generation
)
```

#### 2. For Bezier-based SVG (C commands)
When the SVG already contains Bezier curves:

**Additional Step: Curve Merging**
- Merges adjacent Bezier curves with similar tangent directions
- Controlled by `angle_threshold` parameter (degrees)
- Only merges curves when angle difference < threshold
- **Note:** This parameter has NO effect on line-based SVG

**Example:**
- `angle_threshold=179`: Only merge nearly straight curves (conservative)
- `angle_threshold=150`: More aggressive merging (may lose detail)

### Workflow Summary

```
Line-based SVG → RDP Simplification → Bezier Fitting → Optimized SVG
                 (tolerance)           (epsilon)

Bezier-based SVG → Curve Merging → Optimized SVG
                   (angle_threshold)
```

### Implementation Reference

The skeleton-to-SVG workflow uses:
1. **Skeleton Extraction**: Produces single-pixel wide paths
2. **Path Tracing**: Converts pixels to line segments (L commands)
3. **SVG Simplification**: Applies RDP + Bezier fitting
4. **Output**: Smooth, compact SVG with Bezier curves

## 📚 References

- InstructPix2Pix: [Paper](https://arxiv.org/abs/2211.09800)
- Stable Diffusion: [Paper](https://arxiv.org/abs/2112.10752)
- Anime models: [CivitAI Community](https://civitai.com)
- RDP Algorithm: [Ramer-Douglas-Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
- Bezier Curve Fitting: [Paper.js PathFitter](https://github.com/paperjs/paper.js/blob/develop/src/path/PathFitter.js)
