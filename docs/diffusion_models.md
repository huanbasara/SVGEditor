# Diffusion Models for SVG Editing

## 概述
本文档总结了四种可用于 SVG 编辑的 diffusion 模型方案，包括当前使用的 InstructPix2Pix 以及三种备选方案。

## 模型方案对比

| 模型 | 类型 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|----------|
| InstructPix2Pix | 指令式编辑 | 简单易用，支持自然语言指令 | 效果不稳定，容易偏离原图 | 快速原型，简单编辑 |
| Stable Diffusion Img2Img | 图像到图像 | 稳定可靠，精确控制 | 需要调整 strength 参数 | 小幅修改，保持结构 |
| Stable Diffusion Inpainting | 局部编辑 | 精确局部控制，稳定性高 | 需要手动创建 mask | 局部形状/颜色修改 |
| ControlNet (Canny) | 结构约束 | 强力锁定边缘结构 | 需要边缘检测预处理 | 简洁线条，动画风格 |

## 1. InstructPix2Pix (当前方案)

### 模型信息
- **模型名称**: `timbrooks/instruct-pix2pix`
- **模型类型**: 指令式图像编辑
- **文件大小**: ~2.3GB
- **Google Drive 路径**: `/content/drive/MyDrive/SVGEditor/models/instruct-pix2pix/`

### 获取方式
```bash
# 使用 huggingface_hub 下载
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="timbrooks/instruct-pix2pix",
    local_dir="/content/drive/MyDrive/SVGEditor/models/instruct-pix2pix"
)
```

### 使用示例
```python
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "/content/drive/MyDrive/SVGEditor/models/instruct-pix2pix",
    dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 编辑图像
images = pipe(
    "Make the tree's branches thicker and more robust",
    image=pil_image,
    num_inference_steps=10,
    image_guidance_scale=2
).images
```

### 优缺点
- ✅ 支持自然语言指令
- ✅ 使用简单
- ❌ 效果不稳定
- ❌ 容易偏离原图结构
- ❌ 容易生成重复图案

## 2. Stable Diffusion Img2Img (SDEdit)

### 模型信息
- **模型名称**: `runwayml/stable-diffusion-v1-5`
- **模型类型**: 图像到图像生成
- **文件大小**: ~4.2GB
- **Google Drive 路径**: `/content/drive/MyDrive/SVGEditor/models/stable-diffusion-v1-5/`

### 获取方式
```bash
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="/content/drive/MyDrive/SVGEditor/models/stable-diffusion-v1-5"
)
```

### 使用示例
```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/content/drive/MyDrive/SVGEditor/models/stable-diffusion-v1-5",
    dtype=torch.float16
).to("cuda")

# 编辑图像
result = pipe(
    prompt="make the circle red, keep clean edges, flat colors, no textures",
    image=init_img,
    strength=0.35,  # 低 strength 保持原图结构
    guidance_scale=7.5
)
```

### 优缺点
- ✅ 稳定可靠
- ✅ 通过 strength 精确控制保结构 vs 改动幅度
- ✅ 适合小幅修改
- ❌ 需要调整 strength 参数
- ❌ 全图处理，不够精确

## 3. Stable Diffusion Inpainting

### 模型信息
- **模型名称**: `runwayml/stable-diffusion-inpainting`
- **模型类型**: 局部图像编辑
- **文件大小**: ~4.2GB
- **Google Drive 路径**: `/content/drive/MyDrive/SVGEditor/models/stable-diffusion-inpainting/`

### 获取方式
```bash
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    local_dir="/content/drive/MyDrive/SVGEditor/models/stable-diffusion-inpainting"
)
```

### 使用示例
```python
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "/content/drive/MyDrive/SVGEditor/models/stable-diffusion-inpainting",
    dtype=torch.float16
).to("cuda")

# 编辑图像
result = pipe(
    prompt="replace the star with a heart, flat vector look, sharp edges",
    image=image,
    mask_image=mask,  # 白=要改,黑=保留
    guidance_scale=7.5
)
```

### 优缺点
- ✅ 精确局部控制
- ✅ 稳定性高
- ✅ 适合局部形状/颜色修改
- ❌ 需要手动创建 mask
- ❌ 需要额外的 mask 生成步骤

## 4. ControlNet (Canny)

### 模型信息
- **模型名称**: `lllyasviel/sd-controlnet-canny`
- **基础模型**: `runwayml/stable-diffusion-v1-5`
- **模型类型**: 结构约束生成
- **文件大小**: ~1.4GB (ControlNet) + ~4.2GB (基础模型)
- **Google Drive 路径**: 
  - `/content/drive/MyDrive/SVGEditor/models/controlnet-canny/`
  - `/content/drive/MyDrive/SVGEditor/models/stable-diffusion-v1-5/`

### 获取方式
```bash
from huggingface_hub import snapshot_download

# 下载 ControlNet 模型
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-canny",
    local_dir="/content/drive/MyDrive/SVGEditor/models/controlnet-canny"
)

# 下载基础模型
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="/content/drive/MyDrive/SVGEditor/models/stable-diffusion-v1-5"
)
```

### 使用示例
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import cv2
import numpy as np

# 加载 ControlNet
controlnet = ControlNetModel.from_pretrained(
    "/content/drive/MyDrive/SVGEditor/models/controlnet-canny",
    dtype=torch.float16
)

# 加载基础模型
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "/content/drive/MyDrive/SVGEditor/models/stable-diffusion-v1-5",
    controlnet=controlnet,
    dtype=torch.float16
).to("cuda")

# 生成 Canny 边缘图
img = cv2.imread("rasterized_svg.png")
edges = cv2.Canny(img, 100, 200)
cond = Image.fromarray(edges)

# 编辑图像
result = pipe(
    prompt="change the triangle to a circle; keep flat colors; anime/cartoon style; no texture",
    image=cond,
    controlnet_conditioning_scale=0.9,
    guidance_scale=7.5,
    num_inference_steps=30
)
```

### 优缺点
- ✅ 强力锁定边缘结构
- ✅ 适合简洁线条/动画风格
- ✅ 可组合多种条件 (Multi-ControlNet)
- ❌ 需要边缘检测预处理
- ❌ 设置相对复杂

## 推荐使用顺序

1. **Stable Diffusion Img2Img**: 最稳定，适合小幅修改
2. **Stable Diffusion Inpainting**: 精确局部编辑
3. **ControlNet (Canny)**: 需要保持精确结构时
4. **InstructPix2Pix**: 快速原型，简单指令

## 模型下载脚本

```python
# 下载所有模型的脚本
from huggingface_hub import snapshot_download
import os

MODEL_BASE_PATH = "/content/drive/MyDrive/SVGEditor/models"

models_config = {
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "stable-diffusion-inpainting": "runwayml/stable-diffusion-inpainting",
    "controlnet-canny": "lllyasviel/sd-controlnet-canny"
}

def download_all_models():
    for model_name, model_id in models_config.items():
        model_path = f"{MODEL_BASE_PATH}/{model_name}"
        print(f"Downloading {model_name}...")
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path
        )
        print(f"✅ {model_name} downloaded successfully")

# 执行下载
download_all_models()
```

## 下一步计划

1. **测试 Img2Img**: 先测试最稳定的方案
2. **测试 Inpainting**: 如果需要精确局部编辑
3. **测试 ControlNet**: 如果需要保持精确结构
4. **对比效果**: 评估各模型在 SVG 编辑中的表现
5. **选择最佳方案**: 根据效果选择最适合的模型

## 注意事项

- 所有模型都需要 GPU 支持
- 建议使用 Colab Pro 或更高版本获得更好的 GPU
- 模型文件较大，确保 Google Drive 有足够空间
- 首次运行需要下载模型，后续会使用缓存
