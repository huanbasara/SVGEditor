# Diffusion 图像编辑方案参考 3 - 指令式编辑模型

## 🎯 目标

- **输入**：图像（init image） + 指令（instruction prompt）
- **要求**：精准编辑，保持原图风格（尤其是动画/纯色块风格）
- **问题**：现有 IP-Adapter 方案主要改变颜色，结构编辑能力不足
- **方法**：探索专门的指令式编辑模型，提升编辑精准度

---

## 📦 环境依赖

```bash
pip install -U diffusers transformers accelerate safetensors controlnet-aux
```

需要 GPU (CUDA)。

---

## 🚀 推荐模型（按优先级排序）

### **第一优先级（容易集成）**

#### **1. InstructPix2Pix 系列**
```python
# 基础版本（已下载）
"timbrooks/instruct-pix2pix"

# SDXL版本（更高质量）
"diffusers/sdxl-instructpix2pix-768"

# 社区微调版本
"instruction-tuning/instruct-pix2pix-finetuned"
```

#### **2. 参数优化的 IP-Adapter**
```python
# 当前问题：编辑能力不足，主要改变颜色
# 解决方案：调整参数平衡
strength=0.5,           # 增加编辑强度（当前0.3）
guidance_scale=10.0,    # 增强文本引导（当前7.0）
ip_adapter_scale=0.3,   # 降低风格控制（当前0.8）
```

### **第二优先级（中等复杂度）**

#### **3. InstructPix2Pix + ControlNet 组合**
```python
# 结合指令编辑 + 结构保持
pipeline = StableDiffusionInstructPix2PixPipeline
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
```

#### **4. 专门的编辑模型**
```python
# 局部编辑专用
"ByteDance/MagicEdit"

# 增强指令理解
"instruction-tuning/InstructDiffusion"

# SAM精确编辑
"ShilongLiu/EditAnything"
```

### **第三优先级（最新技术）**

#### **5. 风格保持 + 指令编辑**
```python
# Google风格保持编辑
"google/styledrop-sd15"

# 结合风格保持和指令编辑
"DreamEdit/dreamedit-sd15"
```

#### **6. 最新强力模型**
```python
# 最新编辑模型
"black-forest-labs/FLUX.1-Fill-dev"

# SD3编辑版本
"stabilityai/stable-diffusion-3-medium-edit"
```

---

## 🔧 测试方案

### **方案 A: InstructPix2Pix（推荐首选）**

```python
from diffusers import StableDiffusionInstructPix2PixPipeline

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16
).to("cuda")

result = pipe(
    prompt="Remove the leaves from the apple",
    image=pil_image_256,
    num_inference_steps=20,
    image_guidance_scale=1.5,  # 保持原图相似度
    guidance_scale=7.5         # 文本指令强度
).images[0]
```

### **方案 B: 优化的 IP-Adapter**

```python
result = pipe(
    prompt=edit_prompt,  # 简化prompt，只用编辑指令
    negative_prompt="purple background, colored background",
    image=pil_image_256,
    ip_adapter_image=style_reference,
    strength=0.6,        # 增加编辑强度
    guidance_scale=12.0, # 增强文本引导
    ip_adapter_scale=0.2, # 大幅降低风格控制
    num_inference_steps=30
)
```

### **方案 C: ControlNet + IP-Adapter 双重控制**

```python
# 加载 ControlNet 保持结构
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15s2_lineart_anime"
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# 加载 IP-Adapter 保持风格
pipe.load_ip_adapter("h94/IP-Adapter", weight_name="ip-adapter_sd15.bin")

# 生成边缘控制图
import cv2
edges = cv2.Canny(np.array(pil_image_256), 50, 150)
control_image = Image.fromarray(edges)

result = pipe(
    prompt=edit_prompt,
    image=pil_image_256,
    control_image=control_image,
    ip_adapter_image=pil_image_256,
    strength=0.4,
    controlnet_conditioning_scale=0.8,
    ip_adapter_scale=0.5
)
```

---

## ⚙️ 调参策略

### **针对结构编辑不足的问题**

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `strength` | 0.3 | 0.5-0.6 | 增加编辑强度 |
| `guidance_scale` | 7.0 | 10.0-12.0 | 增强文本指令权重 |
| `ip_adapter_scale` | 0.8 | 0.2-0.4 | 降低风格控制，让编辑更自由 |
| `num_inference_steps` | 25 | 30-40 | 增加推理精度 |

### **针对颜色偏移的问题**

```python
negative_prompt = "purple background, colored background, wrong colors, color shift, tinted"
```

---

## 🔍 模型下载配置

### **更新 models_config**

```python
models_config = {
    # 现有模型
    "instruct-pix2pix": "timbrooks/instruct-pix2pix",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "stable-diffusion-xl-base": "stabilityai/stable-diffusion-xl-base-1.0",
    "ip-adapter": "h94/IP-Adapter",
    
    # 新增指令编辑模型
    "instruct-pix2pix-sdxl": "diffusers/sdxl-instructpix2pix-768",
    "controlnet-lineart-anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "magic-edit": "ByteDance/MagicEdit",
    
    # 实验性模型
    "flux-edit": "black-forest-labs/FLUX.1-Fill-dev",
    "sd3-edit": "stabilityai/stable-diffusion-3-medium-edit"
}
```

---

## 🧪 测试计划

### **阶段 1：快速验证**
1. 调整当前 IP-Adapter 参数
2. 切换到 InstructPix2Pix
3. 对比效果差异

### **阶段 2：组合测试**
1. InstructPix2Pix + ControlNet
2. 不同 ControlNet 类型测试
3. 参数网格搜索

### **阶段 3：新模型评估**
1. SDXL InstructPix2Pix
2. MagicEdit 专用编辑
3. 最新 FLUX/SD3 模型

---

## 📊 评估指标

### **编辑精准度**
- 是否按指令正确编辑（如真的移除叶子）
- 编辑区域是否精确
- 是否保持其他区域不变

### **风格保持度**
- 颜色是否保持原样
- 艺术风格是否一致
- 背景是否保持白色

### **图像质量**
- 分辨率和清晰度
- 是否有人工痕迹
- 整体视觉效果

---

## ✅ 下一步行动

1. **立即测试**：调整 IP-Adapter 参数
2. **并行测试**：InstructPix2Pix 基础版本
3. **深入研究**：ControlNet + IP-Adapter 组合
4. **长期规划**：评估新模型和下载部署

重点解决当前**编辑能力不足**和**颜色偏移**问题，逐步提升指令编辑的精准度。
