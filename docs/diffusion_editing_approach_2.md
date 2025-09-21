# Diffusion 图像编辑方案参考

## 🎯 目标

- **输入**：图像（init image） + 指令（instruction prompt）
- **要求**：精准编辑，保持原图风格（尤其是动画/纯色块风格）
- **方法**：使用 Stable Diffusion (SDXL / SD1.5) 的 **img2img** 管线，结合
  - **ControlNet** → 锁定结构
  - **IP-Adapter** → 保持风格

---

## 📦 环境依赖

```bash
pip install -U diffusers transformers accelerate safetensors controlnet-aux
```

需要 GPU (CUDA)。

---

## 🅰️ 方案 A: SDXL + ControlNet + IP-Adapter + Img2Img

```python
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import load_image

device = "cuda"

# 1) ControlNet (结构条件: canny/lineart/pose/seg)
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to(device)

# 2) SDXL + ControlNet 的 img2img 管线
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, torch_dtype=torch.float16
).to(device)

# 3) 加载 IP-Adapter (风格参考)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sdxl_vit-h.bin"
)

# 4) 输入
init_img   = Image.open("input.png").convert("RGB")      # 原图
control_im = Image.open("control.png").convert("RGB")    # 结构条件图 (线稿/边缘)
ref_style  = Image.open("ref_style.png").convert("RGB")  # 风格参考图

prompt = "Replace the left triangle with a red circle; flat colors; clean edges; anime style."
neg    = "photorealistic, texture, shading, gradients, noise, blur"

# 5) 推理
out = pipe(
    prompt=prompt,
    negative_prompt=neg,
    image=init_img,
    control_image=control_im,
    ip_adapter_image=ref_style,
    strength=0.35,
    guidance_scale=6.0,
    controlnet_conditioning_scale=0.9,
    ip_adapter_scale=0.9,
    num_inference_steps=30
)

out.images[0].save("edit_sdxl_cn_ip_i2i.png")
```

---

## 🅱️ 方案 B: SD1.5 + ControlNet (Anime/Lineart) + IP-Adapter + Img2Img

> 对动画/纯色块风格，SD1.5 的生态（ControlNet-Anime/Lineart）更成熟。

```python
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

device = "cuda"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15s2_lineart_anime", torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

# SD1.5 版 IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

init_img   = Image.open("input.png").convert("RGB")
control_im = Image.open("lineart.png").convert("RGB")
ref_style  = Image.open("ref_style.png").convert("RGB")

prompt = "Change the left triangle to a red circle; flat colors; crisp lines; vector-like, no textures."
neg    = "photorealistic, detailed textures, shading, gradients, noise, blur"

out = pipe(
    prompt=prompt, negative_prompt=neg,
    image=init_img, control_image=control_im, ip_adapter_image=ref_style,
    strength=0.3, guidance_scale=6.5,
    controlnet_conditioning_scale=0.95, ip_adapter_scale=0.9,
    num_inference_steps=30
)

out.images[0].save("edit_sd15_cn_ip_i2i.png")
```

---

## ⚙️ 调参要点

- **双锚法**：
  - 结构锚 → `control_image` + `controlnet_conditioning_scale`
  - 风格锚 → `ip_adapter_image` + `ip_adapter_scale`
- **strength (img2img 改动幅度)**：
  - 0.2–0.4 更贴近原图
  -
    > 0.6 容易风格跑偏
- **负提示 (negative prompt)**：
  - 强排写实化：`photorealistic, texture, shading, gradients, grain, blur`
- **inpaint 更稳**：局部改动可换 `InpaintPipeline`，加 `mask_image`
- **模型选择**：
  - **SDXL**：更高分辨率，但动画风格相对少
  - **SD1.5**：社区 anime/flat-color 模型多，风格更容易稳住

---

## 🔍 检索与资源

- **Hugging Face Models** 搜索关键词：

  - `controlnet lineart anime`
  - `controlnet canny sdxl`
  - `t2i-adapter lineart sdxl`
  - `ip-adapter sdxl`
  - `anime flat color`
  - `instruction image editing`

- **推荐作者/组织**：

  - `lllyasviel` → ControlNet 系列
  - `TencentARC` → T2I-Adapter, MasaCtrl
  - `instantX-research` → InstantStyle

- **GitHub 项目**：

  - [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
  - [InstantStyle](https://github.com/instantX-research/InstantStyle)
  - [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
  - [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt)
  - [MasaCtrl](https://github.com/TencentARC/MasaCtrl)

---

## ✅ 总结

- 用 **img2img** 保证“图像+指令”输入模式。
- **ControlNet** 保结构，**IP-Adapter** 保风格。
- SD1.5 更适合动画/纯色块，SDXL 分辨率高但风格模型少。
- 调好 `strength`、`guidance`、`scale` 三类参数，就能在“风格不跑偏”和“按指令编辑”之间找到平衡。

