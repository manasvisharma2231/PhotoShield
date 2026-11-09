"""
Robustness testing utilities.

Runs a series of JPEG‑compression, resizing, cropping and screenshot‑style
transformations on a shielded image, then measures PSNR/SSIM against the
original.  Also supports an “expectation‑over‑transformation” (EOT) PGD
variant that includes these transformations inside the attack loop.
"""
import io
import numpy as np
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import skimage.metrics

from .shield import pgd_vae_attack, diffusion_attack, protect_image, pil_to_tensor, tensor_to_pil
from .models import get_sd_inpaint_pipe, DEVICE

# --------------------------------------------------------------------------- #
# Simple transformations
# --------------------------------------------------------------------------- #
def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(buf)

def resize_image(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def crop_center(img: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def screenshot_simulation(img: Image.Image) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(blurred).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.02, arr.shape)
    arr = np.clip(arr + noise, 0, 1) * 255
    return Image.fromarray(arr.astype(np.uint8))

# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1).astype(np.float64) / 255.0
    arr2 = np.array(img2).astype(np.float64) / 255.0
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1).astype(np.float64) / 255.0
    arr2 = np.array(img2).astype(np.float64) / 255.0
    return skimage.metrics.structural_similarity(arr1, arr2, multichannel=True)

# --------------------------------------------------------------------------- #
# Robustness benchmark
# --------------------------------------------------------------------------- #
def robustness_report(
    original_bytes: bytes,
    shielded_bytes: bytes,
    transformations: List[Dict] = None,
) -> dict:
    """
    Apply each transformation to *shielded* image and compute PSNR/SSIM
    against the *original* image.

    `transformations` is a list of dicts:
        {"name": "jpeg-80", "func": lambda img: jpeg_compress(img, 80)}
    If omitted, a sensible default set is used.

    Returns:
        {"results": [{"name": str, "psnr": float, "ssim": float}, ...]}
    """
    if transformations is None:
        transformations = [
            {"name": "jpeg-30", "func": lambda i: jpeg_compress(i, 30)},
            {"name": "jpeg-70", "func": lambda i: jpeg_compress(i, 70)},
            {"name": "resize-0.8", "func": lambda i: resize_image(i, 0.8)},
            {"name": "crop-0.9", "func": lambda i: crop_center(i, 0.9)},
            {"name": "screenshot", "func": screenshot_simulation},
        ]

    orig = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    shielded = Image.open(io.BytesIO(shielded_bytes)).convert("RGB")

    results = []
    for tr in transformations:
        try:
            transformed = tr["func"](shielded)
            psnr = compute_psnr(orig, transformed)
            ssim = compute_ssim(orig, transformed)
            results.append({"name": tr["name"], "psnr": psnr, "ssim": ssim})
        except Exception as e:
            results.append({"name": tr["name"], "error": str(e)})
    return {"results": results}

# --------------------------------------------------------------------------- #
# EOT‑PGD (robustness‑aware shield)
# --------------------------------------------------------------------------- #
def robust_protect(
    image_bytes: bytes,
    strength: float = 0.02,
    iterations: int = 10,
    mode: str = "pgd",
) -> bytes:
    """
    Run PGD while randomly applying a transformation at each step
    (Expectation‑Over‑Transformation).  This yields a shield that survives
    the listed attacks better.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    eps = strength
    trans_fns = [
        lambda x: jpeg_compress(x, quality=np.random.choice([30, 60, 90])),
        lambda x: resize_image(x, scale=np.random.uniform(0.7, 1.0)),
        lambda x: crop_center(x, crop_ratio=np.random.uniform(0.8, 1.0)),
        lambda x: screenshot_simulation(x),
        lambda x: x,
    ]
    if mode != "pgd":
        raise ValueError("EOT robust mode is implemented only for pgd")
    pipe = get_sd_inpaint_pipe()
    vae = pipe.vae
    x = pil_to_tensor(img)
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
    target_lat = torch.zeros_like(latents)
    delta = torch.zeros_like(latents, requires_grad=True)
    step_size = eps / iterations
    for _ in range(iterations):
        perturbed = latents + delta
        # Decode, apply random transform, re‑encode
        with torch.no_grad():
            decoded = pipe.vae.decode(perturbed / pipe.vae.config.scaling_factor).sample
            pil = tensor_to_pil(decoded)
        transformed = np.random.choice(trans_fns)(pil)
        transformed_tensor = pil_to_tensor(transformed)
        with torch.no_grad():
            latents = vae.encode(transformed_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        loss = F.mse_loss(latents, target_lat)
        loss.backward()
        with torch.no_grad():
            delta += step_size * torch.sign(delta.grad)
            delta = torch.clamp(delta, -eps, eps)
            delta.grad.zero_()
    with torch.no_grad():
        final_lat = (latents + delta) / pipe.vae.config.scaling_factor
        shielded = pipe.vae.decode(final_lat).sample
    out_buf = io.BytesIO()
    tensor_to_pil(shielded).save(out_buf, format="PNG")
    return out_buf.getvalue()
