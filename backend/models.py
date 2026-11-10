"""
Model loading & caching utilities.

All heavy models (Stable Diffusion, detector, etc.) are loaded once
and reused across requests.  A small CPU‑fallback is provided for
environments without a GPU.
"""

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification

# --------------------------------------------------------------------------- #
# Device handling – prefer CUDA if available, otherwise CPU (with a warning).
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[⚡] Apple Silicon detected – running on MPS (GPU acceleration).")
        return torch.device("mps")
    else:
        print("[⚠️] CUDA/MPS not available – running on CPU. Performance will be slower.")
        return torch.device("cpu")

DEVICE = get_device()

# --------------------------------------------------------------------------- #
# Lightweight VAE Loader (~330 MB instead of full 5GB pipeline)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_vae():
    """
    Loads only the AutoencoderKL VAE subfolder to drastically reduce RAM
    usage from ~5GB down to ~330MB, perfectly fitting within cloud memory limits.
    """
    try:
        model_id = "runwayml/stable-diffusion-inpainting"
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        vae = vae.to(DEVICE)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        return vae
    except Exception as e:
        print(f"[⚠️] Could not load VAE from HuggingFace: {e}")
        return None

# --------------------------------------------------------------------------- #
# Full Stable Diffusion Inpainting pipeline (optional, for live inpainting)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_sd_inpaint_pipe():
    """
    Load the Stable Diffusion Inpainting pipeline if requested for live inpainting.
    """
    try:
        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        return pipe
    except Exception as e:
        print(f"[⚠️] Could not load full diffusion pipeline: {e}")
        return None

# --------------------------------------------------------------------------- #
# Deepfake / AI‑edit detector – HuggingFace image‑classification pipeline.
# Using a model fine‑tuned for deepfake detection.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_detector_pipe() -> Tuple[pipeline, AutoFeatureExtractor]:
    """
    Returns a tuple `(classifier, feature_extractor)`.
    The default model is `microsoft/dit-base-finetuned-deepfake-detection`.
    """
    model_name = "microsoft/dit-base-finetuned-deepfake-detection"
    classifier = pipeline(
        "image-classification",
        model=model_name,
        device=DEVICE if DEVICE.type in ("cuda", "mps") else -1,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return classifier, feature_extractor