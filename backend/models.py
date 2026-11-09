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
from diffusers import StableDiffusionInpaintPipeline
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
# Stable Diffusion Inpainting pipeline (VAE encoder is needed for the shield).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_sd_inpaint_pipe() -> StableDiffusionInpaintPipeline:
    """
    Load the Stable Diffusion Inpainting pipeline with cached VAE weights.
    The pipeline is set to `torch.float16` on GPU for speed.
    """
    model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()  # saves memory
    return pipe

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