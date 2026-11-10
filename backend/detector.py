"""
AI‑edit detector.

* Error Level Analysis (ELA) visualisation.
* HuggingFace deep‑fake classifier.
* Returns a heat‑map (binary mask) and an overall confidence score.
"""
import io
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from .models import get_detector_pipe, DEVICE

# --------------------------------------------------------------------------- #
# Error Level Analysis (ELA)
# --------------------------------------------------------------------------- #
def compute_ela(img: Image.Image, quality: int = 90) -> Image.Image:
    """Produce an ELA image: re‑save at `quality`, compute the absolute difference,
    then amplify the result for visualisation.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    jpeg = Image.open(buf)
    diff = ImageChops.difference(img, jpeg)
    max_diff = max(diff.getextrema()[0])
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela = ImageEnhance.Brightness(diff).enhance(scale)
    return ela

# --------------------------------------------------------------------------- #
# DeepFake classifier (binary: real / fake)
# --------------------------------------------------------------------------- #
def classify_image(img: Image.Image) -> tuple[float, str]:
    try:
        classifier, _ = get_detector_pipe()
        results = classifier(img)
        scores = {r["label"].lower(): r["score"] for r in results}
        fake_score = scores.get("fake", 0.0)
        real_score = scores.get("real", 0.0)
        label = "fake" if fake_score > real_score else "real"
        return fake_score, label
    except Exception as e:
        print(f"[⚠️] Classifier fallback: {e}")
        ela = compute_ela(img)
        gray = np.array(ela.convert("L"))
        high_diff_ratio = float((gray > 30).sum()) / float(gray.size)
        fake_score = min(0.92, max(0.08, high_diff_ratio * 3.5))
        label = "fake" if fake_score > 0.5 else "real"
        return fake_score, label

# --------------------------------------------------------------------------- #
# Heat‑map generation (simple threshold on ELA intensity)
# --------------------------------------------------------------------------- #
def generate_heatmap(ela_img: Image.Image, threshold: int = 30) -> Image.Image:
    gray = ela_img.convert("L")
    arr = np.array(gray)
    mask = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_edit(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ela = compute_ela(img)
    fake_conf, label = classify_image(img)
    heatmap = generate_heatmap(ela)
    def png_bytes(pil_img: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    return {
        "ela_image": png_bytes(ela),
        "heatmap": png_bytes(heatmap),
        "fake_confidence": fake_conf,
        "label": label,
    }
