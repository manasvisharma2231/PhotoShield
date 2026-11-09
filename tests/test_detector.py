# tests/test_detector.py
"""
Simple sanity‑checks for the AI‑edit detector.
"""
import io

import pytest
from PIL import Image

from backend.detector import detect_edit

def _make_image(gray=128, size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, (gray, gray, gray))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_detect_returns_expected_keys():
    raw = _make_image()
    out = detect_edit(raw)
    assert {"ela_image", "heatmap", "fake_confidence", "label"} <= out.keys()
    assert isinstance(out["fake_confidence"], float)
    assert out["label"] in {"real", "fake"}
