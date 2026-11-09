# tests/test_shield.py
"""
Basic pytest suite for the shield module.
Runs quickly on CPU – useful for CI sanity checks.
"""
import io

import pytest
from PIL import Image

from backend.shield import protect_image

def _make_image(color=(255, 0, 0), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_protect_returns_png():
    raw = _make_image()
    out = protect_image(raw, strength=0.02, iterations=2, mode="pgd")
    result = Image.open(io.BytesIO(out))
    assert result.format == "PNG"
    assert result.size == (64, 64)

def test_strength_affects_output():
    raw = _make_image()
    weak = protect_image(raw, strength=0.01, iterations=2, mode="pgd")
    strong = protect_image(raw, strength=0.04, iterations=2, mode="pgd")
    assert weak != strong
