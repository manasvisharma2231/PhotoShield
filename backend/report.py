"""
PDF evidence report generator.

Uses `reportlab` (pure Python) to produce a one‑page PDF containing:
* Original image
* Shielded image
* Heat‑map from detector
* SHA‑256 hash of the shielded image
* Timestamp
* Confidence score
* Short guidance for Indian cyber‑crime reporting
"""
import io
import hashlib
import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage

def _pil_to_rl(img_bytes: bytes, width_mm: float) -> RLImage:
    """Wrap PNG bytes as a ReportLab Image scaled to `width_mm`."""
    buf = io.BytesIO(img_bytes)
    return RLImage(buf, width=width_mm * mm, preserveAspectRatio=True, hAlign='CENTER')

def generate_report(
    original_bytes: bytes,
    shielded_bytes: bytes,
    heatmap_bytes: bytes,
    fake_confidence: float,
) -> bytes:
    """Return PDF bytes.

    Layout: two columns (original & shielded) on top, heat‑map on the right,
    then a footer box with metadata.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 20 * mm
    col_width = (width - 2 * margin) / 3

    # Top row: Original & Shielded
    orig_img = _pil_to_rl(original_bytes, width_mm=col_width)
    shield_img = _pil_to_rl(shielded_bytes, width_mm=col_width)
    orig_img.drawOn(c, margin, height - margin - orig_img.imageHeight)
    shield_img.drawOn(c, margin + col_width + 10 * mm, height - margin - shield_img.imageHeight)

    # Right column: Heat‑map
    heat_img = _pil_to_rl(heatmap_bytes, width_mm=col_width)
    heat_img.drawOn(c, margin + 2 * (col_width + 10 * mm), height - margin - heat_img.imageHeight)

    # Footer metadata
    footer_y = margin + 10 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, footer_y + 40, "Evidence Report")
    c.setFont("Helvetica", 10)

    sha = hashlib.sha256(shielded_bytes).hexdigest()
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"SHA‑256 (shielded image): {sha}",
        f"Generated on: {timestamp}",
        f"AI‑edit confidence (fake): {fake_confidence:.2%}",
        "",
        "Reporting Guidance (India):",
        " • Cyber Cell portal: https://cybercrime.gov.in",
        " • Helpline: 1930",
        " • Attach this PDF and the original & shielded images",
    ]
    for i, line in enumerate(lines):
        c.drawString(margin, footer_y + 20 - i * 12, line)

    c.showPage()
    c.save()
    return buf.getvalue()
