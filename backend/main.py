# backend/main.py
"""
FastAPI server exposing the core Photo Shield endpoints.

Endpoints:
* POST /protect            → shield generation (PGD or diffusion)
* POST /detect             → AI‑edit detector (ELA + classifier)
* POST /robustness         → robustness benchmark (PSNR/SSIM)
* POST /robust_protect     → Expectation‑over‑Transformation shield
* POST /report             → PDF evidence report generation
* GET  /health             → simple health check
"""

import io
import sys
import base64
from pathlib import Path
from typing import Optional

# Ensure both current directory and project root are on sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for p in (str(PROJECT_ROOT), str(BASE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    from backend.shield import protect_image, robust_protect, inpaint_image, apply_ai_attack
    from backend.detector import detect_edit
    from backend.robustness import robustness_report
    from backend.report import generate_report
except (ImportError, ModuleNotFoundError):
    from shield import protect_image, robust_protect, inpaint_image, apply_ai_attack
    from detector import detect_edit
    from robustness import robustness_report
    from report import generate_report

app = FastAPI(title="Photo Shield API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper – read raw bytes from an UploadFile
# ---------------------------------------------------------------------------
async def _read_bytes(up: UploadFile) -> bytes:
    return await up.read()

# ---------------------------------------------------------------------------
# 0️⃣ Inpaint endpoint (for live attack simulation — original photo)
# ---------------------------------------------------------------------------
@app.post("/inpaint")
async def api_inpaint(
    image: UploadFile = File(...),
    prompt: str = Form("stylish sunglasses on face"),
    edit_type: str = Form("glasses"),
    mask: Optional[UploadFile] = File(None),
    use_diffusion: bool = Form(False),
):
    img_bytes = await _read_bytes(image)
    mask_bytes = await _read_bytes(mask) if mask else None
    try:
        out = apply_ai_attack(
            image_bytes=img_bytes,
            mask_bytes=mask_bytes,
            prompt=prompt,
            edit_type=edit_type,
            is_shielded=False,
            use_diffusion=use_diffusion,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(out), media_type="image/png")


# ---------------------------------------------------------------------------
# 0b️⃣ Shield disruption endpoint (for protected photo side of the demo)
# ---------------------------------------------------------------------------
@app.post("/inpaint_shielded")
async def api_inpaint_shielded(
    image: UploadFile = File(...),
    prompt: str = Form("stylish sunglasses on face"),
    edit_type: str = Form("glasses"),
    mask: Optional[UploadFile] = File(None),
):
    img_bytes = await _read_bytes(image)
    mask_bytes = await _read_bytes(mask) if mask else None
    try:
        out = apply_ai_attack(
            image_bytes=img_bytes,
            mask_bytes=mask_bytes,
            prompt=prompt,
            edit_type=edit_type,
            is_shielded=True,
            use_diffusion=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(out), media_type="image/png")

# ---------------------------------------------------------------------------
# 1️⃣ Shield endpoint
# ---------------------------------------------------------------------------
@app.post("/protect")
async def api_protect(
    image: UploadFile = File(...),
    strength: float = Form(0.02),
    iterations: int = Form(10),
    mode: str = Form("pgd"),
    mask: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
):
    img_bytes = await _read_bytes(image)
    mask_bytes = await _read_bytes(mask) if mask else None
    try:
        out = protect_image(
            img_bytes,
            strength=strength,
            iterations=iterations,
            mode=mode,
            mask_bytes=mask_bytes,
            prompt=prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StreamingResponse(io.BytesIO(out), media_type="image/png")

# ---------------------------------------------------------------------------
# 2️⃣ Detector endpoint
# ---------------------------------------------------------------------------
@app.post("/detect")
async def api_detect(image: UploadFile = File(...)):
    img_bytes = await _read_bytes(image)
    try:
        res = detect_edit(img_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    # Encode binary PNGs as base64 for JSON transport
    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode()
    return {
        "ela_image_b64": b64(res["ela_image"]),
        "heatmap_b64": b64(res["heatmap"]),
        "fake_confidence": res["fake_confidence"],
        "label": res["label"],
    }

# ---------------------------------------------------------------------------
# 3️⃣ Robustness benchmark endpoint
# ---------------------------------------------------------------------------
@app.post("/robustness")
async def api_robustness(
    original: UploadFile = File(...),
    shielded: UploadFile = File(...),
):
    orig_bytes = await _read_bytes(original)
    shield_bytes = await _read_bytes(shielded)
    report = robustness_report(orig_bytes, shield_bytes)
    return report

# ---------------------------------------------------------------------------
# 4️⃣ Expectation‑over‑Transformation (EOT) shield endpoint
# ---------------------------------------------------------------------------
@app.post("/robust_protect")
async def api_robust_protect(
    image: UploadFile = File(...),
    strength: float = Form(0.02),
    iterations: int = Form(10),
):
    img_bytes = await _read_bytes(image)
    out = robust_protect(img_bytes, strength=strength, iterations=iterations, mode="pgd")
    return StreamingResponse(io.BytesIO(out), media_type="image/png")

# ---------------------------------------------------------------------------
# 5️⃣ PDF evidence report endpoint
# ---------------------------------------------------------------------------
@app.post("/report")
async def api_report(
    original: UploadFile = File(...),
    shielded: UploadFile = File(...),
    heatmap: UploadFile = File(...),
    fake_confidence: float = Form(...),
):
    orig = await _read_bytes(original)
    shield = await _read_bytes(shielded)
    hm = await _read_bytes(heatmap)
    pdf = generate_report(orig, shield, hm, fake_confidence)
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}