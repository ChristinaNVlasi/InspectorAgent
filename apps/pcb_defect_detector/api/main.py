"""
FastAPI Backend — Guided PCB Defect Detection
Two-stage RAG pipeline: classification (first_vb) → localization (second_vb)
"""
import io
import time
import uuid
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.guided_inspector import GuidedPCBInspector
from config import PCB_CONFIG

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PCB Defect Detection API — Guided RAG",
    description=(
        "Two-stage guided RAG pipeline: "
        "1) classify whole PCB (burned/corrosion/OK via first_vb), "
        "2) localise defect in targeted grid cells (second_vb)."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global inspector ──────────────────────────────────────────────────────────
inspector: Optional[GuidedPCBInspector] = None

MAX_FILE_SIZE   = 10 * 1024 * 1024
ALLOWED_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


@app.on_event("startup")
async def startup_event() -> None:
    global inspector
    try:
        logger.info("Loading GuidedPCBInspector …")
        inspector = GuidedPCBInspector()
        logger.info("GuidedPCBInspector ready.")
    except Exception as exc:
        logger.error(f"Inspector failed to load: {exc}", exc_info=True)


# ── utility ───────────────────────────────────────────────────────────────────

def _validate_image(file: UploadFile) -> Optional[str]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        return f"Invalid format. Allowed: {', '.join(ALLOWED_EXTS)}"
    return None


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "PCB Defect Detection API — Guided RAG",
        "version": "3.0.0",
        "status":  "online",
        "pipeline": "first_vb (classify) → second_vb (localize)",
        "inspector_loaded": inspector is not None,
    }


@app.get("/health")
async def health_check():
    return {
        "status":         "healthy",
        "inspector_ready": inspector is not None,
        "timestamp":       time.time(),
    }


@app.get("/stats")
async def get_stats():
    if inspector is None:
        raise HTTPException(status_code=503, detail="Inspector not initialised")
    return {"success": True, "data": inspector.get_database_stats()}


@app.post("/api/inspect")
async def inspect_pcb(
    image:      UploadFile = File(..., description="PCB image file"),
    board_type: Optional[str] = Form(None),
    step:       Optional[str] = Form(None),
    view_type:  Optional[str] = Form("left"),
):
    """
    Inspect a PCB image using the guided two-stage RAG pipeline.

    Returns:
        - overall_status:            OK / NOT_OK
        - classification:            burned / corrosion / OK
        - classification_confidence: similarity score from first_vb
        - target_cells:              which cells were inspected in stage 2
        - defects_found:             list of confirmed defect locations
        - visualization:             annotated image (base64 PNG)
        - cell_results:              per-cell details (zoomed crops + matches)
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"[{request_id}] Received: {image.filename}")

    if inspector is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error":   "Service temporarily unavailable",
                "message": "AI model not loaded. Please try again later.",
                "request_id": request_id,
            },
        )

    # ── validate ─────────────────────────────────────────────────────────
    error_msg = _validate_image(image)
    if error_msg:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid file", "message": error_msg,
                     "request_id": request_id},
        )

    contents = await image.read()

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error":   "Invalid image",
                "message": "Could not open image. Please upload a valid image file.",
                "request_id": request_id,
            },
        )

    # ── inspect directly from PIL Image (no temp-file save) ─────────────
    # Saving to disk as JPEG and reloading applies lossy recompression which
    # shifts pixel values, alters the HSV crop boundary, and misaligns the
    # 4×4 grid cells so defects land in the wrong cell.  Passing the PIL
    # Image object directly avoids the round-trip entirely.
    try:
        result = inspector.inspect_pcb(
            img,
            view_type=view_type or "left",
            return_cell_details=True,
        )
    except Exception as exc:
        logger.error(f"[{request_id}] Inspection failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error":   "Internal server error",
                "message": "An error occurred during inspection. Please try again.",
                "request_id": request_id,
            },
        )

    processing_time = time.time() - start_time

    # ── format response (backwards-compatible with existing frontend) ────
    defects_list    = result.get("defects_found", [])
    first_defect    = defects_list[0] if defects_list else None
    overall_conf    = (
        first_defect.get("confidence", 0.0)
        if first_defect
        else result.get("classification_confidence", 1.0)
    )

    response = {
        "success":        True,
        "request_id":     request_id,
        "mode":           "guided_rag",
        "processing_time": round(processing_time, 3),
        "data": {
            # ── frontend-compatible fields (unchanged keys) ─────────────
            "status":       result["overall_status"],
            "confidence":   overall_conf,
            "marked_image": result.get("visualization"),
            "grid_overlay": result.get("grid_overlay"),
            "zoomed_image": first_defect.get("zoomed_image") if first_defect else None,
            "defect_info":  first_defect,
            # ── enhanced fields ─────────────────────────────────────────
            "overall_status":             result["overall_status"],
            "classification":             result.get("classification", "OK"),
            "classification_confidence":  result.get("classification_confidence", 0.0),
            "message":                    result.get("message", ""),
            "target_cells":               result.get("target_cells", []),
            "defects_found":              defects_list,
            "defect_count":               result.get("defect_count", 0),
            "assessment": (
                "OK" if result["overall_status"] == "OK" else "DEFECTS_DETECTED"
            ),
            "total_cells":    result.get("total_cells", 16),
            "healthy_cells":  result.get("healthy_cells", 16),
            "view_type":      result.get("view_type", "left"),
            "cell_results":   result.get("cell_results", {}),
            "preprocessing":  result.get("preprocessing", {}),
            "original_size":  result.get("original_size", (0, 0)),
            "processed_size": result.get("processed_size", (0, 0)),
            "board_type":     board_type or PCB_CONFIG["board_type"],
            "step":           step,
        },
    }

    logger.info(
        f"[{request_id}] Done in {processing_time:.2f}s — "
        f"{result['overall_status']}, classification={result.get('classification')}"
    )
    return response


@app.post("/api/batch-inspect")
async def batch_inspect(
    images:     list[UploadFile] = File(...),
    board_type: Optional[str]    = Form(None),
):
    if len(images) > 10:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Too many images",
                     "message": "Maximum 10 images per batch"},
        )
    results = []
    for img in images:
        r = await inspect_pcb(image=img, board_type=board_type)
        results.append(r)
    return {"success": True, "batch_size": len(images), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8771, log_level="info")
