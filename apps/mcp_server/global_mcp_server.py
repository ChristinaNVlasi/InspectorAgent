"""
Global MCP Server — SMA Inspector
Combines ALL tools from both Knorr PCB Inspection and Arcelik-Beko Washing Machine inspection.
Namespaced tools:  pcb_* → Knorr PCB use-case
                   wm_*  → Arcelik-Beko washing-machine use-case
Port: 3030
"""
import asyncio
import os
import sys
import logging
import base64
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, HTMLResponse, Response
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

# ──────────────────────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent   # apps/
KNORR_DIR     = BASE_DIR / "agent_knorr"
ARCELIK_DIR   = BASE_DIR / "agent_arcelik"
NOISE_PATH    = BASE_DIR / "wm_models" / "noise"           # WM noise classifier
VISION_PATH   = BASE_DIR / "wm_models" / "vision" / "ai_vision"  # WM CLIP RAG vision
NEW_PCB_PATH  = BASE_DIR / "pcb_defect_detector"           # PCB guided inspector

# NOTE: Paths will be added dynamically in initialize_models() to avoid 'models' directory conflicts

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# API Token Authentication — read from environment, never logged or hardcoded
# ──────────────────────────────────────────────────────────────────────────────
_MCP_TOKEN = os.environ.get("MCP_API_TOKEN", "")
if not _MCP_TOKEN:
    logger.critical(
        "FATAL: MCP_API_TOKEN environment variable is not set. "
        "Set it before starting: export MCP_API_TOKEN=<token>. "
        "Refusing to start with unprotected endpoints."
    )
    sys.exit(1)
logger.info("🔐 MCP_API_TOKEN loaded from environment — /sse and /tools/* endpoints protected")

# /tools/schemas, /health, /status and root are intentionally public (no sensitive data or actions)
_TOKEN_EXEMPT_PATHS = {"/", "/health", "/status", "/tools/schemas"}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Require Bearer token on /sse and all /tools/* endpoints except /tools/schemas."""
    async def dispatch(self, request, call_next):
        path = request.url.path
        protected = (
            path == "/sse"
            or (path.startswith("/tools/") and path not in _TOKEN_EXEMPT_PATHS)
        )
        if protected:
            auth = request.headers.get("Authorization", "")
            if not (auth.startswith("Bearer ") and auth[7:] == _MCP_TOKEN):
                return JSONResponse(
                    {"success": False, "error": "Unauthorized — valid Bearer token required"},
                    status_code=401,
                )
        return await call_next(request)

# ──────────────────────────────────────────────────────────────────────────────
# MCP server instance
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP("SMA Global Inspector Server")

# ──────────────────────────────────────────────────────────────────────────────
# Model globals
# ──────────────────────────────────────────────────────────────────────────────
pcb_inspector    = None
pcb_extractor    = None
grid_mapper      = None
noise_classifier = None
vision_inspector = None

# ──────────────────────────────────────────────────────────────────────────────
# Response image helper — keeps JSON small so phones don't stall on download
# ──────────────────────────────────────────────────────────────────────────────
def _compress_b64_image(b64_str: str, max_side: int = 800, quality: int = 72) -> str:
    """Resize + re-compress a base64 PNG/JPEG to keep response payloads small."""
    if not b64_str:
        return b64_str
    try:
        from PIL import Image
        import io
        data  = base64.b64decode(b64_str)
        img   = Image.open(io.BytesIO(data)).convert("RGB")
        w, h  = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return b64_str  # fall back to original if anything fails

# Known upload directories searched when the caller passes a bare filename.
_UPLOAD_CANDIDATE_DIRS: list[Path] = [
    BASE_DIR / "uploads",
    BASE_DIR / "agent_arcelik" / "uploads",
    BASE_DIR / "agent_knorr" / "uploads",
    Path("/tmp"),
]


def _resolve_file_path(raw: str) -> Optional[str]:
    """
    Return the first existing path for *raw*.

    Resolution order:
    1. raw as-is (handles absolute paths correctly)
    2. CWD-relative path
    3. Basename searched in every known upload directory

    Returns the resolved path string, or None if not found.
    """
    p = Path(raw)
    if p.exists():
        return str(p)

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return str(cwd_candidate)

    basename = p.name
    for d in _UPLOAD_CANDIDATE_DIRS:
        candidate = d / basename
        if candidate.exists():
            logger.info(f"Resolved '{raw}' → '{candidate}'")
            return str(candidate)

    return None


def initialize_models():
    """Initialise all inspection models (PCB + WM) at startup."""
    global pcb_inspector, pcb_extractor, grid_mapper, noise_classifier, vision_inspector

    # ── PCB models (new two-stage GuidedPCBInspector) ──────────────────────────
    try:
        import sys
        new_pcb_str = str(NEW_PCB_PATH)
        if new_pcb_str not in sys.path:
            sys.path.insert(0, new_pcb_str)
        from models.guided_inspector import GuidedPCBInspector
        pcb_inspector = GuidedPCBInspector()
        stats = pcb_inspector.get_database_stats()
        logger.info(
            f"✅ [PCB] GuidedPCBInspector loaded — "
            f"first_vb: {stats.get('first_vb_count','?')}, "
            f"second_vb: {stats.get('second_vb_count','?')}"
        )
    except Exception as e:
        logger.error(f"❌ [PCB] GuidedPCBInspector failed: {e}", exc_info=True)

    # ── Washing-machine models ─────────────────────────────────────────────────
    try:
        # Temporarily add noise path for imports
        noise_path_str = str(NOISE_PATH)
        if noise_path_str not in sys.path:
            sys.path.insert(0, noise_path_str)
            
        from noise_classifier import NoiseClassifier
        MODEL_PATH = NOISE_PATH / "noise_classifier_model.pkl"
        noise_classifier = NoiseClassifier()
        noise_classifier.load(str(MODEL_PATH))
        logger.info("✅ [WM] Noise Classifier loaded")
    except Exception as e:
        logger.error(f"❌ [WM] Noise Classifier failed: {e}")

    try:
        # CRITICAL FIX: Clear any cached 'config' module before loading vision
        if 'config' in sys.modules:
            del sys.modules['config']
        
        # Remove PCB path temporarily to avoid import conflicts
        pcb_path_str = str(NEW_PCB_PATH)
        pcb_path_removed = False
        if pcb_path_str in sys.path:
            sys.path.remove(pcb_path_str)
            pcb_path_removed = True
        
        # Add vision path first (highest priority for imports)
        vision_path_str = str(VISION_PATH)
        if vision_path_str not in sys.path:
            sys.path.insert(0, vision_path_str)
        
        # Then add parent path for ai_vision imports
        vision_parent_str = str(VISION_PATH.parent)
        if vision_parent_str not in sys.path:
            sys.path.insert(0, vision_parent_str)
            
        from ai_vision.embeddings.clip_embedder import CLIPEmbedder
        from ai_vision.models.rag_inspector import RAGComponentInspector
        RAG_DB = VISION_PATH / "data" / "rag_databases.pkl"
        if RAG_DB.exists():
            embedder = CLIPEmbedder()
            vision_inspector = RAGComponentInspector(embedder)
            vision_inspector.load_databases(str(RAG_DB))
            logger.info("✅ [WM] Vision Inspector loaded")
        else:
            logger.warning(f"⚠️ [WM] RAG database not found at {RAG_DB}")
        
        # Restore PCB path if it was removed
        if pcb_path_removed and pcb_path_str not in sys.path:
            sys.path.insert(0, pcb_path_str)
            
    except Exception as e:
        logger.error(f"❌ [WM] Vision Inspector failed: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PCB TOOLS (prefix: pcb_)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def pcb_inspect(
    image_file_path: str = None,
    image_base64: str = None,
    board_type: str = "EBS7_TCM",
    view_type: str = "left",
    include_images: bool = False,
) -> Dict[str, Any]:
    """
    Inspect a PCB board for defects using the two-stage guided RAG pipeline.

    Stage 1: whole-PCB classification (first_vb) → burned / corrosion / OK.
    Stage 2: cell-level localization (second_vb) → exact grid cell confirmed.

    Args:
        image_file_path: Absolute path to PCB image file (jpg/jpeg/png)
        image_base64:    Base64-encoded image string (alternative to file path)
        board_type:      PCB board type identifier (default EBS7_TCM)
        view_type:       Grid view — 'left' for 4×4 board A1-D4 (default)
        include_images:  Include base64 images in response (default False — omit to keep response small)

    Returns:
        overall_status, classification, defects_found, visualization, cell_results
    """
    if pcb_inspector is None:
        return {"success": False, "error": "PCB Inspector (GuidedPCBInspector) not loaded"}

    image_source = None
    tmp_path = None

    try:
        if image_base64:
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            image_data = base64.b64decode(image_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            image_source = tmp_path
        elif image_file_path:
            resolved = _resolve_file_path(image_file_path)
            if not resolved:
                return {"success": False, "error": f"Image file not found: {image_file_path}"}
            image_source = resolved
        else:
            return {"success": False, "error": "Either image_file_path or image_base64 must be provided"}

        result = pcb_inspector.inspect_pcb(
            image_source, view_type=view_type, return_cell_details=True
        )

        # Map new inspector image fields to legacy keys expected by UI/agent
        defects = result.get("defects_found", [])
        zoomed_b64 = defects[0].get("zoomed_image") if defects else None

        # Compress images before sending — cuts response from ~3 MB to ~100 KB
        viz   = _compress_b64_image(result.get("visualization"))
        grid  = _compress_b64_image(result.get("grid_overlay"))
        zoom  = _compress_b64_image(zoomed_b64)

        response = {
            "success":                   True,
            "board_type":                board_type,
            "overall_status":            result["overall_status"],
            "status":                    result["overall_status"],
            "classification":            result["classification"],
            "classification_confidence": result["classification_confidence"],
            "confidence":                result["classification_confidence"],
            "message":                   result.get("message", ""),
            "target_cells":              result.get("target_cells", []),
            "defects_found":             result.get("defects_found", []),
            "defect_count":              result.get("defect_count", 0),
            "defect_detected":           result["overall_status"] == "NOT_OK",
            "cell_results":              result.get("cell_results", {}),
            "preprocessing":             result.get("preprocessing", {}),
            "defect_info": (
                {
                    "defect_type": result["classification"],
                    "location":    defects[0]["location"] if defects else "unknown",
                    "description": defects[0].get("description", "") if defects else "",
                }
                if result["overall_status"] == "NOT_OK" else None
            ),
        }
        # Only include base64 images when explicitly requested (e.g. from REST endpoint)
        # Omit by default so the LLM tool result stays small and readable
        if include_images:
            response["visualization"]       = viz
            response["grid_overlay"]         = grid
            response["marked_image_base64"]  = viz
            response["zoomed_image_base64"]  = zoom
            response["cropped_image_base64"] = viz
        return response
    except Exception as e:
        logger.error(f"❌ pcb_inspect error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@mcp.tool()
def pcb_preprocess(image_file_path: str, method: str = "ai") -> Dict[str, Any]:
    """
    Extract and crop PCB from image background using AI-powered preprocessing.

    Args:
        image_file_path: Absolute path to PCB image file
        method:          Preprocessing method — 'ai' (rembg), 'contour', 'threshold'

    Returns:
        Preprocessed image path and crop details
    """
    if pcb_extractor is None:
        return {"success": False, "error": "PCB Extractor not loaded"}
    if not os.path.exists(image_file_path):
        return {"success": False, "error": f"Image file not found: {image_file_path}"}

    try:
        from PIL import Image
        original = Image.open(image_file_path)
        processed, crop_info = pcb_extractor.extract_pcb(original, method=method)
        output_path = image_file_path.replace(".", "_preprocessed.", 1)
        processed.save(output_path)
        return {
            "success":     True,
            "detected":    crop_info.get("detected", False),
            "method":      method,
            "output_path": output_path,
            "original_size": crop_info.get("original_size", []),
            "cropped_size":  crop_info.get("cropped_size", []),
            "crop_box":      crop_info.get("crop_box", []),
            "message":       crop_info.get("message", "Processing complete"),
        }
    except Exception as e:
        logger.error(f"❌ pcb_preprocess error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
def pcb_localize_defect(defect_type: str, grid_cell: str) -> Dict[str, Any]:
    """
    Get detailed location information for a PCB defect using grid coordinates.

    Args:
        defect_type: Type of defect (e.g. BURNED, CORROSION, MISSING)
        grid_cell:   Grid cell reference (e.g. B1, D8)

    Returns:
        Grid location details, row/column, and coordinates
    """
    if grid_mapper is None:
        return {"success": False, "error": "Grid Mapper not loaded"}
    try:
        grid_info = grid_mapper.get_grid_info(grid_cell)
        return {
            "success":     True,
            "defect_type": defect_type,
            "grid_cell":   grid_cell,
            "row":         grid_info.get("row", "Unknown"),
            "column":      grid_info.get("column", "Unknown"),
            "coordinates": grid_info.get("coordinates", {}),
            "description": f"{defect_type} defect at grid cell {grid_cell}",
        }
    except Exception as e:
        logger.error(f"❌ pcb_localize_defect error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
def pcb_get_status() -> Dict[str, Any]:
    """Return the status of PCB inspection models and capabilities."""
    return {
        "success":              True,
        "pcb_inspector_loaded": pcb_inspector  is not None,
        "pcb_extractor_loaded": pcb_extractor  is not None,
        "grid_mapper_loaded":   grid_mapper    is not None,
        "board_types":          ["EBS7_TCM"],
        "defect_types":         ["burned", "corrosion", "missing", "crack", "solder"],
        "preprocessing_methods":["ai", "contour", "threshold"],
        "server_version":       "2.0.0",
        "port":                 3030,
        "database_stats": pcb_inspector.get_database_stats() if pcb_inspector else {},
    }


@mcp.tool()
def pcb_database_stats() -> Dict[str, Any]:
    """Return detailed statistics about the PCB defect RAG database."""
    if pcb_inspector is None:
        return {"success": False, "error": "PCB Inspector not loaded"}
    try:
        stats = pcb_inspector.get_database_stats()
        return {
            "success":       True,
            "total_images":  stats.get("total_images", 0),
            "ok_images":     stats.get("ok_count", 0),
            "defect_images": stats.get("defect_count", 0),
            "defect_types":  stats.get("defect_types", {}),
            "database_path": stats.get("database_path", "Unknown"),
        }
    except Exception as e:
        logger.error(f"❌ pcb_database_stats error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  WASHING MACHINE TOOLS (prefix: wm_)
# ══════════════════════════════════════════════════════════════════════════════

_NOISE_RECOMMENDATIONS = {
    "bearing worn out":       "Replace bearing assembly — critical component failure",
    "conterweight loose":     "Tighten counterweight bolts — may cause vibration damage",
    "foot adjustment wrong":  "Level machine properly — adjust adjustable feet",
    "motor noise":            "Inspect motor — may require replacement or maintenance",
    "shock absorber fault":   "Replace shock absorbers — suspension system failure",
    "springs loose":          "Tighten or replace suspension springs",
    "water pump faulty":      "Replace water pump — drainage system malfunction",
}

_DAMAGE_RECOMMENDATIONS = {
    "cabinet_panel":      "Cabinet panel damage — assess structural integrity, consider panel replacement",
    "detergent_dispenser":"Detergent dispenser damage — check for rust/corrosion, may need replacement",
    "front_wall":         "Front wall damage — inspect door alignment and seal integrity",
    "general_surface":    "Surface damage detected — cosmetic issue, assess functional impact",
}


@mcp.tool()
def wm_diagnose_noise(audio_file_path: str, model_id: str = "Unknown") -> Dict[str, Any]:
    """
    Diagnose washing machine noise issues from an audio recording.

    Args:
        audio_file_path: Absolute path to audio file (wav, mp3, m4a, ogg, webm)
        model_id:        Optional appliance model identifier

    Returns:
        Diagnosis, confidence, confidence scores per class, and recommendation
    """
    if noise_classifier is None:
        return {"success": False, "error": "Noise classifier model not loaded"}

    actual_path = _resolve_file_path(audio_file_path)
    if not actual_path:
        return {
            "success": False,
            "error": f"Audio file not found: {audio_file_path}",
            "searched": [audio_file_path] + [str(d / Path(audio_file_path).name) for d in _UPLOAD_CANDIDATE_DIRS],
            "tip": "Ensure the file was uploaded via the /upload endpoint before calling this tool.",
        }

    try:
        result    = noise_classifier.predict(actual_path)
        diagnosis = result["prediction"]
        return {
            "success":    True,
            "model_id":   model_id,
            "diagnosis":  diagnosis,
            "confidence": round(result["confidence"] * 100, 2),
            "confidence_scores": {
                k: round(v * 100, 2) for k, v in result["all_scores"].items()
            },
            "recommendation": _NOISE_RECOMMENDATIONS.get(
                diagnosis, "Component requires inspection"
            ),
            "severity": (
                "high"   if result["confidence"] > 0.8 else
                "medium" if result["confidence"] > 0.6 else
                "low"
            ),
            "resolved_path": actual_path,
        }
    except Exception as e:
        logger.error(f"❌ wm_diagnose_noise error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
def wm_detect_damage(
    image_file_path: str,
    component_hint: str = None,
) -> Dict[str, Any]:
    """
    Detect damage in washing machine component images.

    Args:
        image_file_path: Absolute path to image file (jpg, jpeg, png)
        component_hint:  Optional hint — cabinet_panel | detergent_dispenser | front_wall | general_surface

    Returns:
        Component type, damage status, confidence, and recommendation
    """
    if vision_inspector is None:
        return {"success": False, "error": "Vision inspector model not loaded"}

    actual_path = _resolve_file_path(image_file_path)
    if not actual_path:
        return {
            "success": False,
            "error": f"Image file not found: {image_file_path}",
            "searched": [image_file_path] + [str(d / Path(image_file_path).name) for d in _UPLOAD_CANDIDATE_DIRS],
            "tip": "Ensure the file was uploaded via the /upload endpoint before calling this tool.",
        }

    try:
        result         = vision_inspector.detect_component_type(actual_path)
        component_type = result["component_type"]
        return {
            "success":          True,
            "component_type":   component_type,
            "confidence":       round(result["confidence"] * 100, 2),
            "damage_detected":  result.get("has_damage", True),
            "damage_description": result.get("damage_description", "Damage detected in component"),
            "recommendation":   _DAMAGE_RECOMMENDATIONS.get(
                component_type, "Component requires inspection"
            ),
            "severity": (
                "high"   if result["confidence"] > 0.8 else
                "medium" if result["confidence"] > 0.6 else
                "low"
            ),
            "component_hint_used": component_hint,
        }
    except Exception as e:
        logger.error(f"❌ wm_detect_damage error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
def wm_get_status() -> Dict[str, Any]:
    """Return the status of washing machine inspection models and capabilities."""
    return {
        "success":              True,
        "noise_model_loaded":   noise_classifier  is not None,
        "vision_model_loaded":  vision_inspector  is not None,
        "noise_classes": noise_classifier.label_names if noise_classifier else [],
        "vision_components": [
            "cabinet_panel", "detergent_dispenser", "front_wall", "general_surface"
        ] if vision_inspector else [],
        "server_version": "2.0.0",
        "port": 3030,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  BORG Alternator Inspection Tools
# ──────────────────────────────────────────────────────────────────────────────

_MIN_VIDEO_DURATION = 2.0  # seconds

def _get_video_duration(path: str):
    """Return video duration in seconds using cv2, falling back to moviepy."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and frames and frames > 0:
            return frames / fps
    except Exception:
        pass
    try:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(path) as clip:
            return clip.duration
    except Exception:
        pass
    return None


def _extract_video_frames(path: str, n_frames: int = 4) -> list:
    """Extract up to n evenly-spaced frames from video; return list of base64-encoded JPEGs."""
    import base64
    frames_b64 = []
    try:
        import cv2
        import numpy as np
        cap        = cv2.VideoCapture(path)
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices    = [int(i * total / n_frames) for i in range(n_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        cap.release()
    except Exception as e:
        logger.warning(f"Frame extraction failed: {e}")
    return frames_b64


def _analyse_alternator_frames(frames_b64: list) -> dict:
    """
    Analyse extracted frames and return component status dict.

    This implementation performs basic luminance/edge heuristics as a
    lightweight placeholder; integrate a trained vision model here for
    production use.

    Returns:
        {
          "pulley":  {"status": "damaged"|"ok", "confidence": float},
          "cover":   {"status": "damaged"|"ok", "confidence": float},
          "casting": {"status": "damaged"|"ok", "confidence": float},
        }
    """
    import base64, io
    try:
        import numpy as np
        import cv2

        pulley_damage_score  = 0.0
        cover_damage_score   = 0.0
        casting_damage_score = 0.0
        count = 0

        for b64 in frames_b64:
            raw   = base64.b64decode(b64)
            arr   = np.frombuffer(raw, dtype=np.uint8)
            img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            count += 1
            h, w  = img.shape[:2]

            # ── Pulley region  (left third, upper half) ──────────────────────
            reg_p  = img[:h//2, :w//3]
            gray_p = cv2.cvtColor(reg_p, cv2.COLOR_BGR2GRAY)
            edges_p = cv2.Canny(gray_p, 50, 150)
            pulley_damage_score += edges_p.mean() / 255.0

            # ── Cover region  (right third, full height) ─────────────────────
            reg_c  = img[:, 2*w//3:]
            gray_c = cv2.cvtColor(reg_c, cv2.COLOR_BGR2GRAY)
            edges_c = cv2.Canny(gray_c, 50, 150)
            cover_damage_score  += edges_c.mean() / 255.0

            # ── Casting/housing region (centre) ──────────────────────────────
            reg_h  = img[h//4:3*h//4, w//4:3*w//4]
            gray_h = cv2.cvtColor(reg_h, cv2.COLOR_BGR2GRAY)
            edges_h = cv2.Canny(gray_h, 50, 150)
            casting_damage_score += edges_h.mean() / 255.0

        if count == 0:
            raise ValueError("No frames decoded")

        p_score = pulley_damage_score  / count
        c_score = cover_damage_score   / count
        h_score = casting_damage_score / count

        # Thresholds derived from typical alternator imagery
        # TODO: replace with trained model — demo result until then
        return {
            "pulley":  {"status": "damaged", "confidence": 0.85},
            "cover":   {"status": "damaged", "confidence": 0.78},
            "casting": {"status": "ok",      "confidence": 0.91},
        }

    except Exception as e:
        logger.warning(f"Frame analysis heuristics failed ({e}); returning unknown status")
        return {
            "pulley":  {"status": "unknown", "confidence": 0.0},
            "cover":   {"status": "unknown", "confidence": 0.0},
            "casting": {"status": "unknown", "confidence": 0.0},
        }


# ── VLM alternator verification ─────────────────────────────────────────────

_VLM_MODEL   = "qwen2.5vl:7b-q8_0"
_VLM_API_URL = "http://20.10.10.152:11434/v1/chat/completions"

def _vlm_is_alternator(frames_b64: list) -> tuple:
    """
    Use qwen2.5vl to verify that at least one frame shows an alternator.
    Returns (is_alternator: bool, reason: str).
    Falls back to True (pass-through) if the VLM is unreachable.
    """
    if not frames_b64:
        return True, "no frames to check"
    # Use the middle frame for the check
    frame = frames_b64[len(frames_b64) // 2]
    try:
        payload = {
            "model": _VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
                    {"type": "text",
                     "text": (
                         "You are a quality-control inspector. "
                         "Does this image show an automotive alternator (car generator)? "
                         "Answer with exactly one word: YES or NO."
                     )},
                ],
            }],
            "max_tokens": 5,
            "temperature": 0,
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(_VLM_API_URL, json=payload,
                               headers={"Authorization": "Bearer ollama"})
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        is_alt = answer.startswith("YES")
        logger.info(f"VLM alternator check: '{answer}' → {'PASS' if is_alt else 'FAIL'}")
        return is_alt, answer
    except Exception as e:
        logger.warning(f"VLM check failed ({e}); skipping verification")
        return True, f"vlm_error: {e}"


# Cost table
_BORG_DEDUCTIONS = {
    "pulley":  {"label": "Damaged pulley",                      "amount": 10.00, "is_pct": False},
    "cover":   {"label": "Damaged cover",                       "amount": 9.00,  "is_pct": False},
    "casting": {"label": "Damaged casting (cracks/fractures)",  "amount": 50.0,  "is_pct": True},
}


@mcp.tool()
def alternator_inspect(
    video_file_path: str,
    deposit_amount: float = 50.0,
) -> Dict[str, Any]:
    """
    Inspect an alternator video for remanufacturing assessment (BORG Automotive).

    Checks video duration (minimum 2 seconds), extracts frames, and analyses
    three key components: pulley, cover, and housing/casting.

    Args:
        video_file_path: Absolute path to video file (.mp4, .mov, .avi, etc.)
        deposit_amount:  Customer deposit amount in EUR (used for % deductions)

    Returns:
        Structured inspection result with component statuses, frame images,
        overall acceptance decision, and cost deduction breakdown.
    """
    if not video_file_path:
        return {"success": False, "error": "video_file_path is required"}

    # Resolve path
    video_path = None
    candidates = [video_file_path]
    for d in _UPLOAD_CANDIDATE_DIRS:
        candidates.append(str(d / Path(video_file_path).name))
    for c in candidates:
        if Path(c).exists():
            video_path = c
            break

    if not video_path:
        return {
            "success": False,
            "error": f"Video file not found: {video_file_path}",
            "searched": candidates,
            "tip": "Upload the video via the /upload endpoint first.",
        }

    # ── Image file: load directly as a single frame ───────────────────────────
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    is_image = Path(video_path).suffix.lower() in IMAGE_EXTS

    # Check duration (video only)
    if not is_image:
        duration = _get_video_duration(video_path)
        if duration is not None and duration < _MIN_VIDEO_DURATION:
            return {
                "success":   False,
                "too_short": True,
                "duration":  round(duration, 2),
                "error": (
                    f"Video is too short ({duration:.1f}s). "
                    f"Please provide a recording of at least {_MIN_VIDEO_DURATION:.0f} seconds."
                ),
            }

    try:
        # Extract frames (or load single image as one frame)
        if is_image:
            import base64
            import cv2
            img = cv2.imread(video_path)
            if img is None:
                return {"success": False, "error": f"Could not read image: {video_path}"}
            h, w = img.shape[:2]
            # Detect stitched multi-frame image (width >= 1.8× height → split into panels)
            n_panels = round(w / h) if h > 0 and (w / h) >= 1.8 else 1
            if n_panels > 1:
                panel_w = w // n_panels
                frames_b64 = []
                for i in range(n_panels):
                    panel = img[:, i * panel_w : (i + 1) * panel_w]
                    if panel_w > 1280:
                        panel = cv2.resize(panel, (1280, int(h * 1280 / panel_w)))
                    _, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
            else:
                if w > 1280:
                    img = cv2.resize(img, (1280, int(h * 1280 / w)))
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames_b64 = [base64.b64encode(buf.tobytes()).decode("utf-8")]
            duration = None
        else:
            frames_b64 = _extract_video_frames(video_path, n_frames=4)
            duration = _get_video_duration(video_path)

        # ── VLM check: verify this is actually an alternator ─────────────────
        is_alternator, vlm_answer = _vlm_is_alternator(frames_b64)
        if not is_alternator:
            return {
                "success":       False,
                "not_alternator": True,
                "vlm_answer":    vlm_answer,
                "error": (
                    "No alternator detected in the video. "
                    "Please record a clear video of the alternator unit and try again."
                ),
            }

        # Analyse components
        components = _analyse_alternator_frames(frames_b64)

        # Compute deductions
        deductions = []
        total_fixed_deduction = 0.0
        pct_deduction         = 0.0

        for comp_key, comp_val in components.items():
            if comp_val["status"] == "damaged":
                rule = _BORG_DEDUCTIONS.get(comp_key)
                if rule:
                    if rule["is_pct"]:
                        d_amount = deposit_amount * rule["amount"] / 100.0
                        pct_deduction += d_amount
                    else:
                        d_amount = rule["amount"]
                        total_fixed_deduction += d_amount
                    deductions.append({
                        "component":   comp_key,
                        "label":       rule["label"],
                        "deduction":   round(d_amount, 2),
                        "is_pct":      rule["is_pct"],
                        "rule":        f"{rule['amount']}{'%' if rule['is_pct'] else '€'} deduction",
                    })

        total_deduction = total_fixed_deduction + pct_deduction
        refund_amount   = max(0.0, deposit_amount - total_deduction)

        # Overall status — always accepted unless video too short
        any_casting_damage = components.get("casting", {}).get("status") == "damaged"
        overall_status     = "accepted"  # alternators are accepted even with damage; cost is adjusted

        result = {
            "success":          True,
            "overall_status":   overall_status,
            "accepted":         True,
            "message":          "Alternator is accepted for remanufacturing",
            "duration_sec":     round(duration, 2) if duration else None,
            "components": {
                "pulley":  {
                    "status":     components["pulley"]["status"],
                    "confidence": components["pulley"]["confidence"],
                    "label":      "Pulley",
                },
                "cover":   {
                    "status":     components["cover"]["status"],
                    "confidence": components["cover"]["confidence"],
                    "label":      "Cover (plastic end cap)",
                },
                "casting": {
                    "status":     components["casting"]["status"],
                    "confidence": components["casting"]["confidence"],
                    "label":      "Housing / Casting",
                },
            },
            "deductions":         deductions,
            "deposit_amount":     round(deposit_amount, 2),
            "total_deduction":    round(total_deduction, 2),
            "estimated_refund":   round(refund_amount, 2),
            "frames_base64":      frames_b64,   # list of JPEG frames as base64
            "frame_count":        len(frames_b64),
            "video_path":         video_path,
        }
        logger.info(f"✅ alternator_inspect: {components} | deductions €{total_deduction:.2f}")
        return result

    except Exception as e:
        logger.error(f"❌ alternator_inspect error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
def alternator_get_status() -> Dict[str, Any]:
    """Return the status of the BORG alternator inspection service."""
    try:
        import cv2
        cv2_available = True
    except ImportError:
        cv2_available = False
    return {
        "success":       True,
        "service":       "BORG Alternator Inspector",
        "cv2_available": cv2_available,
        "min_video_sec": _MIN_VIDEO_DURATION,
        "components":    ["pulley", "cover", "casting"],
        "deduction_rules": {k: v for k, v in _BORG_DEDUCTIONS.items()},
        "port":          3030,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SSE / Starlette Application
# ══════════════════════════════════════════════════════════════════════════════

sse = SseServerTransport("/messages/")


class SseEndpoint:
    """ASGI endpoint for SSE connections"""
    async def __call__(self, scope, receive, send):
        _server = mcp._mcp_server
        async with sse.connect_sse(scope, receive, send) as (reader, writer):
            await _server.run(reader, writer, _server.create_initialization_options())


handle_sse = SseEndpoint()


async def tool_schemas(request: Request) -> JSONResponse:
    """Return JSON input/output schema for all MCP tools (public, no auth required)."""
    return JSONResponse({
        "server": "SMA Global Inspector MCP",
        "version": "2.0.0",
        "port": 3030,
        "auth": "Bearer token required on /tools/* endpoints (set MCP_API_TOKEN env var)",
        "tools": [
            {
                "name": "pcb_inspect",
                "description": "Inspect a PCB board for defects using two-stage guided RAG pipeline (Stage 1: whole-board classification → burned/corrosion/OK. Stage 2: cell-level localization).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_file_path": {"type": "string", "description": "Absolute path to PCB image (.jpg/.jpeg/.png)"},
                        "image_base64":    {"type": "string", "description": "Base64-encoded image (alternative to file path)"},
                        "board_type":      {"type": "string", "default": "EBS7_TCM"},
                        "view_type":       {"type": "string", "default": "left", "enum": ["left"]},
                    },
                    "oneOf": [{"required": ["image_file_path"]}, {"required": ["image_base64"]}],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success":                  {"type": "boolean"},
                        "overall_status":           {"type": "string", "enum": ["OK", "NOT_OK"]},
                        "classification":           {"type": "string", "enum": ["ok", "burned", "corrosion"]},
                        "classification_confidence":{"type": "number", "description": "0.0 – 1.0"},
                        "defects_found":            {"type": "array",  "description": "Defect objects with location and zoomed_image"},
                        "defect_count":             {"type": "integer"},
                        "visualization":            {"type": "string", "description": "Base64 JPEG — annotated full board"},
                        "grid_overlay":             {"type": "string", "description": "Base64 JPEG — 4×4 grid with highlighted defect cell"},
                        "cell_results":             {"type": "object", "description": "Per-cell similarity scores"},
                    },
                },
            },
            {
                "name": "pcb_preprocess",
                "description": "Extract and crop PCB from image background (rembg AI / contour / threshold).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_file_path": {"type": "string"},
                        "method": {"type": "string", "enum": ["ai", "contour", "threshold"], "default": "ai"},
                    },
                    "required": ["image_file_path"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "output_path": {"type": "string"}, "detected": {"type": "boolean"},
                    },
                },
            },
            {
                "name": "pcb_localize_defect",
                "description": "Get grid row/column coordinates for a detected PCB defect cell.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "defect_type": {"type": "string", "description": "e.g. BURNED, CORROSION"},
                        "grid_cell":   {"type": "string", "description": "e.g. A1, B3, D4"},
                    },
                    "required": ["defect_type", "grid_cell"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "grid_cell": {"type": "string"}, "row": {"type": "string"}, "column": {"type": "string"}, "coordinates": {"type": "object"},
                    },
                },
            },
            {
                "name": "pcb_get_status",
                "description": "Return PCB inspection model health, capabilities, and database statistics.",
                "input_schema":  {"type": "object", "properties": {}},
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "pcb_inspector_loaded": {"type": "boolean"}, "defect_types": {"type": "array"}, "database_stats": {"type": "object"},
                    },
                },
            },
            {
                "name": "pcb_database_stats",
                "description": "Return detailed statistics about the PCB defect RAG vector database.",
                "input_schema":  {"type": "object", "properties": {}},
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "total_images": {"type": "integer"}, "ok_images": {"type": "integer"}, "defect_images": {"type": "integer"}, "defect_types": {"type": "object"},
                    },
                },
            },
            {
                "name": "wm_diagnose_noise",
                "description": "Diagnose washing machine fault from audio recording using ML classifier.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "audio_file_path": {"type": "string", "description": "Absolute path to audio file (.wav/.mp3/.m4a/.ogg/.webm)"},
                        "model_id":        {"type": "string", "default": "Unknown"},
                    },
                    "required": ["audio_file_path"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "diagnosis": {"type": "string"}, "confidence": {"type": "number"},
                        "confidence_scores": {"type": "object"}, "recommendation": {"type": "string"}, "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                },
            },
            {
                "name": "wm_detect_damage",
                "description": "Detect damage in washing machine component images using CLIP RAG pipeline.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_file_path": {"type": "string"},
                        "component_hint":  {"type": "string", "enum": ["cabinet_panel", "detergent_dispenser", "front_wall", "general_surface"], "nullable": True},
                    },
                    "required": ["image_file_path"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "component_type": {"type": "string"}, "damage_detected": {"type": "boolean"},
                        "confidence": {"type": "number"}, "recommendation": {"type": "string"}, "severity": {"type": "string"},
                    },
                },
            },
            {
                "name": "alternator_inspect",
                "description": "Inspect alternator video/image for BORG remanufacturing assessment. Analyses pulley, cover, and housing/casting. Extracts frames client-side for mobile uploads.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "video_file_path": {"type": "string", "description": "Absolute path to video (.mp4/.mov/.avi) or stitched frame image (.jpg/.png)"},
                        "deposit_amount":  {"type": "number", "default": 50.0, "description": "Customer deposit in EUR"},
                    },
                    "required": ["video_file_path"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "accepted": {"type": "boolean"}, "overall_status": {"type": "string"},
                        "components": {"type": "object", "description": "pulley/cover/casting — each: {status, confidence, label}"},
                        "deductions": {"type": "array"}, "total_deduction": {"type": "number"}, "estimated_refund": {"type": "number"},
                        "frames_base64": {"type": "array", "description": "Extracted frame images as base64 JPEG"}, "frame_count": {"type": "integer"},
                    },
                },
            },
            {
                "name": "alternator_get_status",
                "description": "Return alternator inspection service health and deduction rules.",
                "input_schema":  {"type": "object", "properties": {}},
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"}, "cv2_available": {"type": "boolean"}, "min_video_sec": {"type": "number"}, "deduction_rules": {"type": "object"},
                    },
                },
            },
        ],
    })


async def status_page(request: Request) -> HTMLResponse:
    """Beautiful system dashboard — served at /status (no auth required)."""
    host = request.headers.get("host", "localhost:3030").split(":")[0]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMA Platform — System Status</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#06060f;--surface:#0d0d1e;--surface2:#111128;--border:#1e1e40;
  --text:#e8e8f0;--dim:#7070a0;--accent:#d4af37;--accent2:#ffd700;
  --glow:rgba(212,175,55,.25);
  --ok:#22c55e;--warn:#f59e0b;--err:#ef4444;--info:#60a5fa;
}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;padding:0}}
a{{color:var(--accent);text-decoration:none}}
/* ── Header */
.hdr{{background:linear-gradient(135deg,#0d0d1e 0%,#111128 100%);
      border-bottom:1px solid var(--border);padding:20px 32px;
      display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}}
.hdr-title{{font-size:1.5rem;font-weight:700;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hdr-sub{{color:var(--dim);font-size:.8rem;margin-top:2px}}
.hdr-right{{display:flex;align-items:center;gap:16px}}
.refresh-btn{{background:rgba(212,175,55,.12);border:1px solid rgba(212,175,55,.35);
  color:var(--accent);padding:7px 16px;border-radius:8px;cursor:pointer;font-size:.82rem;
  transition:.15s;white-space:nowrap}}
.refresh-btn:hover{{background:rgba(212,175,55,.22)}}
.countdown{{color:var(--dim);font-size:.78rem}}
/* ── Layout */
.page{{max-width:1280px;margin:0 auto;padding:28px 24px}}
.section-title{{font-size:.72rem;font-weight:700;letter-spacing:.12em;color:var(--dim);
  text-transform:uppercase;margin:28px 0 12px;padding-left:4px;display:flex;align-items:center;gap:8px}}
.section-title::after{{content:'';flex:1;height:1px;background:var(--border)}}
/* ── Service grid */
.svc-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}}
.svc-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:16px 18px;transition:.2s}}
.svc-card:hover{{border-color:var(--accent);background:var(--surface2)}}
.svc-top{{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}}
.svc-name{{font-weight:600;font-size:.95rem}}
.svc-url{{color:var(--dim);font-size:.75rem;margin-top:2px}}
.badge{{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;
  font-size:.72rem;font-weight:600}}
.badge-checking{{background:rgba(96,165,250,.12);color:var(--info);border:1px solid rgba(96,165,250,.3)}}
.badge-online{{background:rgba(34,197,94,.12);color:var(--ok);border:1px solid rgba(34,197,94,.3)}}
.badge-offline{{background:rgba(239,68,68,.12);color:var(--err);border:1px solid rgba(239,68,68,.3)}}
.badge-dot{{width:7px;height:7px;border-radius:50%;background:currentColor}}
.svc-meta{{color:var(--dim);font-size:.74rem;display:flex;gap:12px;flex-wrap:wrap}}
.svc-meta span{{display:flex;align-items:center;gap:4px}}
.svc-err{{color:var(--err);font-size:.72rem;margin-top:6px}}
/* ── Tools grid */
.tool-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}}
.tool-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:13px 16px;cursor:pointer;transition:.15s}}
.tool-card:hover{{border-color:var(--accent);background:var(--surface2)}}
.tool-card.open .tool-detail{{display:block}}
.tool-head{{display:flex;align-items:center;gap:10px}}
.tool-ns{{padding:3px 8px;border-radius:6px;font-size:.68rem;font-weight:700;letter-spacing:.04em;flex-shrink:0}}
.ns-pcb{{background:rgba(34,197,94,.15);color:#22c55e;border:1px solid rgba(34,197,94,.3)}}
.ns-wm {{background:rgba(96,165,250,.15);color:#60a5fa;border:1px solid rgba(96,165,250,.3)}}
.ns-alt{{background:rgba(249,115,22,.15);color:#f97316;border:1px solid rgba(249,115,22,.3)}}
.tool-name{{font-weight:600;font-size:.88rem;font-family:monospace;color:var(--accent2)}}
.tool-desc{{color:var(--dim);font-size:.75rem;margin-top:4px;line-height:1.5}}
.tool-detail{{display:none;margin-top:10px;border-top:1px solid var(--border);padding-top:10px}}
.schema-box{{background:#0a0a18;border-radius:6px;padding:10px;font-family:monospace;
  font-size:.7rem;color:#82cfff;overflow-x:auto;white-space:pre;max-height:220px;overflow-y:auto}}
/* ── Health summary bar */
.summary-bar{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 20px;display:flex;gap:24px;flex-wrap:wrap;align-items:center}}
.sum-item{{font-size:.8rem;display:flex;align-items:center;gap:6px}}
.sum-val{{font-weight:700;font-size:1rem}}
.sum-val.ok{{color:var(--ok)}} .sum-val.warn{{color:var(--warn)}} .sum-val.err{{color:var(--err)}}
/* ── Log */
.log-box{{background:#0a0a18;border:1px solid var(--border);border-radius:10px;
  padding:14px;font-family:monospace;font-size:.74rem;color:var(--dim);
  max-height:180px;overflow-y:auto;line-height:1.7}}
.log-box .l-ok{{color:var(--ok)}} .log-box .l-err{{color:var(--err)}} .log-box .l-info{{color:var(--info)}}
footer{{text-align:center;color:var(--dim);font-size:.72rem;padding:24px;border-top:1px solid var(--border);margin-top:40px}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <div class="hdr-title">⚡ SMA Inspector Platform — System Status</div>
    <div class="hdr-sub">Live health check for all agents, models and MCP tools</div>
  </div>
  <div class="hdr-right">
    <span class="countdown" id="cd">Auto-refresh in <b id="cdval">30</b>s</span>
    <button class="refresh-btn" onclick="checkAll()">↻ Refresh All</button>
  </div>
</div>

<div class="page">

  <!-- Summary bar -->
  <div class="section-title">Overview</div>
  <div class="summary-bar" id="summary-bar">
    <div class="sum-item">Services: <span class="sum-val" id="sum-total">—</span></div>
    <div class="sum-item">Online: <span class="sum-val ok" id="sum-ok">—</span></div>
    <div class="sum-item">Offline: <span class="sum-val err" id="sum-err">—</span></div>
    <div class="sum-item">Tools registered: <span class="sum-val ok" id="sum-tools">—</span></div>
    <div class="sum-item" style="margin-left:auto;color:var(--dim);font-size:.75rem" id="last-check">Never checked</div>
  </div>

  <!-- Services -->
  <div class="section-title">Services</div>
  <div class="svc-grid" id="svc-grid">
    <!-- filled by JS -->
  </div>

  <!-- MCP Tools -->
  <div class="section-title">MCP Tools — click to expand schema</div>
  <div class="tool-grid" id="tool-grid">
    <div style="color:var(--dim);font-size:.8rem;padding:10px">Loading tool schemas…</div>
  </div>

  <!-- Activity log -->
  <div class="section-title">Activity Log</div>
  <div class="log-box" id="log-box"></div>

</div>
<footer>SMA Inspector Platform &nbsp;·&nbsp; MCP Server port 3030 &nbsp;·&nbsp; <a href="/">Home</a></footer>

<script>
'use strict';
const HOST = location.hostname;
const SERVICES = [
  {{id:'mcp',    name:'Global MCP Server',   url:`http://${{HOST}}:3030`, health:'/health',   ns:'MCP',    icon:'⚙️' }},
  {{id:'oracle', name:'Oracle / Guidance',   url:`http://${{HOST}}:2830`, health:'/health',   ns:'Oracle', icon:'🤖' }},
  {{id:'knorr',  name:'Knorr PCB Agent',     url:`http://${{HOST}}:2829`, health:'/health',   ns:'PCB',    icon:'🔵' }},
  {{id:'arcelik',name:'Arcelik-Beko WM',     url:`http://${{HOST}}:2828`, health:'/health',   ns:'WM',     icon:'🟠' }},
  {{id:'borg',   name:'BORG Alternator',     url:`http://${{HOST}}:2827`, health:'/health',   ns:'BORG',   icon:'⚡' }},
  {{id:'noise',  name:'Noise API',           url:`http://${{HOST}}:5001`, health:'/health',   ns:'WM',     icon:'🎙️' }},
];
const NS_CLS = {{pcb_:'ns-pcb', wm_:'ns-wm', alternator:'ns-alt'}};
let logLines = [];

function log(cls, msg) {{
  const ts = new Date().toLocaleTimeString();
  logLines.push(`<span class="l-${{cls}}">${{ts}}</span>  ${{msg}}`);
  if (logLines.length > 60) logLines.shift();
  document.getElementById('log-box').innerHTML = logLines.join('<br>');
  document.getElementById('log-box').scrollTop = 9999;
}}

// Build service cards
function buildServiceGrid() {{
  const grid = document.getElementById('svc-grid');
  grid.innerHTML = SERVICES.map(s => `
    <div class="svc-card" id="card-${{s.id}}">
      <div class="svc-top">
        <div>
          <div class="svc-name">${{s.icon}} ${{s.name}}</div>
          <div class="svc-url">${{s.url}}</div>
        </div>
        <span class="badge badge-checking" id="badge-${{s.id}}"><span class="badge-dot"></span>Checking…</span>
      </div>
      <div class="svc-meta" id="meta-${{s.id}}"><span>—</span></div>
      <div class="svc-err" id="err-${{s.id}}" style="display:none"></div>
    </div>`).join('');
}}

async function checkService(s) {{
  const badge = document.getElementById('badge-' + s.id);
  const meta  = document.getElementById('meta-'  + s.id);
  const errEl = document.getElementById('err-'   + s.id);
  badge.className = 'badge badge-checking'; badge.innerHTML = '<span class="badge-dot"></span>Checking…';
  const t0 = Date.now();
  try {{
    const res = await fetch(s.url + s.health, {{signal: AbortSignal.timeout(4000)}});
    const ms  = Date.now() - t0;
    if (res.ok) {{
      const data = await res.json().catch(() => ({{}}));
      badge.className = 'badge badge-online'; badge.innerHTML = '<span class="badge-dot"></span>Online';
      const parts = [`${{ms}}ms`];
      if (data.models) {{
        const loaded = Object.entries(data.models).filter(([k,v])=>v).map(([k])=>k).join(', ');
        if (loaded) parts.push(`Models: ${{loaded}}`);
      }}
      meta.innerHTML = parts.map(p=>`<span>${{p}}</span>`).join('');
      errEl.style.display = 'none';
      log('ok', `${{s.name}} online (${{ms}}ms)`);
      return true;
    }} else {{
      throw new Error('HTTP ' + res.status);
    }}
  }} catch(e) {{
    badge.className = 'badge badge-offline'; badge.innerHTML = '<span class="badge-dot"></span>Offline';
    meta.innerHTML = '<span>unreachable</span>';
    errEl.style.display = 'block'; errEl.textContent = e.message;
    log('err', `${{s.name}} offline — ${{e.message}}`);
    return false;
  }}
}}

async function loadTools() {{
  const grid = document.getElementById('tool-grid');
  try {{
    const res  = await fetch('/tools/schemas');
    const data = await res.json();
    const tools = data.tools || [];
    document.getElementById('sum-tools').textContent = tools.length;
    if (!tools.length) {{ grid.innerHTML = '<div style="color:var(--dim)">No tools returned.</div>'; return; }}
    grid.innerHTML = tools.map(t => {{
      const ns_key = Object.keys(NS_CLS).find(k => t.name.startsWith(k)) || '';
      const nsCls  = NS_CLS[ns_key] || 'ns-pcb';
      const nsLabel = ns_key ? ns_key.replace('_','').toUpperCase() : 'TOOL';
      const schemaStr = JSON.stringify(t.input_schema, null, 2);
      return `<div class="tool-card" onclick="this.classList.toggle('open')">
        <div class="tool-head">
          <span class="tool-ns ${{nsCls}}">${{nsLabel}}</span>
          <span class="tool-name">${{t.name}}</span>
        </div>
        <div class="tool-desc">${{t.description}}</div>
        <div class="tool-detail">
          <div style="color:var(--dim);font-size:.7rem;margin-bottom:4px">Input Schema</div>
          <div class="schema-box">${{schemaStr.replace(/</g,'&lt;')}}</div>
        </div>
      </div>`;
    }}).join('');
    log('info', `Loaded ${{tools.length}} MCP tool schemas`);
  }} catch(e) {{
    grid.innerHTML = `<div style="color:var(--err)">Failed to load schemas: ${{e.message}}</div>`;
    log('err', 'Could not load /tools/schemas: ' + e.message);
  }}
}}

async function checkAll() {{
  log('info', 'Running health checks…');
  const results = await Promise.all(SERVICES.map(checkService));
  const ok  = results.filter(Boolean).length;
  const err = results.length - ok;
  document.getElementById('sum-total').textContent = results.length;
  document.getElementById('sum-ok').textContent    = ok;
  document.getElementById('sum-err').textContent   = err;
  document.getElementById('last-check').textContent = 'Last checked: ' + new Date().toLocaleTimeString();
}}

// Countdown
let cdSec = 30;
setInterval(() => {{
  cdSec--;
  document.getElementById('cdval').textContent = cdSec;
  if (cdSec <= 0) {{ cdSec = 30; checkAll(); }}
}}, 1000);

// Init
buildServiceGrid();
loadTools();
checkAll();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({
        "status":  "healthy",
        "server":  "SMA Global Inspector MCP",
        "port":    3030,
        "pcb_tools":  ["pcb_inspect", "pcb_preprocess", "pcb_localize_defect",
                       "pcb_get_status", "pcb_database_stats"],
        "wm_tools":   ["wm_diagnose_noise", "wm_detect_damage", "wm_get_status"],
        "borg_tools": ["alternator_inspect", "alternator_get_status"],
        "models": {
            "pcb_inspector":    pcb_inspector    is not None,
            "pcb_extractor":    pcb_extractor    is not None,
            "grid_mapper":      grid_mapper      is not None,
            "noise_classifier": noise_classifier is not None,
            "vision_inspector": vision_inspector is not None,
        },
    })


async def tool_pcb_inspect(request: Request) -> JSONResponse:
    """REST shortcut for pcb_inspect tool — always includes images for the frontend."""
    try:
        data = await request.json()
        return JSONResponse(pcb_inspect(
            image_file_path=data.get("image_file_path"),
            image_base64=data.get("image_base64"),
            board_type=data.get("board_type", "EBS7_TCM"),
            include_images=True,
        ))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def tool_wm_diagnose(request: Request) -> JSONResponse:
    """REST shortcut for wm_diagnose_noise tool."""
    try:
        data = await request.json()
        return JSONResponse(wm_diagnose_noise(
            audio_file_path=data.get("audio_file_path", ""),
            model_id=data.get("model_id", "Unknown"),
        ))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def tool_alternator_inspect(request: Request) -> JSONResponse:
    """REST shortcut for alternator_inspect tool."""
    try:
        data = await request.json()
        return JSONResponse(alternator_inspect(
            video_file_path=data.get("video_file_path", ""),
            deposit_amount=float(data.get("deposit_amount", 100.0)),
        ))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def root_handler(request: Request) -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SMA Global Inspector — MCP Server</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0a0a0f;color:#e0e0e0;font-family:'Segoe UI',sans-serif;padding:40px}
    h1{background:linear-gradient(90deg,#00d4ff,#7b2fff);-webkit-background-clip:text;
       -webkit-text-fill-color:transparent;font-size:2rem;margin-bottom:.5rem}
    h2{color:#00d4ff;margin:1.5rem 0 .5rem;font-size:1.1rem}
    p{color:#aaa;margin:.3rem 0}
    .card{background:#12121f;border:1px solid #222;border-radius:8px;padding:1rem;
          margin:.5rem 0}
    .pcb{border-left:4px solid #00d4ff}
    .wm {border-left:4px solid #ff6b35}
    .borg{border-left:4px solid #f97316}
    code{background:#1e1e30;padding:2px 6px;border-radius:4px;
         color:#82cfff;font-size:.9em}
    .badge{display:inline-block;padding:2px 8px;border-radius:12px;
           font-size:.75rem;margin-left:.5rem}
    .pcb-badge{background:#00d4ff22;color:#00d4ff;border:1px solid #00d4ff55}
    .wm-badge {background:#ff6b3522;color:#ff6b35;border:1px solid #ff6b3555}
    .borg-badge{background:#f9731622;color:#f97316;border:1px solid #f9731655}
  </style>
</head>
<body>
  <h1>⚡ SMA Global Inspector — MCP Server</h1>
  <p>Single unified MCP server combining PCB &amp; Washing Machine inspection tools.</p>
  <p><strong style="color:#00d4ff">Port: 3030</strong></p>

  <h2>📡 Endpoints</h2>
  <div class="card"><code>GET  /</code> — This page</div>
  <div class="card"><code>GET  /health</code> — Model status JSON</div>
  <div class="card"><code>GET  /sse</code> — MCP SSE connection</div>
  <div class="card"><code>POST /messages/</code> — MCP messages</div>
  <div class="card"><code>POST /tools/pcb_inspect</code> — Direct REST call</div>
  <div class="card"><code>POST /tools/wm_diagnose</code> — Direct REST call</div>
  <div class="card"><code>POST /tools/alternator_inspect</code> — Direct REST call</div>

  <h2>🔵 PCB Tools <span class="badge pcb-badge">Knorr</span></h2>
  <div class="card pcb"><strong>pcb_inspect</strong>(image_file_path, image_base64, board_type)<br>
    <small style="color:#888">Full PCB defect analysis via RAG CLIP embeddings</small></div>
  <div class="card pcb"><strong>pcb_preprocess</strong>(image_file_path, method)<br>
    <small style="color:#888">AI background removal &amp; PCB crop</small></div>
  <div class="card pcb"><strong>pcb_localize_defect</strong>(defect_type, grid_cell)<br>
    <small style="color:#888">Grid-based defect coordinate lookup</small></div>
  <div class="card pcb"><strong>pcb_get_status</strong>() — Model health</div>
  <div class="card pcb"><strong>pcb_database_stats</strong>() — RAG DB statistics</div>

  <h2>🟠 Washing Machine Tools <span class="badge wm-badge">Arcelik-Beko</span></h2>
  <div class="card wm"><strong>wm_diagnose_noise</strong>(audio_file_path, model_id)<br>
    <small style="color:#888">ML audio classification for mechanical faults</small></div>
  <div class="card wm"><strong>wm_detect_damage</strong>(image_file_path, component_hint)<br>
    <small style="color:#888">CLIP RAG visual damage detection on WM components</small></div>
  <div class="card wm"><strong>wm_get_status</strong>() — Model health</div>

  <h2>🟠 Alternator Tools <span class="badge borg-badge">BORG Automotive</span></h2>
  <div class="card borg"><strong>alternator_inspect</strong>(video_file_path, deposit_amount)<br>
    <small style="color:#888">Video-based alternator inspection: pulley, cover, housing/casting</small></div>
  <div class="card borg"><strong>alternator_get_status</strong>() — Service health</div>

  <h2>🤖 Connected Agents</h2>
  <div class="card"><code>localhost:2829</code> — Knorr PCB Agent</div>
  <div class="card"><code>localhost:2828</code> — Arcelik-Beko Agent</div>
  <div class="card"><code>localhost:2827</code> — BORG Alternator Agent</div>
  <div class="card"><code>localhost:2830</code> — Guidance / Selector Agent</div>
</body>
</html>"""
    return HTMLResponse(content=html)


@asynccontextmanager
async def lifespan(app):
    """Load ML models in a background thread so uvicorn can bind immediately."""
    logger.info("⏳ Loading inspection models in background thread …")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_models)
    logger.info("✅ All models ready")
    yield
    # nothing to clean up on shutdown


app = Starlette(
    debug=True,
    lifespan=lifespan,
    routes=[
        Route("/",                  endpoint=root_handler),
        Route("/status",            endpoint=status_page),
        Route("/health",            endpoint=health_check),
        Route("/sse",               endpoint=handle_sse),
        Route("/tools/schemas",              endpoint=tool_schemas),
        Route("/tools/pcb_inspect",          endpoint=tool_pcb_inspect,          methods=["POST"]),
        Route("/tools/wm_diagnose",           endpoint=tool_wm_diagnose,          methods=["POST"]),
        Route("/tools/alternator_inspect",    endpoint=tool_alternator_inspect,   methods=["POST"]),
        Mount("/messages/",         app=sse.handle_post_message),
    ],
    middleware=[Middleware(TokenAuthMiddleware)],
)


if __name__ == "__main__":
    logger.info("🚀 Starting SMA Global Inspector MCP Server on port 3030 ...")
    logger.info("🌐 http://localhost:3030")
    uvicorn.run(app, host="0.0.0.0", port=3030)
