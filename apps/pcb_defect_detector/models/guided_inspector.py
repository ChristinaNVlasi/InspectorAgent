"""
Guided PCB Inspector — Two-stage RAG pipeline

Stage 1 — Whole-image classification (first_vb):
    • Embed the full preprocessed PCB
    • Top-1 similarity search against first_vb
    • similarity >= 0.80  →  defect_type  ("burned" or "corrosion")
    • similarity <  0.80  →  "OK", pipeline stops

Stage 2 — Cell-level localization (second_vb):
    • Split PCB into 4×4 grid (A1–D4, columns 1-4 left view)
    • Check ONLY the cells relevant for the detected defect type:
          corrosion  →  A1 and D4
          burned     →  B1 and C4
    • For each target cell: top-2 similarity search with defect_type filter
    • similarity >= 0.80  →  defect confirmed at that cell

Result contains:
    • overall_status:      OK / NOT_OK
    • classification:      burned / corrosion / OK
    • classification_confidence
    • target_cells:        list of cells inspected
    • defects_found:       list of confirmed defect locations
    • visualization:       annotated PCB image (base64 PNG)
    • cell_crops:          dict of base64 cropped cell images
    • preprocessing info
"""
import io
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add parent dirs to path when run standalone
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.clip_embedder import CLIPEmbedder
from rag.vector_store import PCBVectorStore
from utils.grid_segmenter import GridSegmenter
from preprocessing.pcb_extractor import PCBExtractor
from models.corrosion_cv_detector import CorrosionCVDetector
from config import (
    MODEL_CONFIG,
    FIRST_VDB_CONFIG,
    SECOND_VDB_CONFIG,
    PREPROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)


class GuidedPCBInspector:
    """Two-stage guided RAG inspector."""

    # Cells to inspect per defect type (4×4 grid: rows A-D, columns 1-4).
    # These MUST match the subfolder names under NOT_OK/<defect_type>/.
    DEFECT_CELLS: Dict[str, List[str]] = {
        "corrosion": ["A1", "D4"],
        "burned":    ["B1", "C4"],
    }

    # Stage 1 threshold: whole-PCB classification (first_vb)
    SIMILARITY_THRESHOLD = 0.65
    # Stage 2 threshold: cell-level localization (second_vb).
    # Intentionally lower — cell crops vs reference images naturally score lower.
    LOCALIZATION_THRESHOLD = 0.50
    # Top-K for localisation (second_vb)
    TOP_K_LOCALIZATION = 2

    # ── lifecycle ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        logger.info("Initialising GuidedPCBInspector …")

        self.embedder = CLIPEmbedder(
            model_name=MODEL_CONFIG["clip_model"],
            device=MODEL_CONFIG["device"],
        )

        self.first_vb = PCBVectorStore(
            persist_directory=FIRST_VDB_CONFIG["persist_directory"],
            collection_name=FIRST_VDB_CONFIG["collection_name"],
        )

        self.second_vb = PCBVectorStore(
            persist_directory=SECOND_VDB_CONFIG["persist_directory"],
            collection_name=SECOND_VDB_CONFIG["collection_name"],
        )

        self.grid_segmenter = GridSegmenter()

        self.pcb_extractor = PCBExtractor(
            min_area_ratio=PREPROCESSING_CONFIG["min_area_ratio"],
            padding=PREPROCESSING_CONFIG["padding"],
        )

        self.corrosion_cv = CorrosionCVDetector()

        first_count  = self.first_vb.collection.count()
        second_count = self.second_vb.collection.count()
        logger.info(
            f"GuidedPCBInspector ready — first_vb: {first_count} imgs, "
            f"second_vb: {second_count} imgs"
        )

    # ── public interface ──────────────────────────────────────────────────

    def inspect_pcb(
        self,
        image_source,
        view_type: str = "left",   # PCB is always single 4×4 (A1-D4)
        return_cell_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full guided inspection pipeline.

        Args:
            image_source: Path string OR PIL.Image.  Accepting a PIL Image
                          avoids the JPEG save-reload cycle in the API, which
                          causes lossy recompression and shifts the HSV crop.
            view_type:    Always 'left' for this 4×4 board (A1–D4).
            return_cell_details: Include per-cell results in response.

        Returns:
            Full inspection result dictionary.
        """
        if isinstance(image_source, Image.Image):
            logger.info("Inspecting: <PIL Image in memory>")
            original = image_source.convert("RGB")
        else:
            logger.info(f"Inspecting: {image_source}")
            original = Image.open(image_source).convert("RGB")

        # ── Step 1: load + preprocess ─────────────────────────────────────
        processed, crop_info = self._preprocess(original)

        # ── Step 2: first RAG — classify whole PCB ────────────────────────
        classification = self._classify_whole_image(processed)
        logger.info(
            f"Classification → {classification['defect_type'] or 'OK'} "
            f"(similarity={classification['similarity']:.3f})"
        )

        if classification["status"] == "OK":
            overlay = self.grid_segmenter.visualize_grid(processed, view_type)
            return {
                "overall_status": "OK",
                "classification": "OK",
                "classification_confidence": classification["similarity"],
                "message": "No defect detected — PCB looks healthy.",
                "target_cells": [],
                "defects_found": [],
                "defect_count": 0,
                "cell_results": {},
                "visualization": self._to_b64(processed),
                "grid_overlay":  self._to_b64(overlay),
                "preprocessing": crop_info,
                "original_size": original.size,
                "processed_size": processed.size,
            }

        defect_type   = classification["defect_type"]
        target_cells  = self.DEFECT_CELLS.get(defect_type, [])

        # ── Step 3: segment grid ──────────────────────────────────────────
        cell_images = self.grid_segmenter.segment_image(processed, view_type)

        # ── Step 4: second RAG — localize in target cells ─────────────────
        cell_results: Dict[str, Any] = {}
        defects_found: List[Dict[str, Any]] = []

        for cell_id in target_cells:
            if cell_id not in cell_images:
                logger.warning(f"Cell {cell_id} not available in segmented grid")
                continue

            cell_img = cell_images[cell_id]
            result   = self._locate_defect_in_cell(cell_img, cell_id, defect_type)
            cell_results[cell_id] = result

            if result["status"] == "NOT_OK":
                defects_found.append(
                    {
                        "location":     cell_id,
                        "defect_type":  defect_type,
                        "confidence":   result["confidence"],
                        "similarity":   result["similarity"],
                        "description":  result["description"],
                        "zoomed_image": self._to_b64(cell_img),
                        "top_matches":  result["top_matches"],
                    }
                )

        # ── Pick the ONE cell with the highest similarity score.
        #
        # Because each cell was queried ONLY against its own reference images
        # (filtered by grid_cell in second_vb), the score is a direct comparison:
        # "does this live crop look like a known A1-corrosion crop?" vs
        # "does this live crop look like a known D4-corrosion crop?"
        # The winner is the cell whose reference images best match the live crop.
        all_cell_candidates = [
            *(defects_found),
            *(
                [
                    {
                        "location":     cid,
                        "defect_type":  defect_type,
                        "confidence":   cell_results[cid].get("confidence", 0.0),
                        "similarity":   cell_results[cid].get("similarity", 0.0),
                        "description":  cell_results[cid].get("description", ""),
                        "zoomed_image": self._to_b64(cell_images[cid]) if cid in cell_images else None,
                        "top_matches":  cell_results[cid].get("top_matches", []),
                    }
                    for cid in cell_results
                    if cid not in {d["location"] for d in defects_found}
                ]
            ),
        ]

        if all_cell_candidates:
            best = max(all_cell_candidates, key=lambda d: d.get("similarity", 0.0))
            defects_found = [best]
            cell_results[best["location"]]["status"] = "NOT_OK"
            logger.info(
                f"Winning cell → {best['location']} "
                f"(similarity={best['similarity']:.3f})"
            )

        # Stage 1 already confirmed a defect — overall is always NOT_OK here
        overall_status = "NOT_OK"

        # ── Step 5: visualise ─────────────────────────────────────────────
        visualization = self._create_visualization(
            processed, defects_found, target_cells, view_type
        )

        result = {
            "overall_status":             overall_status,
            "classification":             defect_type,
            "classification_confidence":  classification["similarity"],
            "message": (
                f"{defect_type.capitalize()} defect detected. "
                f"Checked cells: {', '.join(target_cells)}."
            ),
            "target_cells":    target_cells,
            "defects_found":   defects_found,
            "defect_count":    len(defects_found),
            "total_cells":     len(cell_images),
            "healthy_cells":   len(cell_images) - len(defects_found),
            "view_type":       view_type,
            "visualization":   self._to_b64(visualization),
            "grid_overlay":    self._to_b64(
                self.grid_segmenter.visualize_grid(processed, view_type)
            ),
            "preprocessing":   crop_info,
            "original_size":   original.size,
            "processed_size":  processed.size,
        }

        if return_cell_details:
            # Include cropped images for ALL target cells (flagged or healthy)
            cells_with_crops = {}
            for cell_id in target_cells:
                cell_data = cell_results.get(cell_id, {})
                if cell_id in cell_images:
                    cell_data["cell_image"] = self._to_b64(cell_images[cell_id])
                cells_with_crops[cell_id] = cell_data
            result["cell_results"] = cells_with_crops

        logger.info(
            f"Inspection done — {overall_status}, "
            f"defects at: {[d['location'] for d in defects_found]}"
        )
        return result

    def get_database_stats(self) -> Dict[str, Any]:
        """Return stats for health check endpoint."""
        try:
            return {
                "first_vb_count":  self.first_vb.collection.count(),
                "second_vb_count": self.second_vb.collection.count(),
                "threshold":       self.SIMILARITY_THRESHOLD,
                "defect_cells":    self.DEFECT_CELLS,
                "grid_processing_enabled": True,
            }
        except Exception as exc:
            logger.error(f"Stats error: {exc}")
            return {"error": str(exc)}

    # ── internal steps ────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Remove background and crop to PCB."""
        if not PREPROCESSING_CONFIG.get("enabled", True):
            return image, {"detected": False, "method": "none"}
        try:
            processed, info = self.pcb_extractor.extract_pcb(
                image, method=PREPROCESSING_CONFIG["method"]
            )
            if info.get("detected", False):
                return processed, info
            logger.warning("PCB detection failed — using original image")
            return image, info
        except Exception as exc:
            logger.error(f"Preprocessing failed: {exc}", exc_info=True)
            return image, {"detected": False, "error": str(exc)}

    def _dist_to_sim(self, distance: float) -> float:
        """
        Convert ChromaDB cosine distance to cosine similarity.
        For L2-normalised embeddings: distance ∈ [0, 2],  similarity = 1 − distance.
        Clip to [0, 1] in case of slight numerical issues.
        """
        return float(np.clip(1.0 - distance, 0.0, 1.0))

    def _classify_whole_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Stage 1: query first_vb with the whole processed PCB.
        Returns {'status', 'defect_type', 'similarity'}.
        """
        embedding = self.embedder.encode_image(image)
        results   = self.first_vb.search_similar(embedding, n_results=1)

        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        if not distances:
            logger.warning("first_vb returned no results — treating as OK")
            return {"status": "OK", "defect_type": None, "similarity": 0.0}

        similarity  = self._dist_to_sim(distances[0])
        defect_type = (metadatas[0] if metadatas else {}).get("defect_type", "unknown")

        if similarity < self.SIMILARITY_THRESHOLD:
            return {"status": "OK", "defect_type": None, "similarity": similarity}

        return {"status": "NOT_OK", "defect_type": defect_type, "similarity": similarity}

    def _locate_defect_in_cell(
        self,
        cell_image:  Image.Image,
        grid_coord:  str,
        defect_type: str,
    ) -> Dict[str, Any]:
        """
        Stage 2: query second_vb for one specific cell.

        Filters by BOTH defect_type AND grid_cell so each live cell crop is
        compared only against its own reference images — e.g. the live A1 crop
        is compared only against corrosion/A1 reference crops, and the live D4
        crop only against corrosion/D4 reference crops.

        Whichever cell returns the higher similarity score is reported as the
        defect location by the caller (guided_inspector.inspect_pcb).
        """
        embedding = self.embedder.encode_image(cell_image)

        results = self.second_vb.search_similar(
            query_embedding=embedding,
            n_results=self.TOP_K_LOCALIZATION,
            where={"$and": [{"defect_type": defect_type}, {"grid_cell": grid_coord}]},
        )

        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        if not distances:
            similarities = []
            max_rag_sim  = 0.0
        else:
            similarities = [self._dist_to_sim(d) for d in distances]
            max_rag_sim  = max(similarities)

        # ── Classical CV path (corrosion only) ────────────────────────────────
        # Run the CV detector on the cell crop and normalise its coverage ratio
        # to a 0-1 score comparable with the RAG similarity.
        # cv_score = 0.5 at 2.5 % coverage, 1.0 at 5 % coverage (MILD threshold).
        cv_result  = None
        cv_score   = 0.0
        if defect_type == "corrosion":
            cv_result = self.corrosion_cv.detect(cell_image)
            cv_score  = min(cv_result["ratio"] / 0.05, 1.0)
            logger.info(
                f"Cell {grid_coord} CV: severity={cv_result['severity']} "
                f"ratio={cv_result['ratio']:.4f}  score={cv_score:.3f}  "
                f"board_hue={cv_result.get('board_hue', 0):.0f}"
            )

        # Combined score — take the higher of RAG and CV signals.
        # This means corrosion is reported even if RAG similarity is low,
        # as long as the classical detector sees oxidation pixels.
        combined = max(max_rag_sim, cv_score)

        logger.info(
            f"Cell {grid_coord} ({defect_type}): "
            f"RAG={max_rag_sim:.3f}  CV={cv_score:.3f}  combined={combined:.3f}  "
            f"threshold={self.LOCALIZATION_THRESHOLD}"
        )

        top_matches = [
            {
                "defect_type": (metadatas[i] if i < len(metadatas) else {}).get(
                    "defect_type", defect_type
                ),
                "similarity": similarities[i],
                "filename":   (metadatas[i] if i < len(metadatas) else {}).get(
                    "original_filename", "unknown"
                ),
            }
            for i in range(len(similarities))
        ]

        # Defect confirmed if either RAG or CV (for corrosion) crosses threshold
        rag_hit = max_rag_sim >= self.LOCALIZATION_THRESHOLD
        cv_hit  = (cv_result is not None) and cv_result["detected"]
        is_defect = rag_hit or cv_hit

        if is_defect:
            sources = []
            if rag_hit:
                sources.append(f"RAG {max_rag_sim:.1%}")
            if cv_hit:
                sev = cv_result['severity'] if cv_result else ''
                sources.append(f"CV {sev} ({cv_result['ratio']*100:.1f}% coverage)")
            description = (
                f"{defect_type.capitalize()} confirmed in cell {grid_coord} "
                f"[{', '.join(sources)}]"
            )
            return {
                "status":       "NOT_OK",
                "confidence":   combined,
                "similarity":   combined,
                "rag_similarity": max_rag_sim,
                "cv_score":     cv_score,
                "cv_severity":  cv_result["severity"] if cv_result else None,
                "description":  description,
                "top_matches":  top_matches,
            }

        description = (
            f"Cell {grid_coord} — no defect "
            f"(RAG {max_rag_sim:.1%}, CV {cv_score:.1%}, "
            f"threshold {self.LOCALIZATION_THRESHOLD:.0%})"
        )
        return {
            "status":       "OK",
            "confidence":   combined,
            "similarity":   combined,
            "rag_similarity": max_rag_sim,
            "cv_score":     cv_score,
            "cv_severity":  cv_result["severity"] if cv_result else None,
            "description":  description,
            "top_matches":  top_matches,
        }

    # ── visualisation ─────────────────────────────────────────────────────

    def _create_visualization(
        self,
        image:        Image.Image,
        defects:      List[Dict],
        target_cells: List[str],
        view_type:    str,
    ) -> Image.Image:
        """
        Draw grid + mark targeted cells and confirmed defects.
        - target cells (inspected but OK) → yellow outline
        - defective cells                 → red outline + label
        """
        vis = self.grid_segmenter.visualize_grid(
            image.copy(), view_type,
            line_color="rgba(255,255,255,120)",
            line_width=2,
            show_labels=True,
        )
        draw = ImageDraw.Draw(vis)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()

        # Mark all target cells with yellow outline (inspected)
        for cell_id in target_cells:
            try:
                x1, y1, x2, y2 = self.grid_segmenter.get_cell_bounds(
                    image.size, cell_id, view_type
                )
                draw.rectangle([x1, y1, x2, y2], outline="#FFD700", width=3)
            except Exception as exc:
                logger.warning(f"Could not draw target outline for {cell_id}: {exc}")

        # Mark defective cells in red
        for defect in defects:
            cell_id = defect["location"]
            try:
                x1, y1, x2, y2 = self.grid_segmenter.get_cell_bounds(
                    image.size, cell_id, view_type
                )
                draw.rectangle([x1, y1, x2, y2], outline="#FF3333", width=5)

                label     = f"{cell_id}: {defect['defect_type']}\n{defect['confidence']:.0%}"
                label_x   = x1 + 4
                label_y   = y1 + 4
                bbox_text = draw.textbbox((label_x, label_y), label, font=font)
                draw.rectangle(
                    [bbox_text[0] - 2, bbox_text[1] - 2,
                     bbox_text[2] + 2, bbox_text[3] + 2],
                    fill="#CC0000",
                )
                draw.text((label_x, label_y), label, fill="white", font=font)
            except Exception as exc:
                logger.warning(f"Could not mark defect at {cell_id}: {exc}")

        return vis

    def _to_b64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 PNG string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
