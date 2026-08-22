"""
Classical Computer Vision Corrosion Detector
=============================================
Detects copper oxidation / verdigris on PCB images using dynamic HSV
hue analysis relative to the board's own measured substrate colour.

Key principle:
  - Corrosion (verdigris) is CYAN-shifted compared to the normal PCB green.
  - Instead of fixed thresholds (which fail when board colour varies), we
    measure the board's own median hue first, then flag pixels that are
    shifted MORE CYAN than that baseline.
  - Two passes handle both bright (well-lit) and shadowed corrosion spots.
  - A one-shot proximity grow step captures adjacent oxidation that sits
    just below the strict saturation cut-off.

Works on:
  - Full PCB images
  - Individual cell crops (4×4 grid cells)
  - Any image size; parameters scale automatically with image dimensions.

No YOLO, no deep learning — runs on CPU with only opencv-python + numpy.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Global thresholds ──────────────────────────────────────────────────────────
CORROSION_RATIO_THRESHOLD = 0.005   # 0.5 % board coverage → positive detection
HUE_SHIFT_THRESHOLD       = 12      # degrees above board median hue → corrosion
BOARD_HUE_MIN             = 25      # minimum board median hue for a valid green PCB
MIN_CORROSION_SATURATION  = 55      # mean S inside a contour — rejects blurry FPs


class CorrosionCVDetector:
    """
    Stateless detector.  Call ``detect(pil_image)`` on any PIL.Image.

    Returns a dict:
        detected      – bool
        severity      – "NONE" | "MILD" | "MODERATE" | "SEVERE"
        ratio         – corrosion area / board area  (float 0-1)
        regions_count – number of corrosion blobs found
        board_hue     – measured board hue (for debugging)
    """

    def detect(self, pil_image: Image.Image) -> dict:
        """Run full CV corrosion pipeline on a PIL Image."""
        try:
            img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            return self._run(img)
        except Exception as exc:
            logger.warning(f"CorrosionCVDetector.detect() failed: {exc}", exc_info=True)
            return {"detected": False, "severity": "NONE", "ratio": 0.0,
                    "regions_count": 0, "board_hue": 0.0}

    # ── main pipeline ──────────────────────────────────────────────────────────

    def _run(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]
        area  = h * w
        min_dim = min(h, w)

        # Adaptive kernel sizes — scale with image dimensions so the detector
        # works equally well on full PCB images (~1000+ px) and cell crops (~200 px).
        board_k    = max(5,  int(min_dim * 0.018))   # board mask close/open
        interior_k = max(7,  int(min_dim * 0.06))    # interior erosion kernel
        grow_px    = max(10, int(min_dim * 0.08))    # proximity grow radius
        # Contour filters — scaled to image area relative to reference 1024×1024
        area_scale  = area / (1024 * 1024)
        min_area    = max(50,  int(1500 * area_scale))
        min_short   = max(5,   int(35   * (min_dim / 1024.0)))

        # ── 1. Preprocessing (denoise + colour balance — NO CLAHE/sharpen) ────
        img = self._denoise(img)
        img = self._white_patch_balance(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ── 2. Board mask — isolate PCB substrate from background ─────────────
        board_mask = self._build_board_mask(hsv, board_k)
        board_area = int(cv2.countNonZero(board_mask))

        # If the mask is too small (e.g. dark/non-standard board, or tiny crop),
        # fall back to using the full image as board so detection still runs.
        if board_area < area * 0.05:
            logger.debug("CV: board mask too small — using full image as board")
            board_mask = np.ones((h, w), dtype=np.uint8) * 255
            board_area = area

        # ── 3. Measure the board's OWN hue & brightness from interior pixels ──
        board_hue, board_v = self._measure_board_colour(hsv, board_mask, interior_k)
        logger.debug(
            f"CV: board_hue={board_hue:.0f} board_v={board_v:.0f} "
            f"board_k={board_k} interior_k={interior_k}"
        )

        # Sanity check — if board hue is outside the green range it's not a
        # standard PCB substrate; our dynamic thresholds would be mis-calibrated.
        if board_hue < BOARD_HUE_MIN:
            logger.debug(f"CV: board_hue={board_hue:.0f} < {BOARD_HUE_MIN} — not a green PCB")
            return {"detected": False, "severity": "NONE", "ratio": 0.0,
                    "regions_count": 0, "board_hue": board_hue}

        # ── 4. Two-pass dynamic corrosion mask ────────────────────────────────
        corr_mask = self._build_corrosion_mask(
            hsv, board_mask, board_hue, board_v, board_k, interior_k
        )

        # ── 5. Strict contour filter → validated seed mask ────────────────────
        strict_regions = self._find_corrosion_contours(
            corr_mask, hsv, min_area, min_short, MIN_CORROSION_SATURATION
        )

        if strict_regions:
            seed_mask = np.zeros((h, w), dtype=np.uint8)
            for r in strict_regions:
                cv2.drawContours(seed_mask, [r["contour"]], -1, 255, -1)

            # ── 6. Proximity grow — capture adjacent oxidation that shares the
            #        same physical corrosion spot but is just below strict sat cut.
            grown = self._grow_corrosion_mask(
                hsv, seed_mask, board_mask, board_hue, grow_px=grow_px
            )
            regions = self._find_corrosion_contours(
                grown, hsv, min_area, min_short, min_sat=45
            )
        else:
            regions = []

        # ── 7. Coverage ratio & severity ──────────────────────────────────────
        corr_area = sum(r["area"] for r in regions)
        ratio     = corr_area / max(board_area, 1)
        severity  = self._classify_severity(ratio)
        detected  = ratio > CORROSION_RATIO_THRESHOLD

        logger.info(
            f"CV corrosion: detected={detected}  ratio={ratio:.3f}  "
            f"severity={severity}  regions={len(regions)}  board_hue={board_hue:.0f}"
        )
        return {
            "detected":      detected,
            "severity":      severity,
            "ratio":         ratio,
            "regions_count": len(regions),
            "board_hue":     board_hue,
        }

    # ── preprocessing helpers ──────────────────────────────────────────────────

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            img, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _white_patch_balance(self, img: np.ndarray) -> np.ndarray:
        result = img.astype(np.float32)
        for i in range(3):
            max_val = np.percentile(result[:, :, i], 99)
            result[:, :, i] = np.clip(
                result[:, :, i] * (255.0 / (max_val + 1e-6)), 0, 255
            )
        return result.astype(np.uint8)

    # ── board isolation ────────────────────────────────────────────────────────

    def _build_board_mask(self, hsv: np.ndarray, k: int) -> np.ndarray:
        """Binary mask of the PCB green substrate."""
        board = cv2.inRange(
            hsv,
            np.array([25, 20, 50]),
            np.array([95, 255, 255]),
        )
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, ke)
        board = cv2.morphologyEx(board, cv2.MORPH_OPEN,  ke)
        return board

    def _measure_board_colour(
        self, hsv: np.ndarray, board_mask: np.ndarray, k_int: int
    ):
        """Return (hue_median, value_median) of the interior PCB green pixels."""
        # Try interior first (eroded board mask — avoids component edges)
        ke  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_int, k_int))
        interior = cv2.erode(board_mask, ke)
        green    = cv2.inRange(hsv, np.array([25, 40, 60]), np.array([80, 200, 170]))
        sample   = cv2.bitwise_and(green, interior)

        h_ch, _, v_ch = cv2.split(hsv)
        h_vals = h_ch[sample > 0]
        v_vals = v_ch[sample > 0]
        if len(h_vals) > 20:
            return float(np.median(h_vals)), float(np.median(v_vals))

        # Fallback: relax thresholds and use full board mask
        # (needed for small cell crops where interior erosion removes everything)
        green2  = cv2.inRange(hsv, np.array([25, 30, 40]), np.array([85, 220, 190]))
        sample2 = cv2.bitwise_and(green2, board_mask)
        h2      = h_ch[sample2 > 0]
        v2      = v_ch[sample2 > 0]
        if len(h2) > 10:
            return float(np.median(h2)), float(np.median(v2))

        return 45.0, 120.0   # sensible default for green FR4

    # ── corrosion masking ──────────────────────────────────────────────────────

    def _build_corrosion_mask(
        self,
        hsv:        np.ndarray,
        board_mask: np.ndarray,
        board_hue:  float,
        board_v:    float,
        board_k:    int,
        interior_k: int,
    ) -> np.ndarray:
        """
        Two-pass dynamic hue masking.

        Pass A — strong hue shift (board_hue + 20°), relaxed brightness.
                  Catches corrosion in shadow / under-exposed areas.
        Pass B — moderate hue shift (board_hue + 12°), must be brighter than board.
                  Catches bright oxidation patches in well-lit images.

        Both passes share an absolute cyan floor (H ≥ 60) to prevent metallic
        connectors on olive boards (hue ≈ 55-58) from triggering detection.
        """
        CORR_HUE_ABS_MIN = 60

        shift_b  = 20 if board_hue < 40 else HUE_SHIFT_THRESHOLD
        hue_lo_a = max(int(board_hue + 20),    CORR_HUE_ABS_MIN)
        hue_lo_b = max(int(board_hue + shift_b), CORR_HUE_ABS_MIN)
        v_lo_b   = int(board_v + 20)

        # Use interior pixels to avoid edge effects; fall back if interior is empty
        ke       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (interior_k, interior_k))
        interior = cv2.erode(board_mask, ke)
        if cv2.countNonZero(interior) < 100:
            interior = board_mask

        # Exclusions
        not_solder = cv2.bitwise_not(
            cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 30, 255]))
        )
        not_black = cv2.inRange(
            hsv, np.array([0, 0, 25]), np.array([180, 255, 255])
        )

        # Pass A: strong cyan hue shift, any brightness above shadow
        pass_a = cv2.inRange(
            hsv,
            np.array([hue_lo_a, 20, 30]),
            np.array([115, 220, 255]),
        )
        # Pass B: moderate cyan shift, must be brighter than the board
        pass_b = cv2.inRange(
            hsv,
            np.array([hue_lo_b, 20, v_lo_b]),
            np.array([115, 220, 255]),
        )

        corr = cv2.bitwise_or(pass_a, pass_b)
        corr = corr & not_solder & not_black & interior

        # Morphological cleanup — scale kernel to image
        ke11 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (min(11, board_k), min(11, board_k))
        )
        ke5 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, board_k // 2), max(3, board_k // 2))
        )
        corr = cv2.morphologyEx(corr, cv2.MORPH_CLOSE, ke11)
        corr = cv2.morphologyEx(corr, cv2.MORPH_OPEN,  ke5)
        return corr

    def _grow_corrosion_mask(
        self,
        hsv:        np.ndarray,
        strict:     np.ndarray,
        board_mask: np.ndarray,
        board_hue:  float,
        grow_px:    int = 20,
        grow_min_sat: int = 50,
    ) -> np.ndarray:
        """Expand confirmed corrosion blobs to capture adjacent oxidation."""
        if cv2.countNonZero(strict) == 0:
            return strict

        k_grow  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1)
        )
        adj_zone = cv2.dilate(strict, k_grow) & board_mask

        hue_lo     = int(board_hue + 10)
        not_solder = cv2.bitwise_not(
            cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 30, 255]))
        )
        not_black = cv2.inRange(
            hsv, np.array([0, 0, 25]), np.array([180, 255, 255])
        )
        relaxed = (
            cv2.inRange(
                hsv,
                np.array([hue_lo, grow_min_sat, 30]),
                np.array([115, 255, 255]),
            )
            & not_solder & not_black & adj_zone
        )

        combined = cv2.bitwise_or(strict, relaxed)
        ke_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (min(15, grow_px // 2 + 3), min(15, grow_px // 2 + 3)),
        )
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, ke_close)

    def _find_corrosion_contours(
        self,
        mask:      np.ndarray,
        hsv:       np.ndarray,
        min_area:  int,
        min_short: int,
        min_sat:   int = MIN_CORROSION_SATURATION,
    ) -> list:
        """Extract and quality-filter corrosion contours."""
        h_img, w_img = mask.shape[:2]
        s_ch = hsv[:, :, 1]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if min(bw, bh) < min_short:
                continue
            # Saturation filter: low-S regions are blur / shadow artefacts
            cnt_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_sat = float(s_ch[cnt_mask > 0].mean())
            if mean_sat < min_sat:
                continue
            regions.append({
                "contour":  cnt,
                "bbox":     (x, y, bw, bh),
                "area":     area,
                "center":   (x + bw // 2, y + bh // 2),
                "mean_sat": mean_sat,
            })
        return sorted(regions, key=lambda r: r["area"], reverse=True)

    # ── severity ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_severity(ratio: float) -> str:
        if ratio == 0:      return "NONE"
        elif ratio < 0.02:  return "MILD"
        elif ratio < 0.08:  return "MODERATE"
        else:               return "SEVERE"
