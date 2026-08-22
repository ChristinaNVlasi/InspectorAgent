"""
PCB Extractor — HSV-based green detection.

Primary method: isolate the PCB green in HSV space, find its bounding box.
  - Lighting-invariant: hue is stable across brightness changes.
  - Robust: works with any non-green background colour.
  - Fast: single pass, no iterative morphology.

Fallback: rembg AI removal.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available, AI background removal disabled")


class PCBExtractor:
    """Automatically extract PCB region from images with background"""

    def __init__(self, min_area_ratio: float = 0.10, padding: int = 10):
        self.min_area_ratio = min_area_ratio
        self.padding = padding

    def extract_pcb(self, image: Image.Image, method: str = "auto") -> Tuple[Image.Image, dict]:
        """Crop the PCB out of the image using HSV green detection.
        Only crops — does not modify colours or background in any way.

        Two-pass approach:
          1. Coarse crop: detect the green PCB in the full image, crop to its bbox.
          2. Tight crop:  re-run the green mask on the coarse crop to trim any dark
                         border (black holder, table, etc.) that crept into pass 1.
        The grid segmenter always receives the tight crop so cells align to the PCB.
        """
        logger.info(f"Extracting PCB from image using method: {method}")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        bbox = self._hsv_green_crop(img_cv)
        if bbox is not None:
            coarse, info = self._apply_crop(image, img_cv, bbox, "hsv_green")
            tight, tight_info = self._tighten_to_green(coarse)
            info.update(tight_info)
            return tight, info

        if REMBG_AVAILABLE:
            logger.info("HSV green crop failed — trying rembg")
            try:
                return self._ai_background_removal(image)
            except Exception as e:
                logger.warning(f"rembg failed: {e}")

        logger.warning("All crop methods failed — returning original image")
        return image, {"detected": False, "method": "none", "message": "PCB detection failed"}

    def _tighten_to_green(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """
        Second-pass crop: given an already-coarsely-cropped image, find the
        tight bounding box of the green PCB substrate and crop to it.

        This removes dark borders (black plastic holder, shadows) that the
        coarse pass included because the holder physically surrounds the PCB.
        Without this, the 4×4 grid cells are shifted into the black area and
        defects are reported in the wrong cell.

        Uses the same detection-only whitening trick as _hsv_green_crop:
        dark pixels are replaced with white on a copy before masking so the green
        substrate is clearly separable. The returned pixels are always the
        original unmodified colours.
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # Detection-only copy: white-out dark pixels
        detect = img_cv.copy()
        hsv_tmp = cv2.cvtColor(detect, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv_tmp, np.array([0, 0, 0]), np.array([180, 255, 60]))
        detect[dark_mask == 255] = [255, 255, 255]

        hsv = cv2.cvtColor(detect, cv2.COLOR_BGR2HSV)
        mask_a = cv2.inRange(hsv, np.array([35, 50, 40]), np.array([90, 255, 255]))
        mask_b = cv2.inRange(hsv, np.array([20, 40, 30]), np.array([90, 210, 180]))
        green_mask = cv2.bitwise_or(mask_a, mask_b)

        # Close small holes so the green region is solid
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_k)

        coords = cv2.findNonZero(green_mask)
        if coords is None:
            logger.info("Tighten: no green found in coarse crop — keeping coarse crop")
            return image, {"tightened": False}

        x, y, bw, bh = cv2.boundingRect(coords)

        # Keep a small padding so we don't cut the very edge of the PCB
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        # Only apply if the tight box is meaningfully smaller on at least one side
        # (≥4% of width/height removed) — avoids unnecessary crops when the PCB
        # already fills the coarse crop cleanly.
        removed_left   = x1 / w
        removed_top    = y1 / h
        removed_right  = (w - x2) / w
        removed_bottom = (h - y2) / h
        meaningful = max(removed_left, removed_top, removed_right, removed_bottom) >= 0.04

        if meaningful:
            tight = image.crop((x1, y1, x2, y2))
            logger.info(
                f"Tighten: removed L={removed_left:.1%} T={removed_top:.1%} "
                f"R={removed_right:.1%} B={removed_bottom:.1%} → {tight.size}"
            )
            return tight, {"tightened": True, "tight_bbox": (x1, y1, x2 - x1, y2 - y1)}

        logger.info("Tighten: coarse crop already tight — no change")
        return image, {"tightened": False}

    def preprocess_for_inspection(self, image_path: str) -> Tuple[Image.Image, dict]:
        image = Image.open(image_path)
        return self.extract_pcb(image)

    def _hsv_green_crop(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the PCB green substrate in HSV space.
        Hue is independent of brightness — robust to lighting changes.
        Runs on a 600px downscale; result is scaled back to original size.

        Detection trick: dark/black pixels are replaced with white on a
        detection-only copy BEFORE the green mask is applied.  This prevents
        the black plastic holder and black components from breaking up the green
        area or being mistaken for background.  The original image colours are
        never touched — the crop is always applied to the unmodified pixels.
        """
        orig_h, orig_w = img.shape[:2]

        # ── Step 1: Downscale for speed (longest side → 600px) ──────────────────
        scale = min(1.0, 600.0 / max(orig_h, orig_w))
        small = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        # ── Step 2: Detection-only copy — replace dark pixels with white ─────────
        # Dark pixels (V < 60 in HSV) are the black plastic holder, black
        # components, and shadows.  Turning them white ensures the green mask
        # sees a clean green substrate on a white background rather than green
        # fragments surrounded by black noise.
        # This copy is ONLY used for mask computation — the crop itself always
        # uses the original unmodified image.
        detect = small.copy()
        hsv_tmp = cv2.cvtColor(detect, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv_tmp, np.array([0, 0, 0]), np.array([180, 255, 60]))
        detect[dark_mask == 255] = [255, 255, 255]

        # ── Step 3: BGR → HSV on the brightened detection copy ──────────────────
        hsv = cv2.cvtColor(detect, cv2.COLOR_BGR2HSV)

        # ── Step 4: Green mask (two ranges unioned) ──────────────────────────────
        # Range A — bright/saturated PCB green (classic FR4 / standard green boards)
        #   H 35-90: pure green band, covers lime → forest green
        #   S 50+: well-saturated; excludes low-sat grey/white backgrounds
        #   V 40+: sufficiently bright pixels
        mask_a = cv2.inRange(hsv,
                             np.array([35, 50, 40]),
                             np.array([90, 255, 255]))

        # Range B — oil-green / olive / dark-muted PCB green
        #   H 20-90: covers yellow-green, olive, khaki
        #   S 40+: raised from 18 to exclude neutral/grey backgrounds (carpet, table)
        #   V 30+: still catches dimly lit substrate areas
        mask_b = cv2.inRange(hsv,
                             np.array([20, 40, 30]),
                             np.array([90, 210, 180]))

        mask = cv2.bitwise_or(mask_a, mask_b)

        # ── Step 5: MORPH_CLOSE → fill holes in the PCB surface ─────────────────
        # PCB substrates have components/traces that break up the green area into
        # fragments. CLOSE (dilate→erode) reconnects them into one solid region.
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        # MORPH_OPEN (small, 3×3): remove any tiny isolated noise specks
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

        # ── Step 6: Find largest contour (the PCB body) ──────────────────────────
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.info("HSV green crop: no green contours found")
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(best_cnt)

        ar = (bw * bh) / (sw * sh)
        if ar < 0.05:
            logger.info(f"HSV green crop: contour too small ({ar:.1%})")
            return None
        if ar > 0.85:
            logger.info(f"HSV green crop: contour suspiciously large ({ar:.1%}) — likely background noise, rejecting")
            return None

        logger.info(f"HSV green crop OK: {bw}×{bh} @ ({x},{y}), area={ar:.1%} on {sw}×{sh}")

        # ── Step 7: Scale bbox back to original resolution ───────────────────────
        inv = 1.0 / scale
        return (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv))

    def _mask_green_only(self, image: Image.Image) -> Image.Image:
        """
        Remove only the background strips that sit at the edges of the (already
        cropped) image.  PCB interior is NEVER touched.

        Strategy: border-zone masking.
        Any pixel that is (a) NOT green AND (b) within the edge strip is
        background.  The edge strip is 5 % of the shorter image dimension
        (minimum 20 px) — wide enough to catch leaked background, narrow enough
        that chips/connectors deep inside the PCB are never affected.

        This avoids flood-fill entirely, which was unreliable when dark
        background colours matched dark PCB components, causing interior
        components to be erased.
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        # Green mask — same two-range union as the crop step
        mask_a = cv2.inRange(hsv, np.array([35, 50, 40]),  np.array([90, 255, 255]))
        mask_b = cv2.inRange(hsv, np.array([20, 20, 20]),  np.array([75, 200, 180]))
        green_mask = cv2.bitwise_or(mask_a, mask_b)   # 255 = green (PCB), 0 = non-green
        non_green  = cv2.bitwise_not(green_mask)

        # Build the border zone: a ring around the image perimeter
        border_px = max(20, int(min(h, w) * 0.05))
        border_zone = np.zeros((h, w), dtype=np.uint8)
        border_zone[:border_px, :]  = 255   # top
        border_zone[-border_px:, :] = 255   # bottom
        border_zone[:, :border_px]  = 255   # left
        border_zone[:, -border_px:] = 255   # right

        # Background = non-green pixel that sits in the border zone
        bg_mask = cv2.bitwise_and(non_green, border_zone)

        # Fill with the median colour of the non-background PCB pixels.
        # Median is more robust than mean when components skew the average.
        result = img_cv.copy()
        pcb_pixels = img_cv[bg_mask == 0]
        if len(pcb_pixels) > 100:
            fill_color = np.median(pcb_pixels, axis=0).astype(np.uint8)
        else:
            fill_color = np.array([60, 100, 60], dtype=np.uint8)  # fallback: plain green
        result[bg_mask == 255] = fill_color

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def _apply_crop(self, pil_image, img_cv, bbox, method_name):
        orig_h, orig_w = img_cv.shape[:2]
        x, y, bw, bh = bbox

        xp = max(0, x - self.padding)
        yp = max(0, y - self.padding)
        wp = min(orig_w - xp, bw + 2 * self.padding)
        hp = min(orig_h - yp, bh + 2 * self.padding)

        cropped = pil_image.crop((xp, yp, xp + wp, yp + hp))
        area_ratio = (wp * hp) / (orig_w * orig_h)
        margin_ratio = 1.0 - area_ratio

        info = {
            "detected": True, "method": method_name,
            "bbox": (xp, yp, wp, hp), "original_size": (orig_w, orig_h),
            "cropped_size": (wp, hp), "area_ratio": area_ratio,
            "margin_ratio": margin_ratio,
            "message": f"{method_name}: margins={margin_ratio:.1%}",
        }
        logger.info(f"{method_name} crop: {wp}x{hp} (area: {area_ratio:.1%}, margins: {margin_ratio:.1%})")
        return cropped, info

    def _ai_background_removal(self, image):
        logger.info("Using AI background removal (rembg)...")
        no_bg = remove(image)
        np_img = np.array(no_bg)
        if np_img.shape[2] != 4:
            return image, {"detected": False, "method": "ai", "message": "No alpha channel"}
        alpha = np_img[:, :, 3]
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return image, {"detected": False, "method": "ai", "message": "No foreground detected"}
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        x, y, w, h = cv2.boundingRect(coords)
        return self._apply_crop(image, img_cv, (x, y, w, h), "ai")
