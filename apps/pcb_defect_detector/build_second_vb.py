"""
Build SECOND vector DB (second_vb) for grid-cell defect localization.

Expected source layout
──────────────────────
  data/EBS7_TCM/NOT_OK/
    corrosion/
      A1/   ← cell-level reference crops showing corrosion at grid cell A1
      D4/   ← cell-level reference crops showing corrosion at grid cell D4
    burned/
      B1/   ← cell-level reference crops showing burn damage at grid cell B1
      C4/   ← cell-level reference crops showing burn damage at grid cell C4

Each image is stored with two metadata keys:
  • defect_type  — parent-folder name  ("corrosion" | "burned")
  • grid_cell    — subfolder name      ("A1" | "D4" | "B1" | "C4")

At query time the inspector crops the live image at a specific cell and asks
"does this crop look like grid_cell=A1 corrosion?", filtering by both keys so
each cell is only compared against its own reference images.
The cell with the highest similarity score wins and is reported as the location.

Run:
    cd pcb_defect_detector
    python build_second_vb.py [--reset]
"""
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from embeddings.clip_embedder import CLIPEmbedder
from rag.vector_store import PCBVectorStore
from config import (
    PCB_DATA_DIR,
    MODEL_CONFIG,
    SECOND_VDB_CONFIG,
    PCB_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Location inside EBS7_TCM
NOT_OK_DIR = PCB_DATA_DIR / "NOT_OK"

# Top-level folder name → defect_type label
DEFECT_FOLDERS = {
    "burned":    "burned",
    "corrosion": "corrosion",
}


def build_second_vb(reset: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("Building SECOND_VB — cell-level defect localization database")
    logger.info(f"Source : {NOT_OK_DIR}")
    logger.info(f"Target : {SECOND_VDB_CONFIG['persist_directory']}")
    logger.info("=" * 60)

    # ── initialise ────────────────────────────────────────────────────────
    embedder = CLIPEmbedder(
        model_name=MODEL_CONFIG["clip_model"],
        device=MODEL_CONFIG["device"],
    )

    store = PCBVectorStore(
        persist_directory=SECOND_VDB_CONFIG["persist_directory"],
        collection_name=SECOND_VDB_CONFIG["collection_name"],
    )

    if reset:
        logger.warning("Resetting second_vb …")
        store.reset()

    # ── collect images ────────────────────────────────────────────────────
    # Layout: NOT_OK/<defect_type>/<grid_cell>/<image files>
    extensions = set(PCB_CONFIG["image_extensions"])
    # Each entry: (image_path, defect_type, grid_cell)
    image_entries: list[tuple[Path, str, str]] = []

    if not NOT_OK_DIR.exists():
        logger.error(f"NOT_OK directory not found: {NOT_OK_DIR}")
        return

    for defect_folder in sorted(NOT_OK_DIR.iterdir()):
        if not defect_folder.is_dir():
            continue
        defect_type = DEFECT_FOLDERS.get(defect_folder.name.lower())
        if defect_type is None:
            logger.warning(f"Unknown defect folder '{defect_folder.name}' — skipping.")
            continue

        for cell_folder in sorted(defect_folder.iterdir()):
            if not cell_folder.is_dir():
                continue
            grid_cell = cell_folder.name.upper()   # e.g. "A1", "D4", "B1", "C4"
            images_in_cell = [
                p for p in sorted(cell_folder.iterdir())
                if p.suffix.lower() in extensions
            ]
            if not images_in_cell:
                logger.warning(f"  No images in {defect_folder.name}/{cell_folder.name}")
                continue
            for img_path in images_in_cell:
                image_entries.append((img_path, defect_type, grid_cell))
            logger.info(f"  {defect_type}/{grid_cell}: {len(images_in_cell)} images")

    if not image_entries:
        logger.error(f"No images found under {NOT_OK_DIR}")
        return

    logger.info(f"Total: {len(image_entries)} cell images to embed.")

    # ── embed & store ─────────────────────────────────────────────────────
    # Reference images are already cropped cell-level crops — no preprocessing.
    from PIL import Image

    embeddings_list = []
    metadatas_list  = []
    paths_list      = []
    ids_list        = []

    for idx, (img_path, defect_type, grid_cell) in enumerate(
        tqdm(image_entries, desc="Embedding")
    ):
        try:
            image     = Image.open(img_path).convert("RGB")
            embedding = embedder.encode_image(image)

            embeddings_list.append(embedding)
            metadatas_list.append(
                {
                    "defect_type":       defect_type,   # "corrosion" | "burned"
                    "grid_cell":         grid_cell,     # "A1" | "D4" | "B1" | "C4"
                    "original_filename": img_path.name,
                    "source_folder":     img_path.parent.name,
                    "db":                "second_vb",
                }
            )
            paths_list.append(str(img_path))
            ids_list.append(f"second_vb_{idx:05d}")

        except Exception as exc:
            logger.error(f"  Failed to embed {img_path}: {exc}")

    if not embeddings_list:
        logger.error("No embeddings generated — aborting.")
        return

    store.add_images(
        embeddings=np.array(embeddings_list),
        metadatas=metadatas_list,
        image_paths=paths_list,
        ids=ids_list,
    )

    logger.info("=" * 60)
    logger.info(f"second_vb built: {len(embeddings_list)} images stored")
    for dt in sorted(set(m["defect_type"] for m in metadatas_list)):
        cells = sorted(set(
            m["grid_cell"] for m in metadatas_list if m["defect_type"] == dt
        ))
        for cell in cells:
            count = sum(
                1 for m in metadatas_list
                if m["defect_type"] == dt and m["grid_cell"] == cell
            )
            logger.info(f"  {dt}/{cell}: {count} images")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build second_vb localization database")
    parser.add_argument("--reset", action="store_true", help="Reset DB before building")
    args = parser.parse_args()
    build_second_vb(reset=args.reset)
