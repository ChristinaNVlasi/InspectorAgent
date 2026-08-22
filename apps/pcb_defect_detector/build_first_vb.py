"""
Build FIRST vector DB (first_vb) for whole-PCB classification.

Source images: /PCB/borned/ + /PCB/corrosion/  (whole-PCB images)
Purpose      : classify an incoming PCB as "burned", "corrosion", or "OK"
Logic        : top-1 similarity search → if similarity < 0.80 → OK

Run:
    cd pcb_defect_detector
    python build_first_vb.py [--reset]
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
from preprocessing.pcb_extractor import PCBExtractor
from config import (
    PCB_CLASSIFICATION_DIR,
    MODEL_CONFIG,
    FIRST_VDB_CONFIG,
    PCB_CONFIG,
    PREPROCESSING_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Folder-name → defect_type label mapping
FOLDER_LABELS = {
    "borned": "burned",
    "burned": "burned",
    "corrosion": "corrosion",
}


def build_first_vb(reset: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("Building FIRST_VB — whole-PCB classification database")
    logger.info(f"Source : {PCB_CLASSIFICATION_DIR}")
    logger.info(f"Target : {FIRST_VDB_CONFIG['persist_directory']}")
    logger.info("=" * 60)

    # ── initialise components ─────────────────────────────────────────────
    embedder = CLIPEmbedder(
        model_name=MODEL_CONFIG["clip_model"],
        device=MODEL_CONFIG["device"],
    )

    store = PCBVectorStore(
        persist_directory=FIRST_VDB_CONFIG["persist_directory"],
        collection_name=FIRST_VDB_CONFIG["collection_name"],
    )

    if reset:
        logger.warning("Resetting first_vb …")
        store.reset()

    extractor = PCBExtractor(
        min_area_ratio=PREPROCESSING_CONFIG["min_area_ratio"],
        padding=PREPROCESSING_CONFIG["padding"],
    )

    # ── collect images ────────────────────────────────────────────────────
    extensions = set(PCB_CONFIG["image_extensions"])
    image_entries: list[tuple[Path, str]] = []  # (path, defect_type)

    for folder in PCB_CLASSIFICATION_DIR.iterdir():
        if not folder.is_dir():
            continue
        label = FOLDER_LABELS.get(folder.name.lower())
        if label is None:
            logger.warning(f"Unknown folder '{folder.name}', skipping.")
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in extensions:
                image_entries.append((img_path, label))

    if not image_entries:
        logger.error(f"No images found under {PCB_CLASSIFICATION_DIR}")
        return

    logger.info(f"Found {len(image_entries)} images to embed.")

    # ── embed & store ─────────────────────────────────────────────────────
    embeddings_list = []
    metadatas_list = []
    paths_list = []
    ids_list = []

    from PIL import Image

    for idx, (img_path, defect_type) in enumerate(tqdm(image_entries, desc="Embedding")):
        try:
            image = Image.open(img_path).convert("RGB")

            # preprocess: background removal + crop
            processed, crop_info = extractor.extract_pcb(
                image, method=PREPROCESSING_CONFIG["method"]
            )
            if not crop_info.get("detected", False):
                processed = image  # fallback to original

            embedding = embedder.encode_image(processed)

            embeddings_list.append(embedding)
            metadatas_list.append(
                {
                    "defect_type": defect_type,
                    "original_filename": img_path.name,
                    "source_folder": img_path.parent.name,
                    "db": "first_vb",
                }
            )
            paths_list.append(str(img_path))
            ids_list.append(f"first_vb_{idx:05d}")

        except Exception as exc:
            logger.error(f"Failed to process {img_path}: {exc}")

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
    logger.info(f"first_vb built: {len(embeddings_list)} images stored")
    for lbl in set(m["defect_type"] for m in metadatas_list):
        count = sum(1 for m in metadatas_list if m["defect_type"] == lbl)
        logger.info(f"  {lbl}: {count} images")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build first_vb classification database")
    parser.add_argument("--reset", action="store_true", help="Reset DB before building")
    args = parser.parse_args()
    build_first_vb(reset=args.reset)
