#!/usr/bin/env python3
"""
Migrate Ganjuur from DINOv2-base (768-dim) to ConvNeXt-v2-base (1024-dim).

DEFAULT mode  ── re-embed exactly the crops that are ALREADY in the
                  'ganjuur_frames' collection (reads each point's local_path,
                  embeds with ConvNeXt-v2-base, writes to 'ganjuur_frames_v2').
                  Same scope, same payloads, ~113K vectors, ~10 minutes.

--full mode   ── walk the ENTIRE crop directory (~1.44M files) and embed
                  everything.  Hours of work, but catches crops the original
                  ingest may have skipped.

The original 'ganjuur_frames' collection is NEVER touched.

Usage:
    python scripts/migrate_to_convnextv2.py                # default: match existing
    python scripts/migrate_to_convnextv2.py --full         # embed all 1.44M crops
    python scripts/migrate_to_convnextv2.py --dry-run      # count only
    python scripts/migrate_to_convnextv2.py --max 1000     # test on first 1000

After migration, swap in config.py / .env:
    COLLECTION_NAME = "ganjuur_frames_v2"
    VECTOR_DIM      = 1024
    DINO_MODEL      = "facebook/convnextv2-base-22k-224"
"""

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Configuration (matches the production app's Qdrant connection)
# ---------------------------------------------------------------------------
QDRANT_HOST      = "localhost"
QDRANT_PORT      = 6333
QDRANT_GRPC_PORT = 6334

CROPS_DIR = Path("/home/trinity/ganjuur/data/ganjuur_crops/db_frames")

SRC_COLLECTION = "ganjuur_frames"
NEW_COLLECTION = "ganjuur_frames_v2"
VECTOR_DIM     = 1024
MODEL_ID       = "facebook/convnextv2-base-22k-224"
BATCH_SIZE     = 32

DEVICE = "cuda"  # set to "cpu" if no GPU


# ---------------------------------------------------------------------------
def scroll_existing_paths(qdrant: QdrantClient, limit: int | None) -> list[str]:
    """Return local_path of every point in the source collection."""
    paths: list[str] = []
    offset = None
    while True:
        records, offset = qdrant.scroll(
            collection_name=SRC_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["local_path"],
            with_vectors=False,
        )
        for r in records:
            p = (r.payload or {}).get("local_path")
            if p:
                paths.append(p)
            if limit and len(paths) >= limit:
                return paths[:limit]
        if offset is None or not records:
            break
    return paths


def collect_crop_paths(root: Path, limit: int | None) -> list[Path]:
    """Walk the full crop directory."""
    paths: list[Path] = []
    for shard_dir in sorted(root.iterdir()):
        if not shard_dir.is_dir():
            continue
        for sub in sorted(shard_dir.iterdir()):
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.suffix.lower() in (".webp", ".jpg", ".png"):
                    paths.append(f)
                if limit and len(paths) >= limit:
                    return paths
    return paths


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_batch(model, processor, images: list[Image.Image]) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = model(**inputs)
    hs = out.last_hidden_state
    if hs.ndim == 4:    # ConvNeXt (B, C, H, W)
        vecs = hs.mean(dim=[-2, -1])
    elif hs.ndim == 3:  # ViT (B, seq, D)
        vecs = hs[:, 0, :]
    else:
        vecs = hs
    vecs = vecs.cpu().numpy().astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Embed ALL crops in the directory, not just existing vectors")
    parser.add_argument("--max", type=int, default=None,
                        help="Limit the number of crops (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count, don't embed")
    parser.add_argument("--collection", default=NEW_COLLECTION)
    args = parser.parse_args()

    global torch
    import torch

    # 1. Resolve the crop list
    print("[1/4] Resolving crop list …")
    if args.full:
        source = "full directory walk"
        paths = collect_crop_paths(CROPS_DIR, args.max)
    else:
        source = f"scrolling existing '{SRC_COLLECTION}'"
        print(f"  connecting to Qdrant to read existing points …")
        qdrant_tmp = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False)
        paths = [Path(p) for p in scroll_existing_paths(qdrant_tmp, args.max)]
        del qdrant_tmp
    print(f"  source: {source}")
    print(f"  crops to embed: {len(paths)}")
    if args.dry_run:
        return 0
    if not paths:
        print("ERROR: no crops found")
        return 1

    # 2. Load model
    print(f"[2/4] Loading {MODEL_ID} on {DEVICE} …")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # 3. Create Qdrant collection
    print(f"[3/4] Creating '{args.collection}' …")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT,
                          grpc_port=QDRANT_GRPC_PORT, prefer_grpc=False,
                          check_compatibility=False)
    if qdrant.collection_exists(args.collection):
        qdrant.delete_collection(args.collection)
        print(f"  deleted existing '{args.collection}'")
    qdrant.create_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    for field, schema in [
        ("umap_x", models.PayloadSchemaType.FLOAT),
        ("anomaly_score", models.PayloadSchemaType.FLOAT),
        ("status", models.PayloadSchemaType.KEYWORD),
        ("source", models.PayloadSchemaType.KEYWORD),
    ]:
        qdrant.create_payload_index(args.collection, field, field_schema=schema)
    print(f"  created ({VECTOR_DIM}-dim COSINE)")

    # 4. Embed + flush
    print(f"[4/4] Embedding and flushing (batch={BATCH_SIZE}) …")
    flushed = 0
    corrupt = 0
    missing = 0
    batch_imgs: list[Image.Image] = []
    batch_paths: list[Path] = []
    t_start = time.time()

    def flush():
        nonlocal flushed, batch_imgs, batch_paths
        if not batch_imgs:
            return
        vecs = embed_batch(model, processor, batch_imgs)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "source": path.name,
                    "local_path": str(path),
                    "status": "EMBEDDED",
                    "embedding_model": MODEL_ID,
                    "preprocessing": "convnextv2-default",
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "volume": _vol(path.name),
                    "frame": _frame(path.name),
                },
            )
            for vec, path in zip(vecs, batch_paths)
        ]
        qdrant.upsert(args.collection, points=points, wait=True)
        flushed += len(points)
        batch_imgs.clear()
        batch_paths.clear()
        elapsed = time.time() - t_start
        rate = flushed / elapsed if elapsed else 0
        print(f"  {flushed:>7d} vectors  ({rate:.0f} vec/s, {elapsed:.0f}s)")

    for path in paths:
        if not path.exists():
            missing += 1
            continue
        img = load_image(path)
        if img is None:
            corrupt += 1
            continue
        batch_imgs.append(img)
        batch_paths.append(path)
        if len(batch_imgs) >= BATCH_SIZE:
            flush()
    flush()

    elapsed = time.time() - t_start
    info = qdrant.get_collection(args.collection)
    print()
    print("=" * 50)
    print("MIGRATION COMPLETE")
    print(f"  collection   : {args.collection}")
    print(f"  dim          : {VECTOR_DIM}")
    print(f"  model        : {MODEL_ID}")
    print(f"  vectors      : {info.points_count}")
    print(f"  corrupt skip : {corrupt}")
    print(f"  missing skip : {missing}")
    print(f"  time         : {elapsed:.0f}s")
    print()
    print("Next step — edit config.py (or .env):")
    print(f"  COLLECTION_NAME = \"{args.collection}\"")
    print(f"  VECTOR_DIM      = {VECTOR_DIM}")
    print(f"  DINO_MODEL      = \"{MODEL_ID}\"")
    return 0


def _vol(name: str) -> str:
    import re
    m = re.search(r"(vol\d+)", name)
    return m.group(1) if m else ""


def _frame(name: str) -> int:
    import re
    m = re.search(r"_f(\d+)", name)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    sys.exit(main())
