#!/usr/bin/env python3
"""
Complete the BDRC ingest for the 44 missing Ganjuur volumes.

The original ingest (ingest_from_bdrc.py) ran out of disk space at volume 75,
leaving 44 of 108 volumes without crops. This script:

  1. Detects which volumes are missing (by scanning existing crops).
  2. Downloads IIIF pages only for those volumes from BDRC.
  3. Extracts sliding-window crops.
  4. Embeds with ConvNeXt-v2-base (1024-dim) — the new production model.
  5. Stores into the 'ganjuur_frames_v2' collection.
  6. Monitors disk space and pauses if free space drops below 5 GB.

Self-contained: does NOT import gpt.py (which still has DINOv2 + old pooling).

Usage:
    python scripts/ingest_missing_volumes.py --dry-run             # list missing
    python scripts/ingest_missing_volumes.py --ingest --test 2     # ingest 2 volumes
    python scripts/ingest_missing_volumes.py --ingest              # all missing
"""
import argparse
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
BASE = Path(os.getenv("GANJUUR_BASE_DIR", "/home/trinity/ganjuur")).expanduser()
CROPS_DIR = BASE / "data" / "ganjuur_crops" / "db_frames"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "ganjuur_frames_v2"
VECTOR_DIM = 1024
MODEL_ID = "facebook/convnextv2-base-22k-224"

LDSPDI = "https://ldspdi.bdrc.io/resource"
IIIF_SRV = "https://iiif.bdrc.io"
INSTANCE_ID = "bdr:MW4CZ5370"

MAX_IIIF_WIDTH = 2000
EMPTY_PAGE_LIMIT = 20
STRIDE = 50
BATCH_SIZE = 32
POLITENESS_DELAY = 0.2
MIN_FREE_GB = 5.0

USER_AGENT = "Ganjuur-BDRC-Ingester/1.0 (+https://library.bdrc.io/)"


# ──────────────────────────────────────────────────────────────────────
# 1. Detect missing volumes
# ──────────────────────────────────────────────────────────────────────
def existing_volumes() -> set[int]:
    """Return set of volume numbers that already have crops."""
    found = set()
    for f in CROPS_DIR.rglob("*"):
        m = re.search(r"vol(\d+)_p(\d+)", f.name)
        if m:
            found.add(int(m.group(1)))
    return found


# ──────────────────────────────────────────────────────────────────────
# 2. BDRC metadata + IIIF (same RDF pipeline as ingest_from_bdrc.py)
# ──────────────────────────────────────────────────────────────────────
def fetch_rdf(bdr_id: str) -> dict:
    short_id = bdr_id[4:] if bdr_id.startswith("bdr:") else bdr_id
    url = f"{LDSPDI}/{short_id}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/ld+json"}
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt < 3:
                time.sleep(2.0 * attempt)
            else:
                raise RuntimeError(f"RDF failed: {url} -> {exc}")


def instance_to_reproduction(instance_id: str) -> str:
    data = fetch_rdf(instance_id)
    graph = data.get("@graph") or [data]
    for node in graph:
        if node.get("@id") == instance_id:
            repro = node.get("instanceHasReproduction")
            if isinstance(repro, dict):
                return repro["@id"]
            if isinstance(repro, str):
                return repro
    raise RuntimeError(f"No reproduction for {instance_id}")


def get_volume_list(work_id: str) -> list[str]:
    data = fetch_rdf(work_id)
    graph = data.get("@graph") or [data]
    for node in graph:
        if node.get("@id") == work_id:
            vols = node.get("instanceHasVolume", [])
            return [v["@id"] for v in vols if isinstance(v, dict) and "@id" in v]
    raise RuntimeError(f"No volumes on {work_id}")


def get_volume_meta(imagegroup_id: str) -> dict:
    data = fetch_rdf(imagegroup_id)
    graph = data.get("@graph") or [data]
    for node in graph:
        if node.get("@id") == imagegroup_id:
            pt = node.get("volumePagesTotal", {})
            pi = node.get("volumePagesTbrcIntro", {})
            vn = node.get("volumeNumber", {})
            return {
                "total": int(pt.get("@value", 0)) if isinstance(pt, dict) else 0,
                "intro": int(pi.get("@value", 0)) if isinstance(pi, dict) else 0,
                "number": int(vn.get("@value", 0)) if isinstance(vn, dict) else 0,
            }
    return {}


def page_iiif_url(vol_local_id: str, seq: int) -> str:
    img_id = f"bdr:{vol_local_id}::{vol_local_id}{seq:04d}.jpg"
    return f"{IIIF_SRV}/{img_id}/full/!{MAX_IIIF_WIDTH},{MAX_IIIF_WIDTH}/0/default.jpg"


def download_page(vol_local_id: str, seq: int) -> np.ndarray | None:
    url = page_iiif_url(vol_local_id, seq)
    headers = {"User-Agent": USER_AGENT, "Accept": "image/jpeg,image/*"}
    try:
        r = requests.get(url, headers=headers, timeout=120, stream=True)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        raw = r.content
        if len(raw) < 100:
            return None
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except requests.HTTPError:
        return None


# ──────────────────────────────────────────────────────────────────────
# 3. Crop + embed (ConvNeXt-aware)
# ──────────────────────────────────────────────────────────────────────
def process_page(image: np.ndarray) -> np.ndarray:
    """green-channel CLAHE → gray → BGR (matches original pipeline)."""
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(green)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def extract_crops(image: np.ndarray, stride: int = STRIDE) -> list[np.ndarray]:
    """Sliding-window crops over the page."""
    processed = process_page(image)
    h, w = processed.shape[:2]
    window = min(h, w)
    axis_is_x = w >= h
    max_dim = w if axis_is_x else h
    crops = []
    for pos in range(0, max_dim - window + 1, stride):
        crop = processed[:, pos:pos + window] if axis_is_x else processed[pos:pos + window, :]
        crops.append(crop)
    return crops


def save_crop(crop: np.ndarray, page_id: str, frame_idx: int) -> tuple[str, str]:
    frame_id = str(uuid.uuid4())
    shard = CROPS_DIR / frame_id[:2] / frame_id[2:4]
    shard.mkdir(parents=True, exist_ok=True)
    fname = f"{frame_id}_p{page_id}_f{frame_idx:06d}.webp"
    full = shard / fname
    ok = cv2.imwrite(str(full), crop, [cv2.IMWRITE_WEBP_QUALITY, 85])
    if not ok:
        raise IOError(f"crop write failed: {full}")
    return str(full), frame_id


def embed_batch(model, processor, images: list[np.ndarray], device: str) -> np.ndarray:
    """ConvNeXt-aware embedding: 4D → global pool, 3D → CLS."""
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs).last_hidden_state
    if out.ndim == 4:
        vecs = out.mean(dim=[-2, -1])
    else:
        vecs = out[:, 0, :]
    vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
    return vecs.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 4. Qdrant
# ──────────────────────────────────────────────────────────────────────
def make_client():
    from qdrant_client import QdrantClient, models
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False,
                          check_compatibility=False)
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
        for field, schema in [("status", models.PayloadSchemaType.KEYWORD),
                              ("source", models.PayloadSchemaType.KEYWORD)]:
            qdrant.create_payload_index(COLLECTION, field, field_schema=schema)
    return qdrant


def disk_free_gb(path: Path = BASE) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--test", type=int, default=None,
                        help="Ingest only N missing volumes (for testing)")
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    # lazy import so --dry-run works without torch
    global torch
    import torch
    from transformers import AutoImageProcessor, AutoModel

    missing = sorted(set(range(1, 109)) - existing_volumes())
    print(f"Missing volumes ({len(missing)}): {missing}")
    if not args.ingest:
        return 0

    if args.test:
        missing = missing[:args.test]
        print(f"  test mode: {missing}")

    # BDRC metadata — build number → (local_id, meta) once
    print("\nResolving BDRC volumes …")
    work_id = instance_to_reproduction(INSTANCE_ID)
    volume_ids = get_volume_list(work_id)
    vol_by_num: dict[int, tuple[str, dict]] = {}
    for vid in volume_ids:
        try:
            m = get_volume_meta(vid)
            num = m.get("number", 0)
            vol_by_num[num] = (vid.replace("bdr:", ""), m)
        except Exception as exc:
            print(f"  meta failed for {vid}: {exc}")
    print(f"  resolved {len(vol_by_num)} volumes from BDRC")

    # model + qdrant
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} …")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    qdrant = make_client()

    total_indexed = 0
    for vol_num in missing:
        entry = vol_by_num.get(vol_num)
        if not entry:
            print(f"\nVolume {vol_num}: not found in BDRC, skipping")
            continue
        vol_local, meta = entry
        total_pages = meta.get("total", 0)
        intro = meta.get("intro", 0)
        print(f"\n{'='*50}\nVolume {vol_num} (~{total_pages - intro} content pages)")
        print(f"  disk free: {disk_free_gb():.1f} GB")

        if disk_free_gb() < MIN_FREE_GB:
            print(f"  !! LOW DISK ({disk_free_gb():.1f} GB) — pausing 60s …")
            time.sleep(60)

        indexed = 0
        empty_streak = 0
        page = intro + 1  # skip intro pages
        batch_crops, batch_paths, batch_ids = [], [], []

        while empty_streak < EMPTY_PAGE_LIMIT:
            img = download_page(vol_local, page)
            if img is None:
                empty_streak += 1
                page += 1
                continue
            empty_streak = 0
            processed = process_page(img)
            crops = extract_crops(processed, args.stride)
            for ci, crop in enumerate(crops):
                try:
                    fp, fid = save_crop(crop, f"bdrc:bdr:MW4CZ5370/vol{vol_num}/p{page}", ci)
                except IOError as exc:
                    print(f"  write error: {exc}")
                    continue
                batch_crops.append(cv2.imread(fp))
                batch_paths.append(fp)
                batch_ids.append(fid)
            page += 1
            time.sleep(POLITENESS_DELAY)

            if len(batch_crops) >= BATCH_SIZE:
                vecs = embed_batch(model, processor, batch_crops, device)
                points = [
                    models.PointStruct(
                        id=pid,
                        vector=v.tolist(),
                        payload={
                            "source": f"vol{vol_num}_p{page}",
                            "local_path": p,
                            "status": "EMBEDDED",
                            "embedding_model": MODEL_ID,
                            "ingested_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    for v, p, pid in zip(vecs, batch_paths, batch_ids)
                ]
                qdrant.upsert(COLLECTION, points=points, wait=True)
                indexed += len(points)
                total_indexed += len(points)
                batch_crops.clear()
                batch_paths.clear()
                batch_ids.clear()
                print(f"  vol{vol_num}: {indexed} crops indexed (total {total_indexed})")

        # flush remainder
        if batch_crops:
            vecs = embed_batch(model, processor, batch_crops, device)
            points = [
                models.PointStruct(
                    id=pid,
                    vector=v.tolist(),
                    payload={"source": f"vol{vol_num}", "local_path": p, "status": "EMBEDDED",
                             "embedding_model": MODEL_ID,
                             "ingested_at": datetime.now(timezone.utc).isoformat()},
                )
                for v, p, pid in zip(vecs, batch_paths, batch_ids)
            ]
            qdrant.upsert(COLLECTION, points=points, wait=True)
            indexed += len(points)
            total_indexed += len(points)
        print(f"  Volume {vol_num} done: {indexed} crops")

    print(f"\n{'='*50}\nALL DONE — {total_indexed} total crops indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
