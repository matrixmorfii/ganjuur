#!/usr/bin/env python3
"""
Visual-embedding model comparison for Ganjuur — page-level coherence.

Your crops are named like:
    {uuid}_pbdrc_bdr_MW4CZ5370_vol089_p0639_f000008.webp
                                      ^^^^  ^^^^
                                      vol   page

This script measures whether a model retrieves crops from the SAME PAGE as
the query — the actual job of the search engine.  Higher page-coherence =
better glyph grouping.

Default: DINOv2-base (current production) vs DINOv2-large (free, ungated).
DINOv3 variants work too but need HuggingFace login + transformers>=4.50.

Usage:
    python scripts/compare_dino_models.py                    # 300 crops, 30 queries
    python scripts/compare_dino_models.py --crops 1000 --queries 50
    python scripts/compare_dino_models.py --model-b facebook/dinov2-giant

    # DINOv3 (gated — needs: pip install -U transformers huggingface_hub,
    #                      huggingface-cli login, access request on HF):
    python scripts/compare_dino_models.py --model-b facebook/dinov3-vitb16-pretrain-lvd1689m
    python scripts/compare_dino_models.py --model-b facebook/dinov3-convnext-base-pretrain-lvd1689m
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# vol089_p0639  →  ("vol089", "p0639")
_PAGE_RE = re.compile(r"(vol\d+)_p(\d+)")
# _f000008  →  8
_FRAME_RE = re.compile(r"_f(\d+)")


def page_id_from_filename(path: Path) -> str | None:
    """Return 'volXXX_pXXXX' identity, or None if the name doesn't match."""
    m = _PAGE_RE.search(path.name)
    return f"{m.group(1)}_p{m.group(2)}" if m else None


def frame_index_from_filename(path: Path) -> int | None:
    """Return the frame index from '_fXXXXXX', or None."""
    m = _FRAME_RE.search(path.name)
    return int(m.group(1)) if m else None


def collect_crop_paths(root: Path, max_images: int, seed: int = 42) -> list[Path]:
    """Walk the sharded crop directory and return up to *max_images* webp paths."""
    rng = random.Random(seed)
    all_paths: list[Path] = []
    for shard_dir in sorted(root.iterdir()):
        if not shard_dir.is_dir():
            continue
        for sub in sorted(shard_dir.iterdir()):
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.suffix.lower() in (".webp", ".jpg", ".png"):
                    all_paths.append(f)
    rng.shuffle(all_paths)
    return sorted(all_paths[:max_images])


def collect_dense_from_one_page(root: Path, seed: int = 42) -> list[Path]:
    """
    Return ALL consecutive crops from the page that has the most crops.

    Consecutive frames (_f000100, _f000101, …) are near-duplicates because the
    sliding-window extractor overlaps them by ~75%.  A good model ranks them
    near each other, so this is a meaningful automatic test.
    """
    rng = random.Random(seed)
    by_page: dict[str, list[Path]] = {}
    for shard_dir in sorted(root.iterdir()):
        if not shard_dir.is_dir():
            continue
        for sub in sorted(shard_dir.iterdir()):
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.suffix.lower() not in (".webp", ".jpg", ".png"):
                    continue
                pid = page_id_from_filename(f)
                if pid:
                    by_page.setdefault(pid, []).append(f)
    if not by_page:
        return []
    best_page = max(by_page, key=lambda p: len(by_page[p]))
    paths = sorted(by_page[best_page])
    rng.shuffle(paths)
    return paths


def load_image(path: Path) -> Image.Image | None:
    """Load an image, returning None if the file is corrupt/unreadable."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_batch(model, processor, images: list[Image.Image], device: str) -> np.ndarray:
    """Return a (N, D) float32 matrix of L2-normalised embeddings.

    Handles three output formats:
      - ViT (DINOv2, DINOv3-ViT): last_hidden_state is (B, seq, D) → CLS token
      - ConvNeXt / ConvNeXt v2: last_hidden_state is (B, C, H, W) → global pool
    """
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    hs = outputs.last_hidden_state
    if hs.ndim == 4:
        # (B, C, H, W) → global average pool → (B, C)
        vecs = hs.mean(dim=[-2, -1])
    elif hs.ndim == 3:
        # (B, seq, D) → CLS token
        vecs = hs[:, 0, :]
    else:
        vecs = hs
    vecs = vecs.cpu().numpy().astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def cosine_search(query: np.ndarray, gallery: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the top_k gallery vectors most similar to query."""
    scores = gallery @ query
    return np.argsort(scores)[::-1][:top_k]


def near_duplicate_coherence(query_frame: int | None, result_indices: np.ndarray,
                            frame_indices: list[int | None], window: int = 3) -> float:
    """Fraction of top-K results whose frame index is within ±window of the query."""
    if query_frame is None:
        return 0.0
    near = sum(
        1 for idx in result_indices
        if frame_indices[idx] is not None and abs(frame_indices[idx] - query_frame) <= window
    )
    return near / len(result_indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-dir", type=Path, required=False,
                        help="Root of sharded crop dir  "
                             "(default: /home/trinity/ganjuur/data/ganjuur_crops/db_frames)")
    parser.add_argument("--crops", type=int, default=300,
                        help="Number of crop images to embed (default: 300)")
    parser.add_argument("--queries", type=int, default=30,
                        help="Number of query images to test (default: 30)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K for retrieval (default: 10)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Embedding batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-a", default="facebook/dinov2-base",
                        help="model A (default: facebook/dinov2-base)")
    parser.add_argument("--model-b", default="facebook/dinov2-large",
                        help="model B (default: facebook/dinov2-large)")
    parser.add_argument("--dense", action="store_true",
                        help="Sample densely from ONE page instead of randomly "
                             "across all pages.  Makes frame-proximity "
                             "coherence meaningful (near-duplicate crops).")
    parser.add_argument("--proximity", type=int, default=3,
                        help="Frame window for near-duplicate coherence "
                             "(default: ±3 frames)")
    args = parser.parse_args()

    # lazy import so --help works without torch installed
    global torch
    import torch

    crop_dir = args.crop_dir or Path("/home/trinity/ganjuur/data/ganjuur_crops/db_frames")
    if not crop_dir.exists():
        print(f"ERROR: crop directory not found: {crop_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Visual-embedding model comparison ===")
    print(f"  device        : {device}")
    print(f"  crop dir      : {crop_dir}")
    print(f"  num crops     : {args.crops}")
    print(f"  num queries   : {args.queries}")
    print(f"  top-K         : {args.top_k}")
    print(f"  dense sampling: {'yes (one page)' if args.dense else 'no (random across pages)'}")
    print()

    # 1. Collect images + frame ids
    print("[1/5] Collecting crop paths …")
    if args.dense:
        # Dense mode: ALL crops from the page with the most crops.
        # Queries are a RANDOM SUBSET of the gallery (so each query IS in the
        # gallery, enabling self-retrieval + frame-proximity tests).
        all_paths = collect_dense_from_one_page(crop_dir, seed=args.seed)
        if len(all_paths) < 20:
            print(f"  only found {len(all_paths)} images on the best page — need at least 20")
            return 1
        n_queries = max(5, len(all_paths) // 5)
        query_paths   = all_paths[:n_queries]
        gallery_paths = all_paths[:]                       # gallery includes the queries
        # record the index of each query inside the gallery for self-MRR
        query_gallery_indices = [all_paths.index(qp) for qp in query_paths]
    else:
        all_paths = collect_crop_paths(crop_dir, args.crops + args.queries, seed=args.seed)
        if len(all_paths) < args.queries + 10:
            print(f"  only found {len(all_paths)} images — need at least {args.queries + 10}")
            return 1
        gallery_paths = all_paths[:args.crops]
        query_paths   = all_paths[args.crops:args.crops + args.queries]
        query_gallery_indices = None

    gallery_frames = [frame_index_from_filename(p) for p in gallery_paths]
    query_frames   = [frame_index_from_filename(p) for p in query_paths]
    n_with_frame = sum(1 for f in gallery_frames if f is not None)
    print(f"  gallery: {len(gallery_paths)}   queries: {len(query_paths)}")
    print(f"  crops with parseable frame index: {n_with_frame}/{len(gallery_paths)}")

    # 2. Load images (skip corrupt files)
    print("[2/5] Loading images …")
    t0 = time.time()
    gallery_loaded = [load_image(p) for p in gallery_paths]
    query_loaded   = [load_image(p) for p in query_paths]
    # keep only successfully loaded images, and sync the frame-id lists
    gallery_pairs = [(img, frame) for img, frame in zip(gallery_loaded, gallery_frames) if img is not None]
    query_pairs   = [(img, frame) for img, frame in zip(query_loaded, query_frames) if img is not None]
    gallery_imgs   = [p[0] for p in gallery_pairs]
    gallery_frames = [p[1] for p in gallery_pairs]
    query_imgs     = [p[0] for p in query_pairs]
    query_frames   = [p[1] for p in query_pairs]
    print(f"  done in {time.time() - t0:.1f}s  ({len(gallery_loaded) - len(gallery_imgs)} corrupt files skipped)")
    if len(gallery_imgs) < 50 or len(query_imgs) < 10:
        print(f"ERROR: too few usable images (gallery={len(gallery_imgs)}, queries={len(query_imgs)})")
        return 1

    # 3. Embed with both models
    models_info = [
        ("model-a", args.model_a),
        ("model-b", args.model_b),
    ]
    # Friendly display names for the known defaults.
    name_map = {
        "facebook/dinov2-base":  "DINOv2-base",
        "facebook/dinov2-large": "DINOv2-large",
        "facebook/dinov2-giant": "DINOv2-giant",
        "facebook/convnextv2-base-22k-224":  "ConvNeXt-v2-base",
        "facebook/convnextv2-large-22k-224": "ConvNeXt-v2-large",
        "facebook/convnextv2-huge-22k-224":  "ConvNeXt-v2-huge",
        "facebook/dinov3-vitb16-pretrain-lvd1689m":  "DINOv3-ViT-base",
        "facebook/dinov3-vitl16-pretrain-lvd1689m":  "DINOv3-ViT-large",
        "facebook/dinov3-vith16plus-pretrain-lvd1689m": "DINOv3-ViT-huge",
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m":  "DINOv3-ConvNeXt-tiny",
        "facebook/dinov3-convnext-small-pretrain-lvd1689m": "DINOv3-ConvNeXt-small",
        "facebook/dinov3-convnext-base-pretrain-lvd1689m":  "DINOv3-ConvNeXt-base",
        "facebook/dinov3-convnext-large-pretrain-lvd1689m": "DINOv3-ConvNeXt-large",
    }

    embeddings: dict[str, np.ndarray] = {}
    display_names: dict[str, str] = {}
    for label, hub_id in models_info:
        display = name_map.get(hub_id, label)
        display_names[label] = display
        print(f"[3/5] Loading {display} ({hub_id}) …")
        t0 = time.time()
        processor = AutoImageProcessor.from_pretrained(hub_id)
        model     = AutoModel.from_pretrained(hub_id).to(device).eval()
        with torch.inference_mode():
            dummy = model(torch.randn(1, 3, 224, 224).to(device)).last_hidden_state
        if dummy.ndim == 4:
            dummy = dummy.mean(dim=[-2, -1])
        elif dummy.ndim == 3:
            dummy = dummy[:, 0, :]
        detected_dim = dummy.shape[1]
        print(f"  loaded in {time.time() - t0:.1f}s  dim={detected_dim}")

        print(f"  embedding gallery …")
        t0 = time.time()
        gallery_vecs: list[np.ndarray] = []
        for i in range(0, len(gallery_imgs), args.batch):
            batch = gallery_imgs[i:i + args.batch]
            gallery_vecs.append(embed_batch(model, processor, batch, device))
        gallery_mat = np.concatenate(gallery_vecs, axis=0)
        print(f"  gallery done in {time.time() - t0:.1f}s  shape={gallery_mat.shape}")

        print(f"  embedding queries …")
        t0 = time.time()
        query_vecs: list[np.ndarray] = []
        for i in range(0, len(query_imgs), args.batch):
            batch = query_imgs[i:i + args.batch]
            query_vecs.append(embed_batch(model, processor, batch, device))
        query_mat = np.concatenate(query_vecs, axis=0)
        print(f"  queries done in {time.time() - t0:.1f}s  shape={query_mat.shape}")

        embeddings[label] = {"gallery": gallery_mat, "query": query_mat}
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    # 4. Run retrieval
    print("[4/5] Running retrieval …")
    name_a = display_names["model-a"]
    name_b = display_names["model-b"]
    gal_a = embeddings["model-a"]["gallery"]
    gal_b = embeddings["model-b"]["gallery"]
    qry_a = embeddings["model-a"]["query"]
    qry_b = embeddings["model-b"]["query"]

    overlaps: list[float] = []
    prox_a: list[float] = []
    prox_b: list[float] = []
    mrr_a: list[float] = []
    mrr_b: list[float] = []

    for qi in range(len(query_imgs)):
        top_a = cosine_search(qry_a[qi], gal_a, args.top_k)
        top_b = cosine_search(qry_b[qi], gal_b, args.top_k)
        overlaps.append(len(set(top_a) & set(top_b)) / args.top_k)

        # frame-proximity coherence (meaningful in dense mode)
        prox_a.append(near_duplicate_coherence(query_frames[qi], top_a, gallery_frames, args.proximity))
        prox_b.append(near_duplicate_coherence(query_frames[qi], top_b, gallery_frames, args.proximity))

        # self-retrieval MRR (a crop should rank itself #1).
        # In dense mode the query sits at query_gallery_indices[qi]; in random
        # mode the query is NOT in the gallery so self-rank is reported as 0.
        if args.dense and query_gallery_indices is not None:
            true_idx = query_gallery_indices[qi]
            rank_a = np.where(top_a == true_idx)[0]
            rank_b = np.where(top_b == true_idx)[0]
            mrr_a.append(1.0 / (rank_a[0] + 1) if len(rank_a) else 0.0)
            mrr_b.append(1.0 / (rank_b[0] + 1) if len(rank_b) else 0.0)
        else:
            mrr_a.append(0.0)
            mrr_b.append(0.0)

    # 5. Report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  mean top-{args.top_k} overlap              : {np.mean(overlaps):.3f}")
    print()
    print(f"  FRAME-PROXIMITY COHERENCE (top-{args.top_k} within ±{args.proximity} frames):")
    print(f"    {name_a:22s}  {np.mean(prox_a):.3f}")
    print(f"    {name_b:22s}  {np.mean(prox_b):.3f}")
    print()
    print(f"  SELF-RETRIEVAL MRR (crop retrieves itself as #1):")
    print(f"    {name_a:22s}  {np.mean(mrr_a):.4f}")
    print(f"    {name_b:22s}  {np.mean(mrr_b):.4f}")
    print()

    # Winner by self-MRR (the most reliable automatic metric)
    if np.mean(mrr_b) > np.mean(mrr_a) + 0.01:
        print(f"  >> WINNER: {name_b}  (higher self-retrieval MRR)")
    elif np.mean(mrr_a) > np.mean(mrr_b) + 0.01:
        print(f"  >> WINNER: {name_a}  (higher self-retrieval MRR)")
    else:
        print(f"  >> TIE — both models retrieve themselves equally well")
    print()
    if args.dense:
        print("Note: dense mode samples consecutive crops from one page.")
        print("Frame-proximity measures whether the model groups near-duplicate")
        print("patches together.  Higher = better local coherence.")
    else:
        print("Note: random mode samples across all pages.  Frame-proximity is")
        print("near 0 (expected — random crops aren't near-duplicates).  Use")
        print("--dense for a meaningful frame-proximity number.")
    print("Self-MRR is reliable in BOTH modes: a good model ranks each crop")
    print("as most similar to itself.")


if __name__ == "__main__":
    sys.exit(main())
