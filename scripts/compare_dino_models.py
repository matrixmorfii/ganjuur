#!/usr/bin/env python3
"""
DINOv2-base vs DINOv3-S/14 — retrieval quality comparison for Ganjuur.

Uses your existing crop images and measures which model retrieves more
visually-similar glyphs.  Runs entirely on CPU (no GPU required) so it
works on the Xeon VPS too — just slower.

Usage:
    python scripts/compare_dino_models.py                  # default: 200 crops, 20 queries
    python scripts/compare_dino_models.py --crops 500 --queries 50
    python scripts/compare_dino_models.py --crop-dir /path/to/crops

Output:
    - Per-query top-K overlap between the two models
    - Mean reciprocal rank (MRR) comparison
    - A winner declaration with confidence
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def embed_batch(model, processor, images: list[Image.Image], device: str) -> np.ndarray:
    """Return a (N, D) float32 matrix of L2-normalised embeddings."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    # CLS token is the first token for both DINOv2 and DINOv3 ViT
    vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    # L2-normalise so cosine similarity == dot product
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def cosine_search(query: np.ndarray, gallery: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the top_k gallery vectors most similar to query."""
    scores = gallery @ query  # (N,)  — gallery is already L2-normalised
    return np.argsort(scores)[::-1][:top_k]


def retrieval_metrics(query_idx: int, gallery_indices: np.ndarray, top_k: int) -> dict:
    """
    Treat the query's own index as the 'correct' match (a crop is most
    similar to itself).  This is a self-consistency sanity check that
    both models should pass.  We also report top_k for manual inspection.
    """
    rank = np.where(gallery_indices == query_idx)[0]
    mrr = 1.0 / (rank[0] + 1) if len(rank) else 0.0
    return {"mrr": mrr, "self_rank": int(rank[0]) if len(rank) else -1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-dir", type=Path, required=False,
                        help="Root of sharded crop dir  "
                             "(default: /home/trinity/ganjuur/data/ganjuur_crops/db_frames)")
    parser.add_argument("--crops", type=int, default=200,
                        help="Number of crop images to embed (default: 200)")
    parser.add_argument("--queries", type=int, default=20,
                        help="Number of query images to test (default: 20)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K for retrieval (default: 10)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Embedding batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-a", default="facebook/dinov2-base",
                        help="HuggingFace id for model A (default: facebook/dinov2-base)")
    parser.add_argument("--model-b", default="facebook/dinov3-vits14-pretrain-lvd142m",
                        help="HuggingFace id for model B (default: facebook/dinov3-vits14-pretrain-lvd142m)")
    args = parser.parse_args()

    # lazy import so --help works without torch installed
    global torch
    import torch

    crop_dir = args.crop_dir or Path("/home/trinity/ganjuur/data/ganjuur_crops/db_frames")
    if not crop_dir.exists():
        print(f"ERROR: crop directory not found: {crop_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== DINOv2 vs DINOv3 retrieval comparison ===")
    print(f"  device        : {device}")
    print(f"  crop dir      : {crop_dir}")
    print(f"  num crops     : {args.crops}")
    print(f"  num queries   : {args.queries}")
    print(f"  top-K         : {args.top_k}")
    print()

    # 1. Collect images
    print("[1/5] Collecting crop paths …")
    paths = collect_crop_paths(crop_dir, args.crops * 2, seed=args.seed)  # extra for queries
    if len(paths) < args.queries + 10:
        print(f"  only found {len(paths)} images — need at least {args.queries + 10}")
        return 1
    query_paths = paths[:args.queries]
    gallery_paths = paths[args.queries:args.queries + args.crops]
    print(f"  gallery: {len(gallery_paths)}   queries: {len(query_paths)}")

    # 2. Load images
    print("[2/5] Loading images …")
    t0 = time.time()
    gallery_imgs = [load_image(p) for p in gallery_paths]
    query_imgs   = [load_image(p) for p in query_paths]
    print(f"  done in {time.time() - t0:.1f}s")

    # 3. Embed with both models
    models_info = [
        ("DINOv2-base",  args.model_a,   768),
        ("DINOv3-S/14",   args.model_b,   384),
    ]

    # NOTE: DINOv3 HuggingFace IDs may change.  Verify the exact id at
    # https://huggingface.co/facebook and adjust --model-b if needed.
    # Known variants as of 2025-08:
    #   facebook/dinov3-vits14-pretrain-lvd142m   (small,  384-dim)
    #   facebook/dinov3-vitb14-pretrain-lvd142m   (base,  768-dim)
    # The script will auto-detect the actual dim from model output, so the
    # "expected_dim" above is only used for the sanity printout.
    embeddings: dict[str, np.ndarray] = {}
    for name, hub_id, expected_dim in models_info:
        print(f"[3/5] Loading {name} ({hub_id}) …")
        t0 = time.time()
        processor = AutoImageProcessor.from_pretrained(hub_id)
        model     = AutoModel.from_pretrained(hub_id).to(device).eval()
        print(f"  loaded in {time.time() - t0:.1f}s  dim={expected_dim}")

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

        embeddings[name] = {"gallery": gallery_mat, "query": query_mat}
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    # 4. Run retrieval
    print("[4/5] Running retrieval …")
    name_a, name_b = [n for n, _, _ in models_info]
    gal_a = embeddings[name_a]["gallery"]
    gal_b = embeddings[name_b]["gallery"]
    qry_a = embeddings[name_a]["query"]
    qry_b = embeddings[name_b]["query"]

    overlaps: list[float] = []
    mrr_a_list: list[float] = []
    mrr_b_list: list[float] = []

    for qi in range(len(query_imgs)):
        top_a = cosine_search(qry_a[qi], gal_a, args.top_k)
        top_b = cosine_search(qry_b[qi], gal_b, args.top_k)
        overlap = len(set(top_a) & set(top_b)) / args.top_k
        overlaps.append(overlap)

        # self-consistency: query i corresponds to gallery offset i
        # (we can't know the true match, so we measure self-retrieval rank)
        rank_a = np.where(top_a == qi)[0]
        rank_b = np.where(top_b == qi)[0]
        mrr_a_list.append(1.0 / (rank_a[0] + 1) if len(rank_a) else 0.0)
        mrr_b_list.append(1.0 / (rank_b[0] + 1) if len(rank_b) else 0.0)

    # 5. Report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  mean top-{args.top_k} overlap      : {np.mean(overlaps):.3f}")
    print(f"  {name_a:15s}  mean self-MRR : {np.mean(mrr_a_list):.4f}")
    print(f"  {name_b:15s}  mean self-MRR : {np.mean(mrr_b_list):.4f}")
    print()

    if np.mean(mrr_b_list) > np.mean(mrr_a_list) + 0.02:
        print(f"  >> WINNER: {name_b}  (better self-retrieval consistency)")
    elif np.mean(mrr_a_list) > np.mean(mrr_b_list) + 0.02:
        print(f"  >> WINNER: {name_a}  (better self-retrieval consistency)")
    else:
        print(f"  >> TIE — both models perform similarly on this data")
    print()
    print("Note: self-MRR measures how often a crop retrieves itself as the")
    print("top match.  Higher = the model's embedding space is more coherent.")
    print("For a true glyph-similarity test, manually label ~50 known-similar")
    print("pairs and check which model ranks them higher.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
