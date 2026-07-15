#!/usr/bin/env python3
"""
BDRC (Buddhist Digital Resource Center) IIIF Ingester for Ganjuur.

Usage:
    python ingest_from_bdrc.py                           # dry-run: list volumes
    python ingest_from_bdrc.py --ingest --max-pages 5    # ingest 5 pages for test
    python ingest_from_bdrc.py --ingest --stride 50      # full ingest

BDRC IIIF / RDF endpoints used:
  Resource RDF   : https://ldspdi.bdrc.io/resource/{bdr:ID}
  Reproduction   : bdr:W4CZ5370 -> bdr:MW4CZ5370 (instanceHasReproduction)
  Volume list    : W4CZ5370 -> instanceHasVolume -> [bdr:I...]
  Volume meta    : bdr:I... -> volumePagesTotal, volumePagesTbrcIntro
  IIIF Image API : https://iiif.bdrc.io/bdr:{vol}::{vol}{seq:04d}.jpg/full/!{w},{w}/0/default.jpg
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

# Reuse the Ganjuur modules so we share the exact same pipeline.
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import gpt as G

# ---------------------------------------------------------------------------
# BDRC API configuration
# ---------------------------------------------------------------------------
LDSPDI = "https://ldspdi.bdrc.io/resource"
IIIF_SRV = "https://iiif.bdrc.io"

# Lokesh Chandra Collection (Mongolian Kanjur)
INSTANCE_ID = "bdr:MW4CZ5370"

# IIIF downloads capped at this width (native is ~5000px; 2000px preserves
# manuscript detail while keeping bandwidth reasonable). The cinnabar+CLAHE
# pipeline + 768-D DINOv2 embedding works at any of these scales.
MAX_IIIF_WIDTH = 2000

# How many consecutive 404s before we assume we have passed the last page.
EMPTY_PAGE_LIMIT = 20

# Be kind to the BDRC servers — small delay between page fetches.
POLITENESS_DELAY_SEC = 0.2

# HTTP retries for the RDF metadata lookups.
RDF_RETRY_LIMIT = 3
RDF_RETRY_DELAY = 2.0
RDF_TIMEOUT = 30

USER_AGENT = "Ganjuur-BDRC-Ingester/1.0 (+https://library.bdrc.io/)"


# ---------------------------------------------------------------------------
# RDF / metadata helpers
# ---------------------------------------------------------------------------
def fetch_rdf(bdr_id: str) -> dict:
    """Fetch JSON-LD from BDRC LDSPDI, with retry & browser-like headers."""
    short_id = bdr_id if not bdr_id.startswith("bdr:") else bdr_id[4:]
    url = f"{LDSPDI}/{short_id}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/ld+json",
    }
    last_err = None
    for attempt in range(1, RDF_RETRY_LIMIT + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=RDF_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < RDF_RETRY_LIMIT:
                time.sleep(RDF_RETRY_DELAY * attempt)
    raise RuntimeError(f"BDRC RDF lookup failed after {RDF_RETRY_LIMIT} tries: {url} -> {last_err}")


def instance_to_reproduction(instance_id: str) -> str:
    """Given bdr:MW... (Instance), return the bdr:W... (Work/reproduction)."""
    data = fetch_rdf(instance_id)
    graph = data.get("@graph") or [data]
    for node in graph:
        if node.get("@id") == instance_id:
            repro = node.get("instanceHasReproduction")
            if isinstance(repro, dict):
                return repro["@id"]
            if isinstance(repro, str):
                return repro
    # Fallback: some BDRC records expose the work ID directly on the root node.
    repro = data.get("instanceHasReproduction", {})
    if isinstance(repro, dict) and "@id" in repro:
        return repro["@id"]
    raise RuntimeError(f"No instanceHasReproduction found for {instance_id}")


def get_volume_list(work_id: str) -> list[str]:
    """Return the list of bdr:I... (ImageGroup) IDs from a work."""
    data = fetch_rdf(work_id)
    graph = data.get("@graph") or [data]
    for node in graph:
        if node.get("@id") == work_id:
            vols = node.get("instanceHasVolume", [])
            ids = []
            for v in vols:
                if isinstance(v, dict) and "@id" in v:
                    ids.append(v["@id"])
            return ids
    raise RuntimeError(f"No instanceHasVolume list found on {work_id}")


def get_volume_meta(imagegroup_id: str) -> dict:
    """Fetch total pages, intro pages, volume number for an ImageGroup."""
    data = fetch_rdf(imagegroup_id)
    graph = data.get("@graph") or [data]
    meta = {}
    for node in graph:
        if node.get("@id") == imagegroup_id:
            pt = node.get("volumePagesTotal", {})
            pi = node.get("volumePagesTbrcIntro", {})
            vn = node.get("volumeNumber", {})
            meta["total"] = int(pt.get("@value", 0)) if isinstance(pt, dict) else 0
            meta["intro"] = int(pi.get("@value", 0)) if isinstance(pi, dict) else 0
            meta["number"] = int(vn.get("@value", 0)) if isinstance(vn, dict) else 0
            break
    return meta


# ---------------------------------------------------------------------------
# IIIF image download
# --------------------------------------------------------------------------=
def page_iiif_url(volume_local_id: str, seq: int, max_width: int = MAX_IIIF_WIDTH) -> str:
    """
    Build a IIIF Image API URL for a single page.

    volume_local_id : str   e.g. "I1PD110269"  (strip the "bdr:" prefix)
    seq             : int   1-based page sequence number
    """
    img_id = f"bdr:{volume_local_id}::{volume_local_id}{seq:04d}.jpg"
    return f"{IIIF_SRV}/{img_id}/full/!{max_width},{max_width}/0/default.jpg"


def download_page(volume_local_id: str, seq: int, max_width: int = MAX_IIIF_WIDTH) -> np.ndarray | None:
    """
    Download a single IIIF page as an OpenCV BGR array.
    Returns None when the page is empty / not present.
    """
    url = page_iiif_url(volume_local_id, seq, max_width=max_width)
    headers = {"User-Agent": USER_AGENT, "Accept": "image/jpeg,image/*"}
    try:
        resp = requests.get(url, headers=headers, timeout=120, stream=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = resp.content
        # It's truly empty or a tiny placeholder (intro / title pages).
        if len(raw) < 100:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except requests.HTTPError:
        return None


# ---------------------------------------------------------------------------
# Main ingestion driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest BDRC open-access manuscript images into Ganjuur."
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Actually download + embed pages. Without this, dry-run only.",
    )
    parser.add_argument(
        "--resource", default=INSTANCE_ID,
        help=f"BDRC Instance URN (default: {INSTANCE_ID})",
    )
    parser.add_argument(
        "--stride", type=int, default=50,
        help="Sliding-window stride in pixels (default: 50)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Stop after N content pages total (for testing)",
    )
    parser.add_argument(
        "--max-volumes", type=int, default=None,
        help="Only ingest this many volumes (default: all)",
    )
    parser.add_argument(
        "--volumes", type=str, default=None,
        help="Comma-separated list of volume numbers to ingest, e.g. 1,2,3",
    )
    parser.add_argument(
        "--iiif-width", type=int, default=MAX_IIIF_WIDTH,
        help=f"Max download width in px (default: {MAX_IIIF_WIDTH})",
    )
    args = parser.parse_args()

    # Use user-specified IIIF width (default MAX_IIIF_WIDTH set above).
    max_iiif_width = args.iiif_width

    resource_id = args.resource.strip()

    # ---- Resolve the work and list all volumes ----
    print(f"Resolving resource: {resource_id}")
    work_id = instance_to_reproduction(resource_id)
    print(f"  Work/reproduction: {work_id}")

    volume_ids = get_volume_list(work_id)
    print(f"  Total volumes: {len(volume_ids)}")

    # ---- Fetch per-volume metadata ----
    volumes: list[dict] = []
    for i, vid in enumerate(volume_ids):
        vid_short = vid.replace("bdr:", "")
        try:
            meta = get_volume_meta(vid)
        except Exception as exc:
            print(f"  [{i+1}/{len(volume_ids)}] {vid_short}: meta lookup failed ({exc}), skipping")
            continue
        volumes.append({
            "id": vid,
            "id_short": vid_short,
            "volume_number": meta.get("number", i + 1),
            "total_pages": meta.get("total", 0),
            "intro_pages": meta.get("intro", 0),
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(volume_ids):
            print(f"  Fetched metadata: {i+1}/{len(volume_ids)} volumes")

    # ---- Apply filters ----
    if args.volumes:
        wanted = {int(v.strip()) for v in args.volumes.split(",") if v.strip().isdigit()}
        volumes = [v for v in volumes if v["volume_number"] in wanted]
        print(f"  Filtered to volumes: {sorted(v['volume_number'] for v in volumes)}")

    if args.max_volumes:
        volumes = volumes[: args.max_volumes]
        print(f"  Limited to first {args.max_volumes} volume(s)")

    total_potential = sum(v["total_pages"] - v["intro_pages"] for v in volumes)
    print(f"\nContent pages available (after intro skip): {total_potential:,}")
    print(f"Max IIIF width: {max_iiif_width}px  |  stride: {args.stride}px")

    # ---- Dry run? ----
    if not getattr(args, "ingest", False):
        print("\n--- DRY RUN (pass --ingest to actually download + embed) ---")
        for v in volumes:
            print(f"  Volume {v['volume_number']:>3} [{v['id_short']}]: "
                  f"{v['total_pages']:>4} total, {v['intro_pages']:>2} intro")
        return

    # ---- Ingest ----
    client = G.make_qdrant_client(require_docker=True)
    G.qdrant = client  # flush_batch uses the module-global qdrant
    G.ensure_collection(client)
    print(f"Qdrant connected | backend={G.qdrant_backend}")
    print(f"Collection: {G.COLLECTION_NAME}\n")

    total_frames = 0
    total_indexed = 0
    content_pages_done = 0
    skipped_pages: list[str] = []

    for vi, vol in enumerate(volumes):
        vol_num = vol["volume_number"]
        vid_short = vol["id_short"]
        intro = vol["intro_pages"]
        last_seq = vol["total_pages"]  # RDF total is the last real seq number

        # Page id embeds resource + volume for traceability + BDRC attribution.
        page_prefix = f"bdrc:{resource_id}/vol{vol_num:03d}"
        # BDRC viewer URL pattern (used as source attribution in the payload).
        bdrc_viewer = f"https://library.bdrc.io/show/{resource_id}?uilang=en"

        # Extra payload merged into every Qdrant point for BDRC attribution.
        bdrc_extra = {
            "bdrc_url": bdrc_viewer,
            "bdrc_resource_id": resource_id,
            "bdrc_volume": vol_num,
            "bdrc_volume_id": vid_short,
            "bdrc_access": "BDRC Open Access + Lokesh Chandra Collection (permission granted)",
        }

        print(f"\n[{vi+1}/{len(volumes)}] Volume {vol_num:>3} [{vid_short}] "
              f"— pages {intro+1}..{last_seq} (skip first {intro} intro pages)")

        empty_streak = 0
        page_count = 0

        # Iterate past intro pages to the last real page (per RDF page count).
        for seq in range(1, last_seq + 1):
            if seq <= intro:
                continue

            if args.max_pages and content_pages_done >= args.max_pages:
                print(f"  Reached --max-pages limit ({args.max_pages}).")
                break

            sys.stdout.write(f"\r  page {seq:>4}/{last_seq:<4} ({content_pages_done+1} done, {empty_streak} empty streak)")
            sys.stdout.flush()

            # Politeness delay between fetches.
            if POLITENESS_DELAY_SEC > 0:
                time.sleep(POLITENESS_DELAY_SEC)

            try:
                image_bgr = download_page(vid_short, seq, max_width=max_iiif_width)
            except Exception as exc:
                skipped_pages.append(f"{page_prefix}/p{seq:04d} (download: {exc})")
                empty_streak += 1
                if empty_streak >= EMPTY_PAGE_LIMIT:
                    print(f"\n  Stopping volume: {EMPTY_PAGE_LIMIT} consecutive empty/error pages.")
                    break
                continue

            if image_bgr is None:
                empty_streak += 1
                skipped_pages.append(f"{page_prefix}/p{seq:04d} (empty)")
                if empty_streak >= EMPTY_PAGE_LIMIT:
                    print(f"\n  Stopping volume: {EMPTY_PAGE_LIMIT} consecutive empty/error pages.")
                    break
                continue

            empty_streak = 0

            try:
                page_id = f"{page_prefix}/p{seq:04d}"
                processed = G.process_manuscript_image(image_bgr)
                h, w = processed.shape[:2]
                window = min(h, w)
                axis_is_x = w >= h
                max_dim = w if axis_is_x else h

                crops: list[np.ndarray] = []
                fps: list[str] = []
                ids: list[str] = []
                page_frames = 0

                for pos in range(0, max_dim - window + 1, args.stride):
                    crop = processed[:, pos:pos + window] if axis_is_x else processed[pos:pos + window, :]
                    fp, fid = G.save_frame_sharded(crop, page_id, page_frames)
                    crops.append(crop)
                    fps.append(fp)
                    ids.append(fid)
                    page_frames += 1
                    if len(crops) >= G.BATCH_SIZE:
                        total_indexed += G.flush_batch(crops, fps, ids, page_id, extra_payload=bdrc_extra)
                        crops, fps, ids = [], [], []

                if crops:
                    total_indexed += G.flush_batch(crops, fps, ids, page_id, extra_payload=bdrc_extra)

                content_pages_done += 1
                page_count += 1
                total_frames += page_frames

            except Exception as exc:
                skipped_pages.append(f"{page_prefix}/p{seq:04d} ({exc})")
                G.log.exception("BDRC page ingest failed for %s", page_id)

        sys.stdout.write(f"\r  Volume {vol_num:>3} done: {page_count} pages ingested, "
                         f"{content_pages_done} cumulative.\n")

    print()
    print("=" * 60)
    print(f"Resource:        {resource_id}")
    print(f"BDRC viewer:     https://library.bdrc.io/show/{resource_id}?uilang=en")
    print(f"Content pages:   {content_pages_done:,}")
    print(f"Frames indexed:  {total_indexed:,}")
    print(f"Volumes touched: {len(volumes)}")
    if skipped_pages:
        review = skipped_pages[:10]
        print(f"Skipped pages:   {len(skipped_pages)}  (first 10: {', '.join(review)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
