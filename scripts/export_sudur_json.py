#!/usr/bin/env python3
"""
Build sudur_data.json for the HTML manuscript viewer (ganjuur_sudur_kharagch.html).

Data sources:
  data/scans/          – full-page scan images (BRC image IDs)
  data/ganjuur_crops/  – crop patches with volume/page metadata in filenames
  transliterations.db  – human-entered Cyrillic transliterations (if available)

Strategy:
  1. Read the volume→page structure from crop filenames (fast, local, accurate).
  2. Match scans to volumes by querying BDRC (image ID → volume), falling back
     to proportional distribution when the API is unavailable.
  3. Group pages into sections; assemble the 4-level hierarchy.
  4. Write sudur_data.json.

Usage:
    python scripts/export_sudur_json.py            # full export
    python scripts/export_sudur_json.py --pages-per-section 10
    python scripts/export_sudur_json.py --max-volumes 5   # test on 5 volumes
"""
import argparse
import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

BASE = Path(os.getenv("GANJUUR_BASE_DIR", "/home/trinity/ganjuur")).expanduser()
SCANS = BASE / "data" / "scans"
CROPS = BASE / "data" / "ganjuur_crops" / "db_frames"
DB = BASE / "transliterations.db"
OUT = BASE / "sudur_data.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────
# 1. Volume → page structure from crops
# ──────────────────────────────────────────────────────────────────────
def structure_from_crops() -> dict[int, list[int]]:
    """Return {volume_number: [page_numbers]} by parsing crop filenames."""
    vol_pages: dict[int, set[int]] = defaultdict(set)
    for f in CROPS.rglob("*"):
        if f.suffix.lower() not in {".webp", ".jpg", ".png"}:
            continue
        m = re.search(r"vol(\d+)_p(\d+)", f.name)
        if m:
            vol_pages[int(m.group(1))].add(int(m.group(2)))
    return {v: sorted(pages) for v, pages in sorted(vol_pages.items())}


# ──────────────────────────────────────────────────────────────────────
# 2. Match scans to volumes
# ──────────────────────────────────────────────────────────────────────
def collect_scans() -> list[Path]:
    """All scan images, sorted by name."""
    imgs = [f for f in SCANS.rglob("*") if f.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=lambda f: f.name)


def map_scans_to_volumes(
    scans: list[Path],
    vol_pages: dict[int, list[int]],
    use_api: bool = False,
) -> dict[int, list[Path]]:
    """
    Assign each scan to a volume.

    If *use_api* is True, query BDRC to map image IDs to volumes (slow but
    exact).  Otherwise distribute scans proportionally across volumes based on
    their page counts (fast, approximate).
    """
    if not scans or not vol_pages:
        return {}

    if use_api:
        try:
            return _map_via_bdrc(scans, vol_pages)
        except Exception as exc:
            print(f"  BDRC API mapping failed ({exc}); falling back to proportional")

    # Proportional distribution: volumes with more pages get more scans.
    total_pages = sum(len(p) for p in vol_pages.values())
    volumes = list(vol_pages.keys())
    scans_per_vol: dict[int, int] = {}
    allocated = 0
    for i, vol in enumerate(volumes):
        share = max(1, round(len(scans) * len(vol_pages[vol]) / total_pages))
        if i == len(volumes) - 1:
            share = len(scans) - allocated  # last volume gets the remainder
        scans_per_vol[vol] = share
        allocated += share

    result: dict[int, list[Path]] = {}
    idx = 0
    for vol in volumes:
        n = scans_per_vol[vol]
        result[vol] = scans[idx:idx + n]
        idx += n
    return result


def _map_via_bdrc(
    scans: list[Path], vol_pages: dict[int, list[int]]
) -> dict[int, list[Path]]:
    """Map scans → volumes via the BDRC RDF API (one query per scan)."""
    import requests

    cache: dict[str, int] = {}
    result: dict[int, list[Path]] = defaultdict(list)

    for scan in scans:
        stem = scan.stem  # e.g. I1PD1102690001
        if stem in cache:
            vol = cache[stem]
        else:
            vol = _bdrc_image_to_volume(stem)
            cache[stem] = vol
            time.sleep(0.05)  # be polite to the API
        if vol and vol in vol_pages:
            result[vol].append(scan)

    return dict(result)


def _bdrc_image_to_volume(image_id: str) -> int | None:
    """Query BDRC RDF to find which volume an image belongs to."""
    import requests

    rdf_url = f"https://ldspdi.bdrc.io/resource/{image_id}.rdf"
    try:
        r = requests.get(rdf_url, headers={"Accept": "application/rdf+xml"}, timeout=10)
        r.raise_for_status()
        m = re.search(r"volumeHasVolume.*?(\d+)", r.text)
        if m:
            return int(m.group(1))
        # fallback: look for volume number in the RDF
        m = re.search(r"/vol(.*?)/", r.text)
        if m:
            nums = re.findall(r"\d+", m.group(1))
            if nums:
                return int(nums[0])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# 3. Transliterations (best-effort)
# ──────────────────────────────────────────────────────────────────────
def load_texts() -> dict[str, str]:
    """Return {source_image: text_content} from the DB."""
    texts: dict[str, str] = {}
    if not DB.exists():
        return texts
    con = sqlite3.connect(str(DB))
    try:
        for src, txt in con.execute(
            "SELECT source_image, text_content FROM transliterations WHERE text_content IS NOT NULL AND text_content != ''"
        ):
            # store by both full path and basename for flexible matching
            texts[Path(src).name] = txt
            texts[src] = txt
    except Exception:
        pass
    finally:
        con.close()
    return texts


# ──────────────────────────────────────────────────────────────────────
# 4. Assemble the hierarchy
# ──────────────────────────────────────────────────────────────────────
def build_hierarchy(
    vol_scans: dict[int, list[Path]],
    pages_per_section: int,
    texts: dict[str, str],
) -> dict:
    aimags = []
    all_vols = []
    for vol_num, scans in sorted(vol_scans.items()):
        if not scans:
            continue
        sections = []
        for k in range(0, len(scans), pages_per_section):
            chunk = scans[k:k + pages_per_section]
            pages = []
            for s in chunk:
                rel = s.relative_to(BASE).as_posix()
                txt = texts.get(s.name, "") or texts.get(rel, "")
                pages.append({"id": s.stem, "image": rel, "text": txt})
            sections.append({
                "id": f"v{vol_num}-s{k // pages_per_section + 1}",
                "title": f"Хэсэг {k // pages_per_section + 1}",
                "pages": pages,
            })
        chapters = [{
            "id": f"v{vol_num}-c1",
            "title": "Нэгдэн бөлөг",
            "sections": sections,
        }]
        all_vols.append({
            "id": f"v{vol_num}",
            "number": vol_num,
            "title": f"Боть {vol_num}",
            "chapters": chapters,
        })

    aimags.append({"id": "a1", "title": "Ганжуур", "volumes": all_vols})
    return {"aimags": aimags}


# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-per-section", type=int, default=8,
                        help="How many pages per section (default: 8)")
    parser.add_argument("--max-volumes", type=int, default=None,
                        help="Limit to first N volumes (for testing)")
    parser.add_argument("--use-api", action="store_true",
                        help="Map scans via BDRC API (slow, exact)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    print("[1/4] Reading volume structure from crops …")
    vol_pages = structure_from_crops()
    print(f"  found {len(vol_pages)} volumes")
    if args.max_volumes:
        vol_pages = {v: vol_pages[v] for v in sorted(vol_pages)[:args.max_volumes]}
        print(f"  limited to {len(vol_pages)} volumes")

    print("[2/4] Collecting scans …")
    scans = collect_scans()
    print(f"  found {len(scans)} scans")

    print("[3/4] Mapping scans to volumes …")
    vol_scans = map_scans_to_volumes(scans, vol_pages, use_api=args.use_api)
    assigned = sum(len(v) for v in vol_scans.values())
    print(f"  assigned {assigned} scans across {len(vol_scans)} volumes")

    print("[4/4] Building hierarchy + writing JSON …")
    texts = load_texts()
    print(f"  loaded {len(texts)} transliterations")
    data = build_hierarchy(vol_scans, args.pages_per_section, texts)

    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    total_pages = sum(
        len(p)
        for a in data["aimags"]
        for v in a["volumes"]
        for c in v["chapters"]
        for s in c["sections"]
        for p in s["pages"]
    )
    total_sections = sum(
        len(c["sections"])
        for a in data["aimags"]
        for v in a["volumes"]
        for c in v["chapters"]
    )
    print(f"\n  ✔ {args.out}")
    print(f"    {len(data['aimags'][0]['volumes'])} volumes, "
          f"{total_sections} sections, {total_pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
