"""Disk cleanup for the Ganjuur server — SAFE version.

Safety rules (learned the hard way):
  * Default is DRY RUN. Pass --apply to actually delete anything.
  * WAL dirs are only removed when Qdrant is NOT running (deleting WAL
    under a live Qdrant corrupts the collection).
  * INDEXED_VOLS is derived live from ganjuur_frames_v2 — no hardcoded
    list that can drift from reality. Both source formats are parsed
    ("vol45" and "vol45_p123").
  * Never touches vectordb/ or vectordb_pre_docker_backup/ (HANDOFF rule).
"""
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

QDRANT_URL = 'http://localhost:6333'
COLLECTION = 'ganjuur_frames_v2'
CROPS_DIR = Path('/home/trinity/ganjuur/data/ganjuur_crops/db_frames')
QDRANT_DIR = Path('/home/trinity/ganjuur/qdrant_storage')

APPLY = '--apply' in sys.argv


def disk_free_gb():
    return shutil.disk_usage('/').free / (1024**3)


def qdrant_is_running():
    try:
        with urllib.request.urlopen(QDRANT_URL + '/readyz', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def indexed_volumes():
    """Scroll ganjuur_frames_v2 and collect distinct volume numbers."""
    vols = set()
    offset = None
    while True:
        body = {
            'limit': 1000,
            'with_payload': ['source'],
            'with_vector': False,
        }
        if offset is not None:
            body['offset'] = offset
        req = urllib.request.Request(
            f'{QDRANT_URL}/collections/{COLLECTION}/points/scroll',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)['result']
        for pt in data['points']:
            src = (pt.get('payload') or {}).get('source', '')
            m = re.match(r'vol(\d+)', str(src))
            if m:
                vols.add(int(m.group(1)))
        offset = data.get('next_page_offset')
        if offset is None:
            break
    return vols


def remove(path, is_dir=False):
    if APPLY:
        if is_dir:
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()
        print(f'  deleted: {path}')
    else:
        print(f'  [dry-run] would delete: {path}')


print(f'Free space before: {disk_free_gb():.1f} GB')
print(f'Mode: {"APPLY (real deletion)" if APPLY else "DRY RUN (pass --apply to delete)"}')

# ── 1. Qdrant WAL ────────────────────────────────────────────────────────────
assert 'vectordb' not in str(QDRANT_DIR), 'refusing to touch vectordb paths'
if qdrant_is_running():
    print('\n[1] WAL cleanup SKIPPED: Qdrant is RUNNING. Stop it first')
    print('    (systemctl stop qdrant_ganjuur / docker stop) and re-run.')
else:
    print('\n[1] Qdrant not running — WAL cleanup:')
    for wal in QDRANT_DIR.rglob('wal'):
        if wal.is_dir():
            remove(wal, is_dir=True)

# ── 2. Loose debug/analysis files directly under ganjuur_crops ──────────────
print('\n[2] Debug files under ganjuur_crops/:')
crops_root = CROPS_DIR.parent
for f in crops_root.iterdir():
    if f.is_file():
        print(f'    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)')
        remove(f)

# ── 3. Crops of volumes that are already indexed in v2 ──────────────────────
print('\n[3] Crops of indexed volumes:')
if qdrant_is_running():
    try:
        vols = indexed_volumes()
        print(f'    indexed volumes in {COLLECTION}: {len(vols)} -> {sorted(vols)}')
        deleted = 0
        deleted_bytes = 0
        for d in CROPS_DIR.iterdir():
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if not f.is_file():
                    continue
                m = re.search(r'vol(\d+)_', f.name)
                if m and int(m.group(1)) in vols:
                    deleted_bytes += f.stat().st_size
                    remove(f)
                    deleted += 1
        print(f'    {deleted} crop files ({deleted_bytes / 1024**3:.1f} GB)')
    except Exception as e:
        print(f'    SKIPPED: could not read {COLLECTION}: {e}')
else:
    print('    SKIPPED: Qdrant is down — cannot derive indexed volumes safely.')

print(f'\nFree space after: {disk_free_gb():.1f} GB')
if not APPLY:
    print('Nothing was deleted (dry run). Re-run with --apply when the plan looks right.')
