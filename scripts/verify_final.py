import os
from pathlib import Path

BASE = Path("/home/trinity/data/prod_ready_ganjuur")
SNAP1 = BASE / "snapshot1"
SP = BASE / "snapshots"
dated = sorted(SP.glob("snapshot_*"))[-1] if list(SP.glob("snapshot_*")) else None

print("=== dated archive ===")
print(" ", dated.name if dated else "NONE")
if dated:
    for e in sorted(dated.iterdir()):
        tag = "(dir)" if e.is_dir() else "(file)"
        print("    ", e.name, tag)
    nested = dated / "snapshot1_new"
    print("    nested snapshot1_new?", nested.exists(), "<- should be False")
    crops = dated / "data" / "ganjuur_crops" / "db_frames"
    n = sum(1 for _ in crops.rglob("*") if _.is_file()) if crops.exists() else -1
    print("    crops files:", n)
    for chk in ("MANIFEST.txt", "README.md", "longcat.py", "config", "qdrant_snapshot", "asset", "data"):
        print("   ", chk, (dated / chk).exists())

print("\n=== snapshot1 deployable ===")
if SNAP1.exists():
    for e in sorted(SNAP1.iterdir()):
        tag = "(dir)" if e.is_dir() else "(file)"
        print("    ", e.name, tag)
    crops = SNAP1 / "data" / "ganjuur_crops" / "db_frames"
    n = sum(1 for _ in crops.rglob("*") if _.is_file()) if crops.exists() else -1
    print("    crops files:", n)
    for chk in ("MANIFEST.txt", "README.md", "longcat.py", "config", "qdrant_snapshot", "asset", "data"):
        print("   ", chk, (SNAP1 / chk).exists())
    m = SNAP1 / "MANIFEST.txt"
    if m.exists():
        print("\n=== MANIFEST ===")
        print(m.read_text())
else:
    print("    MISSING")

print("\n=== disk ===")
os.system("df -h /home/trinity/data | tail -1")

print("\n=== named snapshot1_new leftover anywhere? ===")
print("  ", (BASE / "snapshot1_new").exists(), "(should be False)")
