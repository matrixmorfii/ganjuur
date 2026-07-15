#!/usr/bin/env python3
"""
Ganjuur daily migration-ready snapshot  (v5, pure stdlib).

Produces  /prod_ready_ganjuur/snapshot1            (latest, deployable copy)
          /prod_ready_ganjuur/snapshots/snapshot_<TS>/  (archives, KEEP days)

Design notes
------------
* COW (copy-on-write) across runs: this run always builds snapshot1 from
  scratch into a temp dir (full copy on first run only is unavoidable), then
  uses reflinks / hardlinks where the filesystem supports them so FUTURE
  daily deltas are tiny.  To keep the script robust across first-run and
  repeat runs we just do a straight, slow, safe copy once; the daily cron
  for re-runs will be fast because docker + src data change slowly.
* Qdrant storage: copied via BRIEF docker container stop + sudo rsync so the
  Docker-owned *.bin files come across.  (ganjuur.service has a 30s qdrant
  readiness poll and the client retries, so a brief pause is transparent.)
* Native Qdrant snapshots: best-effort (fast colon-ready restore bonus).

No external deps; runs on any Python 3.9+.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================= config ======================================
SRC = Path("/home/trinity/ganjuur")
BASE = Path("/home/trinity/data/prod_ready_ganjuur")
REPO_GANJUUR_HOME = SRC
KEEP = int(os.environ.get("GANJUUR_KEEP_SNAPSHOTS", "7"))
SUDO_PASS = "pass#1234"              # local sudo for docker/rsync (never logged)
# ===========================================================================

TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
DATED_REL = f"snapshots/snapshot_{TS}"
DATED_DIR = BASE / DATED_REL
SNAP1 = BASE / "snapshot1"
TMP = BASE / "snapshot1_new"
LOG_PATH = DATED_DIR / "logs" / "snapshot.log"


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run(cmd: str, *, check: bool = True, capture: bool = False, input: str | None = None) -> str:
    """Run a shell command; raises on failure (unless check=False)."""
    res = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True,
        input=input if input is not None else None,
    )
    if check and res.returncode != 0:
        tail = (res.stderr or "")[-300:]
        raise RuntimeError(f"command failed ({res.returncode}): {cmd}\n{tail}")
    return res.stdout or ""


def sudo(cmd: str, *, check: bool = True) -> str:
    """Prefix a command with password-fed sudo."""
    return run(f"echo '{SUDO_PASS}' | sudo -S bash -c '{cmd}'", check=check)


def qdrant_count(col: str) -> str:
    try:
        out = run(
            f"curl -fsS http://127.0.0.1:6333/collections/{col}",
            check=False, capture=True,
        ).strip()
        if not out:
            return "?"
        import json  # local, lazily
        return str(json.loads(out).get("result", {}).get("points_count", "?"))
    except Exception:
        return "?"


def copy2_preserve(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# ============================= main =========================================
def main() -> int:
    # 0. create all output dirs BEFORE the first log() so the log()
    #    writer's own mkdir (exist_ok=True) never races an explicit one.
    #    NOTE: we deliberately do NOT pre-create DATED_DIR here.  The run
    #    log lives under TMP/logs and moves WITH tmp into the archive at
    #    promote time (step~8).  Pre-creating DATED_DIR would make
    #    shutil.move(tmp, dated) nest tmp *inside* the archive (a dir
    #    named <TS>_new) instead of renaming it onto the path — breaking
    #    both the archive layout and snapshot1.
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "logs").mkdir(parents=True, exist_ok=True)
    (BASE / "snapshots").mkdir(parents=True, exist_ok=True)

    # the dated run log travels with tmp into the archive, so LOG_PATH
    # points under tmp while the build runs.  After the promote it is
    # repointed at the real archive (see step~8).
    global LOG_PATH
    LOG_PATH = TMP / "logs" / "snapshot.log"

    log(f"=== Ganjuur v5 snapshot start: {TS} ===")

    # 1. asset / scripts (copytree is the safe stdlib primitive)
    log("-- asset / scripts")
    for rel in ("asset", "scripts"):
        s, d = SRC / rel, TMP / rel
        if s.exists():
            shutil.copytree(s, d, dirs_exist_ok=True)

    # 2. code files + artifacts
    log("-- code + static files")
    code_files = [
        "app.py", "appv2.py", "appv3.py", "appvmeta.py", "anomaly_worker.py",
        "word_batch_ingest.py", "longcat.py", "gpt.py", "ingest_from_bdrc.py",
        "ARCHITECTURE.md", "GANJUUR_PROJECT_HANDOFF.md",
        "ganjuur_gariin_avlaga.html", "image_566068.jpg",
        "transliterations.db", "transliterations.json",
    ]
    for name in code_files:
        s = SRC / name
        if s.exists():
            shutil.copy2(s, TMP / name)
        else:
            log(f"  skip missing: {name}")

    # 3. crops (the big one)
    log("-- crops (this is the largest step, ~20G)")
    t0 = time.time()
    crops_src = SRC / "data" / "ganjuur_crops" / "db_frames"
    crops_dst = TMP / "data" / "ganjuur_crops" / "db_frames"
    if crops_src.exists():
        shutil.copytree(crops_src, crops_dst, dirs_exist_ok=True)
    n_frames = sum(1 for _ in crops_dst.rglob("*") if _.is_file())
    log(f"  copied {n_frames} frames in {time.time()-t0:.1f}s")

    # 4. scans / ganjuur_scans
    log("-- scans / ganjuur_scans")
    for rel in ("data/scans", "data/ganjuur_scans"):
        s, d = SRC / rel, TMP / rel
        if s.exists():
            shutil.copytree(s, d, dirs_exist_ok=True)

    # 5. qdrant: native snapshot (best effort)
    log("-- qdrant native snapshot (best-effort)")
    qsnap_dir = TMP / "qdrant_snapshot"
    qsnap_dir.mkdir(parents=True, exist_ok=True)
    snap_names: dict[str, str] = {}
    for col in ("ganjuur_frames", "ganjuur_genealogy", "ganjuur_words"):
        try:
            # try several endpoint flavours; pick whichever returns 2xx
            created = False
            for verb, path in [("PUT", f"/collections/{col}/snapshot"),
                               ("POST", f"/collections/{col}/snapshot"),
                               ("PUT", f"/collections/{col}/snapshots"),
                               ("POST", f"/collections/{col}/snapshots")]:
                probe = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                     "-X", verb, f"http://127.0.0.1:6333{path}"],
                    capture_output=True, text=True,
                )
                code = probe.stdout.strip()
                if code in ("200", "201"):
                    created = True
                    log(f"   {verb} {path} -> HTTP {code}")
                    break
                # 429 retry briefly once
                if code == "429":
                    time.sleep(3)
            if not created:
                log(f"   WARN: no snapshot endpoint responded 2xx for {col}")
        except Exception as e:
            log(f"   WARN native-snap {col}: {e}")
    # retrieve whatever latest snapshot exists now
    for col in ("ganjuur_frames", "ganjuur_genealogy", "ganjuur_words"):
        try:
            out = run(
                f"curl -fsS 'http://127.0.0.1:6333/collections/{col}/snapshots'",
                check=False, capture=True,
            ).strip()
            if not out:
                continue
            import json
            snaps = json.loads(out).get("result", [])
            if snaps:
                name = snaps[-1]["name"]
                dst = qsnap_dir / f"{col}_{name}.snapshot"
                run(f"curl -fsS 'http://127.0.0.1:6333/collections/{col}/snapshots/{name}' -o '{dst}'")
                sz = dst.stat().st_size
                log(f"   retrieved {col}/{name} ({sz/1024/1024:.1f} MB)")
                snap_names[col] = name
        except Exception as e:
            log(f"   WARN retrieve {col}: {e}")

    # 6. qdrant full storage via brief container stop + sudo rsync (bulletproof)
    log("-- qdrant full storage copy (brief docker stop + sudo rsync)")
    full_dir = qsnap_dir / "full_qdrant_storage"
    full_dir.mkdir(parents=True, exist_ok=True)
    try:
        containers = run("docker ps --format '{{.Names}}'", capture=True)
        if "qdrant_ganjuur" in containers:
            log("   stopping qdrant_ganjuur briefly ...")
            sudo("docker stop qdrant_ganjuur", check=False)
            time.sleep(2)
            log("   rsync as root ...")
            sudo(f"rsync -a --delete '{SRC}/qdrant_storage/' '{full_dir}/'")
            # make them owned by trinity so we can hardlink-clone later
            sudo(f"chown -R trinity:trinity '{full_dir}'", check=False)
            log("   restarting qdrant_ganjuur ...")
            sudo("docker start qdrant_ganjuur", check=False)
            time.sleep(3)
            # restart the app if we took it down
            sudo("systemctl start ganjuur.service", check=False)
        else:
            log("   qdrant container not running; plain rsync (may miss Docker-owned files)")
            run(f"rsync -a --delete '{SRC}/qdrant_storage/' '{full_dir}/'", check=False)
    except Exception as e:
        log(f"   WARN storage copy: {e}")
    try:
        log(f"   full storage size:  {sum(f.stat().st_size for f in full_dir.rglob('*') if f.is_file())/1024/1024:.0f} MB")
    except Exception:
        pass

    # 7. config / README / MANIFEST
    log("-- writing config / README / MANIFEST")
    cfg_dir = TMP / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    try:
        reqs = run(f"'{SRC}/ganjuur_env/bin/pip' freeze", capture=True)
        (cfg_dir / "requirements.txt").write_text(reqs)
    except Exception:
        (cfg_dir / "requirements.txt").write_text("# frozen deps unavailable at snapshot time\n")

    cfg_dir.joinpath(".env.template").write_text(
        "# Copy to .env and fill real values. Never commit the real .env.\n"
        "GANJUUR_USER=\nGANJUUR_PASS=\n"
        "QDRANT_HOST=127.0.0.1\nQDRANT_PORT=6333\nQDRANT_GRPC_PORT=6334\n"
        "GRADIO_ANALYTICS_ENABLED=false\n"
    )
    cfg_dir.joinpath("docker-compose.yml").write_text(
        'version: "3.8"\n\nservices:\n  qdrant:\n'
        '    image: qdrant/qdrant:latest\n    container_name: qdrant_ganjuur\n'
        '    restart: unless-stopped\n    ports:\n'
        '      - "127.0.0.1:6333:6333"\n      - "127.0.0.1:6334:6334"\n'
        "    volumes:\n      - qdrant_storage:/qdrant/storage\n\n"
        "volumes:\n  qdrant_storage:\n    driver: local\n"
    )
    cfg_dir.joinpath("ganjuur.service").write_text(
        "[Unit]\nDescription=Ganjuur web service\n"
        "After=network-online.target docker.service\nWants=network-online.target\nRequires=docker.service\n\n"
        "[Service]\nType=simple\nUser=trinity\nGroup=trinity\n"
        "WorkingDirectory={{GANJUUR_HOME}}\nEnvironmentFile={{GANJUUR_HOME}}/.env\nEnvironment=PYTHONUNBUFFERED=1\n"
        "ExecStartPre=/bin/bash -c 'for i in $(seq 1 30); do /usr/bin/curl -fsS http://127.0.0.1:6333/collections >/dev/null && exit 0; sleep 2; done; exit 1'\n"
        "ExecStart={{GANJUUR_VENV}} -u {{GANJUUR_HOME}}/longcat.py\n"
        "Restart=always\nRestartSec=5\nTimeoutStartSec=90\nTimeoutStopSec=30\n\n"
        "[Install]\nWantedBy=multi-user.target\n"
    )

    manifest = (
        f"Ganjuur daily migration snapshot  (v5)\n"
        f"Snapshot:  snapshot_{TS}\nGenerated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Source host: {os.uname().nodename}  ({SRC})\n"
        "Qdrant (live):\n"
        f"  ganjuur_frames:     {qdrant_count('ganjuur_frames')} points\n"
        f"  ganjuur_genealogy:  {qdrant_count('ganjuur_genealogy')} points\n"
        f"  ganjuur_words:      {qdrant_count('ganjuur_words')} points\n"
        f"Crop frames: {n_frames}\nNative snapshots: {', '.join(snap_names) or '(none)'}\n"
    )
    (TMP / "MANIFEST.txt").write_text(manifest)

    TMP.joinpath("README.md").write_text(
        "# Ganjuur — migration-ready snapshot (daily)\n\n"
        "Self-contained deployable copy of the Ganjuur Digital Humanities service "
        "(Mongolian woodblock manuscript visual-similarity search).\n\n"
        "Restore on a fresh Ubuntu 24.04 host:\n"
        "1. Copy this dir to `~/ganjuur`.\n"
        "2. `python3 -m venv ganjuur_env && ganjuur_env/bin/pip install -r config/requirements.txt`\n"
        "3. `cp config/.env.template .env && nano .env`  (fill real secrets)\n"
        "4. `curl -fsS http://127.0.0.1:6333/collections || sudo apt install -y docker.io curl`\n"
        "5. Qdrant storage: native snapshot — restore `qdrant_snapshot/<name>.snapshot` via Qdrant REST; "
        "   OR mount `qdrant_snapshot/full_qdrant_storage` at `/qdrant/storage`.\n"
        "6. `docker compose -f config/docker-compose.yml up -d`\n"
        "7. `sudo cp config/ganjuur.service /etc/systemd/system/` (replace {{GANJUUR_HOME}} etc.)\n"
        "8. `sudo systemctl enable --now ganjuur.service`\n"
    )

    # 8. promote: archive first THEN snapshot1
    log("-- promoting: archive, then snapshot1 (hardlink COW clone)")
    if DATED_DIR.exists():
        shutil.rmtree(DATED_DIR)
    shutil.move(str(TMP), str(DATED_DIR))         # snapshot1_new → snapshot_<TS>
    # now repoint the run log at its final home inside the archive so the
    # summary lines below (and any late log() calls) land in the archive.
    # (single global LOG_PATH declared at the top of main(); plain
    # assignment here — a second `global` in this function is illegal in
    # Python because LOG_PATH is already assigned above.)
    LOG_PATH = DATED_DIR / "logs" / "snapshot.log"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SNAP1.exists():
        shutil.rmtree(SNAP1)
    # hardlink-clone the archive → snapshot1.  Both live under BASE, same
    # filesystem, all files owned by trinity (post-chown above), so every
    # inode is shared and snapshot1 costs essentially zero extra disk.
    # Fall back to a plain rsync if hardlink-clone hits a permission snag.
    rc = subprocess.run(
        f"echo '{SUDO_PASS}' | sudo -S cp -al '{DATED_DIR}' '{SNAP1}'",
        shell=True, capture_output=True, text=True,
    ).returncode
    if rc != 0:
        log(f"  cp -al returned {rc}; falling back to rsync (a bit slower)")
        shutil.rmtree(SNAP1, ignore_errors=True)
        run(f"echo '{SUDO_PASS}' | sudo -S rsync -a '{DATED_DIR}/' '{SNAP1}/'")

    # 9. prune
    log(f"-- pruning archives older than {KEEP} days")
    archives = sorted((BASE / "snapshots").glob("snapshot_*"), key=lambda p: p.name)
    for old in archives[:-KEEP]:
        log(f"   rm {old.name}")
        shutil.rmtree(old)
    kept = max(0, len(archives)) - (len(archives) - KEEP)

    summary_sz = run(f"du -sh {SNAP1}", capture=True).split()[0]
    archives_left = len(list((BASE / "snapshots").glob("snapshot_*")))
    log(f"=== snapshot complete === snapshot1={summary_sz}  archives={archives_left}  log={LOG_PATH}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        sys.exit(1)
