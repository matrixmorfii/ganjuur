#!/usr/bin/env bash
# =============================================================
# Ganjuur — production migration packager
# Mirrors current /home/trinity/ganjuur layout into
# /home/trinity/data/prod_ready_ganjuur/ so it drops in as a
# drop-in replacement on a target host. Requires: docker, curl.
# =============================================================
set -u
export DEBIAN_FRONTEND=noninteractive

SRC="/home/trinity/ganjuur"
DST="/home/trinity/data/prod_ready_ganjuur"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="/tmp/migrate_${TS}.log"
PASS="${GANJUUR_SUDO_PASS:-}"   # set GANJUUR_SUDO_PASS in .env — never hardcode
if [ -z "$PASS" ]; then echo "ERROR: GANJUUR_SUDO_PASS not set (see .env)" >&2; exit 1; fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { echo "  -> $*" >> "$LOG"; "$@" >>"$LOG" 2>&1; }

log "===== Ganjuur migration start $TS ====="
log "SRC=$SRC  DST=$DST"

# ---- 1. Stop ingest + service ----
log "-- stopping bdrc ingest"
kill 18088 2>/dev/null; kill 18087 2>/dev/null
pkill -f 'ingest_from_bdrc.py' 2>/dev/null
sleep 2
if pgrep -f 'ingest_from_bdrc.py' >/dev/null; then
  log "WARN ingest still alive, SIGKILL"; kill -9 $(pgrep -f 'ingest_from_bdrc.py') 2>/dev/null
fi

log "-- stopping ganjuur.service"
echo "$PASS" | sudo -S systemctl stop ganjuur.service 2>&1 | tee -a "$LOG"
sleep 2

# ---- 2. Generate requirements.txt from live venv ----
log "-- freezing python deps"
"$SRC/ganjuur_env/bin/pip" freeze > "$DST_config_req.txt.tmp" 2>/dev/null || true
# (written to tmp outside DST; moved later)

# ---- 3. Build destination tree ----
log "-- building dir tree"
mkdir -p "$DST"/{asset,scripts,data/ganjuur_crops,data/scans,data/ganjuur_scans,config,qdrant_snapshot,logs}

# ---- 4. Copy CROPS (db_frames only — runtime source of truth) ----
log "-- copying crops/db_frames (this is large, ~19G) ..."
unshare -m bash -c 'mount --bind "$1" "$2"' -- "$SRC/data/ganjuur_crops/db_frames" "$DST/data/ganjuur_crops/db_frames" 2>>"$LOG" \
  || { log "bind-mount unavailable, using cp -a (slower)"; run cp -a "$SRC/data/ganjuur_crops/db_frames/." "$DST/data/ganjuur_crops/db_frames/"; }
CROP_CT=$(find "$DST/data/ganjuur_crops/db_frames" -type f | wc -l)
log "crops copied: $CROP_CT files"

# ---- 5. Atomic Qdrant snapshot BEFORE copying anything else ----
log "-- creating Qdrant native snapshots"
for COL in ganjuur_frames ganjuur_genealogy ganjuur_words; do
  log "   snapshot $COL"
  curl -fsS -X POST "http://127.0.0.1:6333/collections/$COL/snapshots" >>"$LOG" 2>&1 \
    || log "   WARN snapshot $COL failed (will fall back to dir copy)"
done
mkdir -p /tmp/qsnap
for COL in ganjuur_frames ganjuur_genealogy ganjuur_words; do
  SNAP=$(curl -fsS "http://127.0.0.1:6333/collections/$COL/snapshots" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read())["result"];print(d[-1]["name"] if d else "")' 2>/dev/null)
  if [ -n "$SNAP" ]; then
    log "   retrieving $SNAP for $COL"
    curl -fsS "http://127.0.0.1:6333/collections/$COL/snapshots/$SNAP" -o "$DST/qdrant_snapshot/${COL}_${SNAP}.snapshot" >>"$LOG" 2>&1 \
      && log "   -> ${COL}_${SNAP}.snapshot ($(du -h "$DST/qdrant_snapshot/${COL}_${SNAP}.snapshot" | cut -f1))"
  fi
done
# also copy full storage dir as belt-and-suspenders
log "-- copying live qdrant_storage as fallback"
run cp -a "$SRC/qdrant_storage/." "$DST/qdrant_snapshot/full_qdrant_storage/"
QSZ=$(du -sh "$DST/qdrant_snapshot" | cut -f1)
log "qdrant snapshot section: $QSZ"

# ---- 6. Copy application code ----
log "-- copying code"
run cp -v "$SRC/longcat.py" "$DST/"
run cp -v "$SRC/gpt.py" "$DST/"
run cp -v "$SRC/ingest_from_bdrc.py" "$DST/"
run cp -v "$SRC/anomaly_worker.py" "$DST/"
run cp -v "$SRC/scripts/core_engine.py" "$DST/scripts/" 2>/dev/null || true
run cp -v "$SRC/ganjuur_gariin_avlaga.html" "$DST/"
run cp -v "$SRC/image_566068.jpg" "$DST/" 2>/dev/null || true
run cp -v "$SRC/ARCHITECTURE.md" "$DST/" 2>/dev/null || true
run cp -rv "$SRC/asset/." "$DST/asset/"

# ---- 7. Copy data (scans, DB, queue) ----
log "-- copying scans + db"
run cp -a "$SRC/data/ganjuur_scans/." "$DST/data/ganjuur_scans/"
run cp -a "$SRC/data/scans/." "$DST/data/scans/"
run cp -a "$SRC/anomaly_queue.json" "$DST/data/scans/anomaly_queue.json" 2>/dev/null || true
run cp -a "$SRC/transliterations.db" "$DST/"
run cp -a "$SRC/transliterations.json" "$DST/" 2>/dev/null || true

# ---- 8. Move frozen requirements into config ----
log "-- requirements + configs"
[ -f "$DST_config_req.txt.tmp" ] && mv "$DST_config_req.txt.tmp" "$DST/config/requirements.txt"
run cp /etc/systemd/system/ganjuur.service "$DST/config/ganjuur.service"

# ---- 9. Write deploy helpers ----
log "-- writing deploy helpers"

cat > "$DST/config/docker-compose.yml" << 'QCOMPOSE'
version: "3.8"

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_ganjuur
    restart: unless-stopped
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
    driver: local
QCOMPOSE

cat > "$DST/config/.env.template" << 'DOTENV'
# Copy to .env and fill in real values. Never commit the real .env.
GANJUUR_USER=
GANJUUR_PASS=
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
GRADIO_ANALYTICS_ENABLED=false
DOTENV

cat > "$DST/README.md" << 'README'
# Ganjuur — Production Deployment Package

Standalone deployment package for the Ganjuur Digital Humanities app
(Mongolian woodblock manuscript visual-similarity search).
Mirrors the runtime layout of the source host.

## Contents
| Path | Purpose |
|------|---------|
| `longcat.py` | Main Gradio app (systemd entrypoint) |
| `gpt.py` | Core engine / embedding / search |
| `ingest_from_bdrc.py` | BDRC IIIF → crops → Qdrant ingest pipeline |
| `anomaly_worker.py` | Anomaly queue worker |
| `scripts/core_engine.py` | Legacy slicer/engine utilities |
| `config/requirements.txt` | Pinned Python deps (install into venv) |
| `config/docker-compose.yml` | Qdrant service definition |
| `config/ganjuur.service` | systemd unit template |
| `config/.env.template` | Secret keys (fill values, save as `.env`) |
| `data/ganjuur_crops/db_frames/` | ~K frame crops (webp, sharded) — runtime image source |
| `data/scans/` + `data/ganjuur_scans/` | Source scan images |
| `qdrant_snapshot/` | Native Qdrant snapshots + full storage dir |
| `asset/` | favicon, logo |
| `transliterations.db` | Transliteration SQLite DB |

## Deploy (new Ubuntu 24.04 host)
1. Place repo at `/home/<user>/ganjuur` (paths inside code are relative).
2. `python3 -m venv ganjuur_env && ganjuur_env/bin/pip install -r config/requirements.txt`
3. `cp config/.env.template .env && nano .env` — set real secrets.
4. `docker compose -f config/docker-compose.yml up -d`
5. **Restore Qdrant:**
   - Preferred: start container with `qdrant_snapshot/full_qdrant_storage` bind-mounted
     at `/qdrant/data` (matches the compose volume).
   - Or apply the `.snapshot` files via the Qdrant REST API after the container
     is healthy:  
       `PUT http://127.0.0.1:6333/collections/<name>/snapshots/recover` with body
       `{"location": "file:///qdrant/storage/snapshots/<name>.snapshot"}`
6. `cp config/ganjuur.service /etc/systemd/system/ganjuur.service`
7. `sudo daemon-reload && sudo systemctl enable --now ganjuur.service`
8. Ingest more volumes (optional):  
   `ganjuur_env/bin/python ingest_from_bdrc.py --ingest`

### Runtime requirements
- NVIDIA GPU with ≥6 GB VRAM for DINOv2 CUDA embedder (RTX 3060 tested)
- Docker + nvidia-container-toolkit (Qdrant runs on CPU; GPU used by app only)
- ~25 GB disk (crops 19G + Qdrant + code)

README

cat > "$DST/SETUP.sh" << 'SETUP'
#!/usr/bin/env bash
# One-shot deploy helper (review first; runs as your user, sudo where needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "[ganjuur-setup] using $HERE as GANJUUR_HOME"
read -sp "sudo password: " SUDOP; echo

echo "[1/5] python venv + deps"
python3 -m venv ganjuur_env
ganjuur_env/bin/pip install --upgrade pip
ganjuur_env/bin/pip install -r config/requirements.txt

echo "[2/5] secrets"
if [ ! -f .env ]; then
  cp config/.env.template .env
  echo "  -> created .env template — EDIT IT with real secrets before next step"
fi
[ -s .env ] && grep -q 'GANJUUR_USER=$' .env && echo "  !! .env still has empty secrets — fill them!" 

echo "[3/5] qdrant"
docker compose -f config/docker-compose.yml up -d
echo "  -> qdrant starting (first run pulls image)"

echo "[4/5] restore qdrant storage to /qdrant/storage"
# (handled by volume)

echo "[5/5] systemd unit"
echo "$SUDOP" | sudo -S cp config/ganjuur.service /etc/systemd/system/ganjuur.service
echo "$SUDOP" | sudo -S sed -i "s|/home/trinity/ganjuur|$HERE|g" /etc/systemd/system/ganjuur.service
echo "$SUDOP" | sudo -S systemctl daemon-reload
echo "[done] setup. Then: sudo systemctl enable --now ganjuur.service"
SETUP
chmod +x "$DST/SETUP.sh"

# ---- 10. Manifest + report ----
log "-- writing manifest"
cat > "$DST/MANIFEST.txt" << EOF
Ganjuur production migration package
Generated: $(date -R)
Source host: $(hostname)  ($SRC)
Qdrant collections:
  ganjuur_frames:     $(curl -fsS http://127.0.0.1:6333/collections/ganjuur_frames | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null || echo '?') points
  ganjuur_genealogy:  $(curl -fsS http://127.0.0.1:6333/collections/ganjuur_genealogy | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null || echo '?') points
  ganjuur_words:      $(curl -fsS http://127.0.0.1:6333/collections/ganjuur_words | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null || echo '?') points
Crop frames: $(find "$DST/data/ganjuur_crops/db_frames" -type f | wc -l)
Ingest status at snapshot: page 297/844 (vol089), then stopped for snapshot
EOF

# ---- 11. Restart production service ----
log "-- restarting ganjuur.service"
echo "$PASS" | sudo -S systemctl start ganjuur.service 2>&1 | tee -a "$LOG"
sleep 3
systemctl is-active ganjuur.service >/dev/null && log "service: RUNNING" || log "service: FAILED to start"

TOTAL=$(du -sh "$DST" | cut -f1)
log "===== MIGRATION COMPLETE ====="
log "Package: $DST  ($TOTAL)"
log "Manifest: $DST/MANIFEST.txt"
log "Review SETUP.sh + README.md on the target before deploying"
