#!/usr/bin/env bash
# ---------------------------------------------------------------
#  Ganjuur daily migration-ready snapshot  (v4)
#
#    - always keeps /prod_ready_ganjuur/snapshot1   (latest)
#    - archives  /prod_ready_ganjuur/snapshots/snapshot_<TS>/  (KEEP days)
#    - large trees reflinked across runs (COW → daily delta cost)
#    - Qdrant backup = native snapshot (API) + full storage dir
#      via brief container stop + root rsync (copies every file)
#    - each snapshot auto-generates config + README + MANIFEST
# ---------------------------------------------------------------
set -uo pipefail
# NOTE: NOT 'set -e' here because the file-copy section intentionally
# proceeds even when an individual snapshot step fails (we always
# fall back to full-storage copy).

SRC="/home/trinity/ganjuur"
BASE="/home/trinity/data/prod_ready_ganjuur"
D_TS="$(date +%Y%m%d_%H%M%S)"
DATED="snapshots/snapshot_${D_TS}"
LATEST="snapshot1"
LOGDIR="logs"
LOG="${BASE}/${DATED}/logs/snapshot.log"
KEEP="${GANJUUR_KEEP_SNAPSHOTS:-7}"
PASS="${GANJUUR_SUDO_PASS:-pass#1234}"

# ---- logging bootstrap (create LOG dir BEFORE any say()) ------------
mkdir -p "${BASE}/${DATED}/logs" "${BASE}/${DATED}/config" "${BASE}/snapshots"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}" 2>/dev/null || true; }

say "=== Ganjuur daily snapshot start: ${D_TS} ==="

# ================================================================
# 1. incremental hardlink baseline (copy-on-write for COW)
# ================================================================
REF="${BASE}/${LATEST}"
TMP="${BASE}/${D_TS}_build"
rm -rf "${TMP}"
mkdir -p "${TMP}"             # <-- must exist even on first run (no REF yet)

linkfarm() {                  # linkfarm <src> <dst>  — hardlink clone if src exists
    if [ -d "$1" ]; then cp -al "$1" "$2"; else mkdir -p "$2"; fi
}

if [ -d "${REF}" ]; then
    for sub in asset scripts data data/ganjuur_crops db_frames qdrant_snapshot/full_qdrant_storage; do
        [ -d "${REF}/${sub}" ] && linkfarm "${REF}/${sub}" "${TMP}/${sub}"
    done
    # dig into data.*/db_frames, scans, ganjuur_scans (nested)
    for sub in data/ganjuur_crops/db_frames data/scans data/ganjuur_scans; do
        [ -d "${REF}/${sub}" ] && linkfarm "${REF}/${sub}" "${TMP}/${sub}"
    done
fi

# Quick sanity: on first run REF didn't exist — just continue; TMP is fresh.

# ================================================================
# 2. code + static   (small — copy straight from SRC)
# ================================================================
say "-- code + static"
for f in app.py appv2.py appv3.py appvmeta.py anomaly_worker.py word_batch_ingest.py \
         longcat.py gpt.py ingest_from_bdrc.py \
         ARCHITECTURE.md GANJUUR_PROJECT_HANDOFF.md \
         ganjuur_gariin_avlaga.html image_566068.jpg \
         transliterations.db transliterations.json; do
    [ -e "${SRC}/$f" ] && cp -a "${SRC}/$f" "${TMP}/$f" || say "    skip missing: $f"
done
mkdir -p "${TMP}/asset"
rsync  -a "${SRC}/asset/"  "${TMP}/asset/"
mkdir -p "${TMP}/scripts"
rsync  -a "${SRC}/scripts/" "${TMP}/scripts/"

# ================================================================
# 3. crops (COW on repeat runs)   <-- largest item
# ================================================================
say "-- crops incremental"
mkdir -p "${TMP}/data/ganjuur_crops/db_frames"
if [ -d "${REF}/data/ganjuur_crops/db_frames" ]; then
    # baseline laid down by linkfarm; rsync only the delta over it
    rsync -a --delete "${SRC}/data/ganjuur_crops/db_frames/" "${TMP}/data/ganjuur_crops/db_frames/" >>"${LOG}" 2>&1
else
    say "   first run — full crop copy (this is the large one, ~19G)"
    cp -a "${SRC}/data/ganjuur_crops/db_frames" "${TMP}/data/ganjuur_crops/"
fi

# ================================================================
# 4. scans + ganjuur_scans + db siblings  (COW)
# ================================================================
say "-- scans + ganjuur_scans"
for d in data/scans data/ganjuur_scans; do
    mkdir -p "${TMP}/${d}"
    rsync -a --delete "${SRC}/${d}/" "${TMP}/${d}/"
done

# ================================================================
# 5. Qdrant = native snapshot (API)   <-- portable across hosts
# ================================================================
say "-- qdrant native snapshot (via API)"
QSNAP_DIR="${TMP}/qdrant_snapshot"
mkdir -p "${QSNAP_DIR}"

# qdrant live count helper
qdcount() {
    curl -fsS "http://127.0.0.1:6333/collections/$1" 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("result",{}).get("points_count","?"))' \
      2>/dev/null || echo "?"
}

# Determine the right snapshot endpoint for this Qdrant version.
# Qdrant >=1.9 uses PUT /collections/{name}/snapshot (singular), older POST.
try_create_snapshot() {
    local col="$1"
    local url_post="http://127.0.0.1:6333/collections/${col}/snapshot"
    # try PUT first (newer), fall back to POST (older)
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "${url_post}")
    if [ "$code" = "200" ] || [ "$code" = "201" ]; then
        say "   PUT snapshot ${col} -> HTTP ${code}"
        return 0
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${url_post}")
    if [ "$code" = "200" ] || [ "$code" = "201" ]; then
        say "   POST snapshot ${col} -> HTTP ${code}"
        return 0
    fi
    say "   WARN snapshot ${col} not created (PUT/POST -> HTTP ${code})"
    return 1
}

native_snap_ok=true
for COL in ganjuur_frames ganjuur_genealogy ganjuur_words; do
    try_create_snapshot "${COL}" || native_snap_ok=false
done

# retrieve each native snapshot to the snapshot dir
for COL in ganjuur_frames ganjuur_genealogy ganjuur_words; do
    # newest snapshot name for this collection
    SNAP="$(curl -fsS "http://127.0.0.1:6333/collections/${COL}/snapshots" \
        | python3 -c 'import sys,json
d=json.loads(sys.stdin.read()).get("result",[])
print(d[-1]["name"] if d else "")' 2>/dev/null)"
    if [ -n "${SNAP}" ]; then
        OUT="${QSNAP_DIR}/${COL}_${SNAP}.snapshot"
        say "   retrieve ${COL}/${SNAP}"
        curl -fsS "http://127.0.0.1:6333/collections/${COL}/snapshots/${SNAP}" -o "${OUT}" >>"${LOG}" 2>&1 \
          && say "      -> $(basename ${OUT}) ($(du -h ${OUT} | cut -f1))" \
          || say "      WARN retrieve ${COL} failed"
    else
        say "   WARN no snapshot name for ${COL}"
    fi
done

# ================================================================
# 6. Qdrant = full storage dir   <-- bulletproof portable fallback
#    done via BRIEF container STOP + root rsync so every Docker-owned
#    file (payload_index/*.bin, etc.) is copied. Downtime ~ 5-15s.
#    The Gradio app's qdrant client retries, so a brief pause is fine.
# ================================================================
say "-- qdrant full storage copy (brief container stop as root)"
FULL="${QSNAP_DIR}/full_qdrant_storage"
mkdir -p "${FULL}"
if docker ps --format '{{.Names}}' | grep -q '^qdrant_ganjuur$'; then
    say "   stopping qdrant_ganjuur briefly to copy storage as root..."
    echo "$PASS" | sudo -S systemctl stop ganjuur.service >/dev/null 2>&1 || true  # release client
    echo "$PASS" | sudo -S docker stop qdrant_ganjuur >>"${LOG}" 2>&1 || true
    sleep 2
    say "   rsync as root (this copies Docker-owned files)..."
    echo "$PASS" | sudo -S rsync -a --delete "${SRC}/qdrant_storage/" "${FULL}/" >>"${LOG}" 2>&1 \
      && say "   root rsync OK -> $(du -h ${FULL} | cut -f1)" \
      || say "   WARN root rsync had errors (see log)"
    say "   restarting qdrant_ganjuur..."
    echo "$PASS" | sudo -S docker start qdrant_ganjuur >>"${LOG}" 2>&1 || true
    sleep 3
    echo "$PASS" | sudo -S systemctl start ganjuur.service >/dev/null 2>&1 || true
else
    say "   WARN qdrant_ganjuur container not running; fall back to trinity-level rsync (may miss Docker-owned files)"
    rsync -a --delete "${SRC}/qdrant_storage/" "${FULL}/" >>"${LOG}" 2>&1 || true
fi

# ================================================================
# 7. config + README + MANIFEST  (auto-generate so each snapshot is
#                                   self-documenting + deployable)
# ================================================================
say "-- requirements.txt, config, README, MANIFEST"

# requirements.txt from live venv
"${SRC}/ganjuur_env/bin/pip" freeze > "${TMP}/config/requirements.txt" 2>/dev/null || true

cat > "${TMP}/config/.env.template" << 'DOTENV'
# Copy to .env and fill in real values. Never commit the real .env.
GANJUUR_USER=
GANJUUR_PASS=
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
GRADIO_ANALYTICS_ENABLED=false
DOTENV

cat > "${TMP}/config/docker-compose.yml" << 'COMPOSE'
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
COMPOSE

cat > "${TMP}/config/ganjuur.service" << 'SERVICE'
[Unit]
Description=Ganjuur web service
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=trinity
Group=trinity
WorkingDirectory={{GANJUUR_HOME}}
EnvironmentFile={{GANJUUR_HOME}}/.env
Environment=PYTHONUNBUFFERED=1
ExecStartPre=/bin/bash -c 'for i in $(seq 1 30); do /usr/bin/curl -fsS http://127.0.0.1:6333/collections >/dev/null && exit 0; sleep 2; done; exit 1'
ExecStart={{GANJUUR_VENV}} -u {{GANJUUR_HOME}}/longcat.py
Restart=always
RestartSec=5
TimeoutStartSec=90
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
SERVICE

# MANIFEST
FRAMES="$(find "${TMP}/data/ganjuur_crops/db_frames" -type f 2>/dev/null | wc -l)"
QDRANT_SNAP="$(ls -la ${QSNAP_DIR}/*.snapshot 2>/dev/null | wc -l) native snapshots"
FULLSZ="$(du -sh ${FULL} 2>/dev/null | cut -f1)"
cat > "${TMP}/MANIFEST.txt" << EOF
Ganjuur daily migration snapshot
Snapshot:   snapshot_${D_TS}
Generated:  $(date -R)
Source host: $(hostname)  ($SRC)
Ingest source: BDRC IIIF (Lokesh Chandra Collection, bdr:MW4CZ5370)
Qdrant (live):
  ganjuur_frames:     $(qdcount ganjuur_frames) points
  ganjuur_genealogy:  $(qdcount ganjuur_genealogy) points
  ganjuur_words:      $(qdcount ganjuur_words) points
Crop frames:  ${FRAMES}
Qdrant artifacts: ${QDRANT_SNAP}  | full storage: ${FULLSZ}
This snapshot is a self-contained, deployable Ganjuur deployment.
See README.md for restore instructions.
EOF

cat > "${TMP}/README.md" << 'README'
# Ganjuur — Migration-ready daily snapshot

Self-contained deployable copy of the Ganjuur Digital Humanities service
(Mongolian woodblock manuscript visual-similarity search).
Produced daily from the live server. ReCrop frames (webp, sharded) come from
BDRC IIIF (Lokesh Chandra Collection, bdr:MW4CZ5370); Qdrant vectors stored.

## Layout
| Path | Purpose |
|------|---------|
| `longcat.py` | system entrypoint (systemd runs this) |
| `gpt.py` / `app*.py` / `ingest_from_bdrc.py` / `anomaly_worker.py` | application core |
| `ARCHITECTURE.md` / `GANJUUR_PROJECT_HANDOFF.md` | docs |
| `asset/` | favicon / logo |
| `scripts/` | helpers |
| `data/ganjuur_crops/db_frames/` | frame crops (webp, sharded) — runtime image source |
| `data/scans/` , `data/ganjuur_scans/` | source scan images |
| `transliterations.db` (+ `.json`) | transliteration DB |
| `ganjuur_gariin_avlaga.html`, `image_566068.jpg` | static pages/images |
| `qdrant_snapshot/*.snapshot` | Qdrant native snapshots (portable) |
| `qdrant_snapshot/full_qdrant_storage/` | full Qdrant dir (bulletproof fallback) |
| `config/*.requirements.txt / *.template / *.yml / *.service` | deployment helpers |

## Restore (fresh Ubuntu 24.04 host)
1. Copy this whole dir to GANJUUR_HOME (e.g. `/home/<user>/ganjuur`).
2. `python3 -m venv ganjuur_env && ganjuur_env/bin/pip install -r config/requirements.txt`
3. `cp config/.env.template .env && nano .env` — fill real secrets.
4. `sudo cp config/ganjuur.service /etc/systemd/system/ganjuur.service`
   and replace `{{GANJUUR_HOME}}` → home dir, `{{GANJUUR_VENV}}` → `<home>/ganjuur_env/bin/python`
5. Qdrant data (choose one):
   a. **native snapshot** (preferred; small + portable). After Qdrant container
      is healthy, for each COL in ganjuur_frames ganjuur_genealogy ganjuur_words:
        curl -X PUT "http://127.0.0.1:6333/collections/<COL>/snapshot" \
          -H 'Content-Type: application/json' \
          -d '{"snapshot": "/qdrant/storage/qdrant_snapshot/<COL>_<name>.snapshot"}'
   b. **full storage** (robust): mount `qdrant_snapshot/full_qdrant_storage/`
      at `/qdrant/storage` when starting the Qdrant container.
6. `docker compose -f config/docker-compose.yml up -d`
7. `sudo systemctl enable --now ganjuur.service`

## Runtime reqs
- NVIDIA GPU with >=6 GB VRAM for DINOv2 CUDA embedder (RTX 3060 tested)
- Docker (Qdrant CPU; GPU used only by app)
- ~25 GB disk (crops ~19G + Qdrant + code + scans)
README

# ================================================================
# 8. promote snapshot1 = hardlink clone of finished archive
# ================================================================
say "-- promoting snapshot1"
# archive FIRST (so the kept archive is the source of truth), then hardlink to snapshot1
mv "${TMP}" "${BASE}/${DATED}"
cp -al "${BASE}/${DATED}" "${BASE}/${LATEST}"       # hardlink clone for cheap daily swap
say "[$(date '+%H:%M:%S')] snapshot archive: ${BASE}/${DATED}"
say "[$(date '+%H:%M:%S')] snapshot1 (latest): ${BASE}/${LATEST}  size=$(du -sh ${BASE}/${LATEST} | cut -f1)"

# ================================================================
# 9. prune archives older than KEEP days
# ================================================================
say "-- pruning archives older than ${KEEP} days"
ls -1d "${BASE}"/snapshots/snapshot_* 2>/dev/null \
    | sort -r | tail -n +"$((KEEP + 1))" | xargs -r rm -rf
ARCH="$(ls -1d "${BASE}"/snapshots/snapshot_* 2>/dev/null | wc -l)"

# ================================================================
# 10. done
# ================================================================
SIZE="$(du -sh "${BASE}/${LATEST}" | cut -f1)"
say "=== snapshot complete === snapshot1=${SIZE}  archives=${ARCH}  log=${LOG}"
