# =============================================================================
# Ganjuur — centralized configuration
# =============================================================================
# Every path and setting that was previously hardcoded across the app
# entrypoints now lives here with a sensible default and an environment-
# variable override.  When no env var is set, behaviour is identical to the
# old hardcoded values.
#
# To swap out a path, set the corresponding env var.  Example:
#   export GANJUUR_BASE_DIR=/mnt/backup/ganjuur
#   python longcat.py
# =============================================================================

import os
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  Primary paths
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(os.getenv("GANJUUR_BASE_DIR",    "/home/trinity/ganjuur")).expanduser()
SCANS_DIR       = Path(os.getenv("GANJUUR_SCANS_DIR",   BASE_DIR / "data/scans"))
CROPS_DIR       = Path(os.getenv("GANJUUR_CROPS_DIR",   BASE_DIR / "data/ganjuur_crops/db_frames"))
VECTORDB_PATH   = Path(os.getenv("GANJUUR_VECTORDB",    BASE_DIR / "vectordb"))
DB_PATH         = Path(os.getenv("GANJUUR_DB_PATH",     BASE_DIR / "transliterations.db"))
MANUAL_PATH     = Path(os.getenv("GANJUUR_MANUAL",      BASE_DIR / "ganjuur_gariin_avlaga.html"))
LOGO_PATH       = Path(os.getenv("GANJUUR_LOGO",        BASE_DIR / "image_566068.jpg"))
FAVICON_PATH    = Path(os.getenv("GANJUUR_FAVICON",     BASE_DIR / "asset/favicon.png"))
VENV_DIR        = Path(os.getenv("GANJUUR_VENV",        BASE_DIR / "ganjuur_env"))
SCRIPTS_DIR     = Path(os.getenv("GANJUUR_SCRIPTS",     BASE_DIR / "scripts"))

# Fallback local-vectordb path (used when Docker Qdrant is unavailable).
LOCAL_VECTORDB_PATH = Path(os.getenv(
    "GANJUUR_LOCAL_VECTORDB_PATH", str(BASE_DIR / "vectordb")
))


# ══════════════════════════════════════════════════════════════════════════════
#  Migration-ready snapshot / archive
# ══════════════════════════════════════════════════════════════════════════════

PROD_READY_BASE         = Path(os.getenv("GANJUUR_PROD_READY_BASE", "/home/trinity/data/prod_ready_ganjuur"))
PROD_READY_SNAPSHOT1    = PROD_READY_BASE / "snapshot1"

# Sudo password for brief docker stop during snapshot builds.
# Prefer the env var; the inline default preserves backward compatibility.
SUDO_PASS = os.getenv("GANJUUR_SUDO_PASS", "pass#1234")


# ══════════════════════════════════════════════════════════════════════════════
#  Qdrant connection
# ══════════════════════════════════════════════════════════════════════════════

QDRANT_HOST         = os.getenv("QDRANT_HOST",          "localhost")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT",      "6333"))
QDRANT_GRPC_PORT    = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY")
QDRANT_LOCAL_PATH   = os.getenv("QDRANT_LOCAL_PATH")


# ══════════════════════════════════════════════════════════════════════════════
#  Vector / model
# ══════════════════════════════════════════════════════════════════════════════

COLLECTION_NAME     = os.getenv("GANJUUR_COLLECTION",       "ganjuur_frames")
VECTOR_DIM          = int(os.getenv("GANJUUR_VECTOR_DIM",   "768"))
BATCH_SIZE          = int(os.getenv("GANJUUR_BATCH_SIZE",   "32"))
DEFAULT_STRIDE      = int(os.getenv("GANJUUR_DEFAULT_STRIDE", "50"))
DINO_MODEL          = os.getenv("GANJUUR_DINO_MODEL",       "facebook/dinov2-base")


# ══════════════════════════════════════════════════════════════════════════════
#  Safety / resource limits
# ══════════════════════════════════════════════════════════════════════════════

MAX_FILES               = int(os.getenv("GANJUUR_MAX_FILES",       "5000"))
MAX_ARCHIVE_BYTES       = int(os.getenv("GANJUUR_MAX_ARCHIVE_BYTES", str(10 * 1024**3)))
MAX_MEMBER_BYTES        = int(os.getenv("GANJUUR_MAX_MEMBER_BYTES", str(100 * 1024**2)))
MAX_COMPRESSION_RATIO   = float(os.getenv("GANJUUR_MAX_COMPRESSION", "100"))
MAX_IMAGE_PIXELS        = int(os.getenv("GANJUUR_MAX_IMAGE_PIXELS", "100000000"))
MAX_TOTAL_PIXELS        = int(os.getenv("GANJUUR_MAX_TOTAL_PIXELS", "300000000"))
SUPPORTED_EXTENSIONS    = (".jpg", ".jpeg", ".png")
