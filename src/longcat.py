# Монгол Шунхан Ганжуур — нэг файлын програм
#
# Хүснэгтэд ашиглагдах орчны хувьсагчууд:
#   GANJUUR_BASE_DIR=/home/trinity/ganjuur
#   QDRANT_HOST=localhost  QDRANT_PORT=6333  QDRANT_GRPC_PORT=6334
#   GANJUUR_USER=<хэрэглэгч>  GANJUUR_PASS=<нууц үг>

import atexit
import logging
import multiprocessing as mp
import os
import shutil
import sqlite3
import threading
import time
import uuid
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import umap
from qdrant_client import QdrantClient, models
from sklearn.neighbors import LocalOutlierFactor
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"IMPORTANT: You are using gradio version.*", category=UserWarning, module=r"gradio\.analytics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ganjuur")

# =============================================================================
# 0. ЧУХАЛ НЭМЭЛТҮҮД — БҮХ ИМПОРТЫН ӨМНӨ БАЙХ ЁСТОЙ
# =============================================================================
import gradio_client.utils
import gradio.networking

_orig_json_schema_to_python_type = gradio_client.utils._json_schema_to_python_type


def _patched_json_schema_to_python_type(schema, defs=None):
    """Gradio-н Pydantic-v2 boolean JSON schema-д зориулсан нийцтэй байдлын засвар."""
    if isinstance(schema, bool):
        return "any"
    return _orig_json_schema_to_python_type(schema, defs)


_orig_get_type = gradio_client.utils.get_type


def _patched_get_type(schema):
    """_patched_json_schema_to_python_type-т хамт ашиглагдах засвар."""
    if isinstance(schema, bool):
        return "bool"
    return _orig_get_type(schema)


gradio_client.utils._json_schema_to_python_type = _patched_json_schema_to_python_type
gradio_client.utils.get_type = _patched_get_type

# LAN/Nginx суулгалтанд зориулсан шаардлагатай.
gradio.networking.url_ok = lambda url: True

# =============================================================================
# 1. ИМПОРТ, ТОХИРГОО БА ХАДГАЛАХ ТӨВ
# =============================================================================

def load_environment_file() -> None:
    """.env файлаас KEY=VALUE тохиргоог ачаална. Орчны хувьсагчийг дарахгүй."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_environment_file()

BASE_DIR = Path(os.getenv("GANJUUR_BASE_DIR", "/home/trinity/ganjuur")).expanduser()
SCANS_DIR = BASE_DIR / "data" / "scans"
CROPS_DIR = BASE_DIR / "data" / "ganjuur_crops" / "db_frames"
DB_PATH = BASE_DIR / "transliterations.db"

COLLECTION_NAME = "ganjuur_frames"
VECTOR_DIM = 768
BATCH_SIZE = int(os.getenv("GANJUUR_BATCH_SIZE", "32"))
DEFAULT_STRIDE = 50

# Хязгаарлалтууд алдагдуулах ёсгүй. Зураг нээгдэхдэсаа илүү ихсаг RAM эзэлнэ.
MAX_FILES = int(os.getenv("GANJUUR_MAX_FILES", "5000"))
MAX_ARCHIVE_BYTES = int(os.getenv("GANJUUR_MAX_ARCHIVE_BYTES", str(10 * 1024**3)))
MAX_MEMBER_BYTES = int(os.getenv("GANJUUR_MAX_MEMBER_BYTES", str(100 * 1024**2)))
MAX_COMPRESSION_RATIO = float(os.getenv("GANJUUR_MAX_COMPRESSION_RATIO", "100"))
MAX_IMAGE_PIXELS = int(os.getenv("GANJUUR_MAX_IMAGE_PIXELS", "100000000"))
MAX_TOTAL_PIXELS = int(os.getenv("GANJUUR_MAX_TOTAL_PIXELS", "300000000"))
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

LOGO_PATH = Path(os.getenv("GANJUUR_LOGO_PATH", str(BASE_DIR / "image_566068.jpg")))
FAVICON_PATH = Path(os.getenv("GANJUUR_FAVICON_PATH", str(BASE_DIR / "asset" / "favicon.png")))
MANUAL_PATH = BASE_DIR / "ganjuur_gariin_avlaga.html"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH")  # Зөвхөн хөгжүүлэлтийн орчинд.
LOCAL_VECTORDB_PATH = Path(os.getenv("GANJUUR_LOCAL_VECTORDB_PATH", str(BASE_DIR / "vectordb")))

for directory in (SCANS_DIR, CROPS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

db_lock = threading.Lock()
model_lock = threading.Lock()
analytics_process: mp.Process | None = None
analytics_process_lock = threading.Lock()
_model: AutoModel | None = None
_processor: AutoImageProcessor | None = None
qdrant: QdrantClient | None = None
qdrant_backend = "unknown"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_qdrant_client(require_docker: bool = False) -> QdrantClient:
    """Docker Qdrant-ыг ашиглана. Шаардлагатай бол Docker-ыг л шаардана."""
    global qdrant_backend
    if QDRANT_LOCAL_PATH and not require_docker:
        qdrant_backend = "local"
        log.warning("Тохиргооны дагуу локал Qdrant ашиглагдаж байна: %s", QDRANT_LOCAL_PATH)
        return QdrantClient(path=QDRANT_LOCAL_PATH)

    remote = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=False,
        api_key=QDRANT_API_KEY,
        timeout=30,
        check_compatibility=False,
    )
    try:
        remote.get_collections()
        qdrant_backend = "docker"
        log.info("Docker Qdrant холбогдлоо: %s:%s", QDRANT_HOST, QDRANT_PORT)
        return remote
    except Exception as remote_error:
        remote.close()
        if require_docker:
            raise RuntimeError(
                f"Docker Qdrant шаардлагатай боловч боломжгүй байна: {remote_error}"
            ) from remote_error
        if LOCAL_VECTORDB_PATH.exists():
            qdrant_backend = "local"
            log.warning(
                "Docker Qdrant боломжгүй (%s). Локал санг ашиглаж байна: %s",
                remote_error,
                LOCAL_VECTORDB_PATH,
            )
            return QdrantClient(path=str(LOCAL_VECTORDB_PATH))
        raise RuntimeError(
            f"Docker Qdrant {QDRANT_HOST}:{QDRANT_PORT} боломжгүй, "
            f"локал санд мөн боломжгүй. Өөрчлэлт хийгдсэнгүй."
        ) from remote_error


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
        log.info("Qdrant цуглуулга үүсгэлээ: %s", COLLECTION_NAME)

    for field, schema in (
        ("umap_x", models.PayloadSchemaType.FLOAT),
        ("anomaly_score", models.PayloadSchemaType.FLOAT),
        ("status", models.PayloadSchemaType.KEYWORD),
        ("source", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(COLLECTION_NAME, field, field_schema=schema)
        except Exception as exc:
            log.info("Пэйлоод индэс өөрчлөгдөөгүй: %s: %s", field, exc)


def db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA busy_timeout=5000;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def init_db() -> None:
    with db_lock, db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transliterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_image TEXT NOT NULL,
                text_content TEXT NOT NULL,
                token_count INTEGER,
                char_count INTEGER,
                status TEXT NOT NULL DEFAULT 'PENDING',
                user_id TEXT DEFAULT 'trinity',
                version INTEGER DEFAULT 1
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_source ON transliterations(source_image)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transliterations(timestamp DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_status ON transliterations(status)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                backend TEXT NOT NULL DEFAULT 'docker',
                started_at TEXT NOT NULL,
                finished_at TEXT
            )
            """
        )
        # Defensive migration: older databases created by gpt.py lack the
        # backend column. Add it if missing so analytics_status() works.
        existing_cols = {row[1] for row in connection.execute("PRAGMA table_info(analytics_jobs)")}
        if "backend" not in existing_cols:
            connection.execute("ALTER TABLE analytics_jobs ADD COLUMN backend TEXT NOT NULL DEFAULT 'docker'")


def update_analytics_job(job_id: str, status: str, message: str, backend: str = "docker", finished: bool = False) -> None:
    with db_lock, db_connection() as connection:
        connection.execute(
            "UPDATE analytics_jobs SET status=?, message=?, backend=?, finished_at=? WHERE id=?",
            (status, message, backend, utc_now() if finished else None, job_id),
        )


def latest_analytics_job() -> tuple[str, str, str] | None:
    with db_lock, db_connection() as connection:
        row = connection.execute(
            "SELECT status, message, backend FROM analytics_jobs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return row


# =============================================================================
# 2. ЗАГВАР БА ЗУРГЫН ТУСЛАХ ФУНКЦҮҮД
# =============================================================================
def get_model_and_processor() -> tuple[AutoModel, AutoImageProcessor]:
    """Загварыг анхны удаад ачаална. Аналитик ажилч CUDA-г ачаалдаггүй."""
    global _model, _processor
    with model_lock:
        if _model is None or _processor is None:
            log.info("DINOv2-base ачааллаж байна: %s", DEVICE.upper())
            _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            _model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE)
            _model.eval()
    return _model, _processor


def get_embeddings_batch(img_bgr_list: list[np.ndarray]) -> list[list[float]]:
    if not img_bgr_list:
        return []
    model, processor = get_model_and_processor()
    rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in img_bgr_list]
    with model_lock, torch.inference_mode():
        inputs = processor(images=rgb_images, return_tensors="pt").to(DEVICE)
        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**inputs).last_hidden_state[:, 0, :]
        else:
            output = model(**inputs).last_hidden_state[:, 0, :]
        output = torch.nn.functional.normalize(output, p=2, dim=-1)
    return output.cpu().numpy().tolist()


def process_manuscript_image(image_bgr: np.ndarray) -> np.ndarray:
    """Ногоон сувгийн CLAHE руу тодорхойлсон өмнөх арга."""
    green_channel = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(green_channel)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def save_frame_sharded(frame: np.ndarray, page_id: str, frame_index: int) -> tuple[str, str]:
    frame_id = str(uuid.uuid4())
    shard = CROPS_DIR / frame_id[:2] / frame_id[2:4]
    shard.mkdir(parents=True, exist_ok=True)
    # Sanitize page_id so that characters like ':' or '/' (e.g. from BDRC
    # resource IDs like "bdrc:MW4CZ5370/vol089/p0023") don't end up as path
    # separators in the filename.
    safe_page_id = page_id.replace(":", "_").replace("/", "_")
    filename = f"{frame_id}_p{safe_page_id}_f{frame_index:06d}.webp"
    full_path = shard / filename
    ok = cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_WEBP_QUALITY, 85])
    if not ok:
        raise IOError(f"Хүсэгтэй зураг бичих боломжгүй: {full_path}")
    return str(full_path), frame_id


def remove_paths(paths: Iterable[str]) -> None:
    for path in paths:
        try:
            candidate = Path(path).resolve()
            crops_root = CROPS_DIR.resolve()
            if crops_root in candidate.parents:
                candidate.unlink(missing_ok=True)
        except OSError:
            log.warning("Дутуу хүсгийг устгаж чадсангүй: %s", path)


def flush_batch(crops: list[np.ndarray], paths: list[str], ids: list[str], page_id: str, extra_payload: dict | None = None) -> int:
    """Нэг багц оруулах; алдаатай багцын зургуудыг л устгана.
    extra_payload: supplementary fields merged into every point (e.g. BDRC attribution URL)."""
    if not crops:
        return 0
    try:
        vectors = get_embeddings_batch(crops)
        points = [
            models.PointStruct(
                id=frame_id,
                vector=vector,
                payload={
                    "source": page_id,
                    "local_path": path,
                    "status": "PENDING",
                    "embedding_model": "facebook/dinov2-base",
                    "preprocessing": "green-channel-clahe-v1",
                    "ingested_at": utc_now(),
                    **(extra_payload or {}),
                },
            )
            for frame_id, path, vector in zip(ids, paths, vectors)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return len(points)
    except Exception as exc:
        remove_paths(paths)
        log.exception("Оруулах/upsert алдаа %s: %s", page_id, exc)
        raise RuntimeError(f"Хуудас индекслэхэд алдаа гарлаа '{page_id}': {exc}") from exc


# =============================================================================
# 3. ОРУУЛАХ БА ХАЙХ
# =============================================================================
def uploaded_path(upload: Any) -> Path | None:
    if upload is None:
        return None
    value = getattr(upload, "name", upload)
    return Path(str(value))


def safe_image_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [info for info in archive.infolist() if Path(info.filename).suffix.lower() in SUPPORTED_EXTENSIONS]
    if not members:
        raise ValueError("ZIP файлд JPG эсвэл PNG зураг байхгүй байна.")
    if len(members) > MAX_FILES:
        raise ValueError(f"ZIP файлд {len(members):,} зураг байна; хязгаар {MAX_FILES:,}.")
    uncompressed = sum(info.file_size for info in members)
    if uncompressed > MAX_ARCHIVE_BYTES:
        raise ValueError("Нээгдсэн зургийн хэмжээ серверийн хязгаараас хэтэрсэн байна.")
    for info in members:
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"'{Path(info.filename).name}' нээгдэхэд хэт том байна.")
        if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise ValueError(f"'{Path(info.filename).name}' хэлбэржүүлэлт аюултай байна.")
    return sorted(members, key=lambda info: info.filename.lower())


def handle_zip_ingestion(zip_file_obj: Any, stride: int, progress=gr.Progress()) -> str:
    path = uploaded_path(zip_file_obj)
    if path is None or not path.exists():
        return "❌ ZIP файл оруулна уу."
    if path.suffix.lower() != ".zip":
        return "❌ .zip өргөтгөлтэй файл оруулна уу."
    if stride <= 0:
        return "❌ Алхам эерэг тоо байх ёстой."

    try:
        progress(0.02, desc="Архивыг шалгаж байна…")
        with zipfile.ZipFile(path, "r") as archive:
            members = safe_image_members(archive)
            total_pixels = 0
            total_frames = 0
            indexed_frames = 0
            skipped: list[str] = []

            for image_number, member in enumerate(members, start=1):
                progress(0.03 + 0.97 * (image_number - 1) / len(members), desc=f"Боловсруулж байна {image_number}/{len(members)}: {Path(member.filename).name}")
                try:
                    with archive.open(member) as handle:
                        raw = np.frombuffer(handle.read(), dtype=np.uint8)
                    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        skipped.append(f"{Path(member.filename).name} (тайгдаагүй)")
                        continue
                    height, width = image_bgr.shape[:2]
                    image_pixels = height * width
                    if image_pixels > MAX_IMAGE_PIXELS:
                        skipped.append(f"{Path(member.filename).name} (хэт том)")
                        continue
                    total_pixels += image_pixels
                    if total_pixels > MAX_TOTAL_PIXELS:
                        return "❌ Боловсруулах зогссон: нийт зургийн хэмжээ хязгаараас хэтэрсэн."

                    page_id = Path(member.filename).stem
                    processed = process_manuscript_image(image_bgr)
                    h, w = processed.shape[:2]
                    window = min(h, w)
                    axis_is_x = w >= h
                    max_dimension = w if axis_is_x else h

                    crops: list[np.ndarray] = []
                    paths: list[str] = []
                    ids: list[str] = []
                    page_frames = 0
                    for position in range(0, max_dimension - window + 1, int(stride)):
                        crop = processed[:, position:position + window] if axis_is_x else processed[position:position + window, :]
                        frame_path, frame_id = save_frame_sharded(crop, page_id, page_frames)
                        crops.append(crop)
                        paths.append(frame_path)
                        ids.append(frame_id)
                        page_frames += 1
                        if len(crops) >= BATCH_SIZE:
                            indexed_frames += flush_batch(crops, paths, ids, page_id)
                            crops, paths, ids = [], [], []
                    if crops:
                        indexed_frames += flush_batch(crops, paths, ids, page_id)
                    total_frames += page_frames
                except Exception as exc:
                    log.exception("Амжилтгүй болсон %s", member.filename)
                    skipped.append(f"{Path(member.filename).name} ({exc})")

                del raw, image_bgr

        progress(1.0, desc="Дууссан")
        message = [
            "## ✅ Оруулаж дууссан",
            f"- **Уншиж авсан хуудас:** {len(members):,}",
            f"- **Үүсгэсэн хүсэг:** {total_frames:,}",
            f"- **Индэслэсэн хүсэг:** {indexed_frames:,}",
            f"- **Алхам:** {stride}px",
        ]
        if skipped:
            preview = "; ".join(skipped[:5])
            message.append(f"- **Алгассан:** {len(skipped)} — {preview}")
        return "\n".join(message)
    except zipfile.BadZipFile:
        return "❌ Энэ файл хүчинтэй ZIP файл биш байна."
    except Exception as exc:
        log.exception("ZIP оруулах амжилтгүй")
        return f"❌ Оруулаж чадсангүй: {exc}"


def execute_visual_query(image_rgb: np.ndarray, top_k: int) -> list[tuple[str, str]]:
    if image_rgb is None:
        return []
    try:
        query_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        vector = get_embeddings_batch([query_image])[0]
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=int(top_k),
            with_payload=["source", "local_path"],
        )
        results: list[tuple[str, str]] = []
        for point in response.points:
            payload = point.payload or {}
            frame_path = payload.get("local_path")
            if frame_path and Path(frame_path).exists():
                results.append((str(frame_path), f"{point.score * 100:.1f}% таарч байна · {payload.get('source', 'Unknown source')}"))
        return results
    except Exception as exc:
        log.exception("Хайлт амжилтгүй")
        raise gr.Error(f"Хайлт амжилтгүй боллоо: {exc}") from exc


# =============================================================================
# 4. АНАЛИТИК — ТУСДАА ПРОЦЕСС, ХАДГАЛСАН АЖЛЫН БАЙДАЛ
# =============================================================================
def scroll_all_vectors(client: QdrantClient) -> tuple[list[Any], np.ndarray]:
    ids: list[Any] = []
    vectors: list[list[float]] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5_000,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )
        if not records:
            break
        ids.extend(record.id for record in records)
        vectors.extend(record.vector for record in records)
        if offset is None:
            break
    return ids, np.asarray(vectors, dtype=np.float32)


def analytics_worker(job_id: str) -> None:
    """Тусдаа ажиллана; DINOv2/CUDA-г ачаалдаггүй. Зөвхөн Docker Qdrant."""
    client = make_qdrant_client(require_docker=True)
    try:
        update_analytics_job(job_id, "RUNNING", "Qdrant-аас вектор уншиж байна…", backend="docker")
        ids, vectors = scroll_all_vectors(client)
        if len(ids) < 3:
            update_analytics_job(job_id, "FAILED", "Аналитикт хамгийн багадаа 3 индэслэсэн хүсэг шаардлагатай.", backend="docker", finished=True)
            return

        update_analytics_job(job_id, "RUNNING", f"{len(ids):,} хүсэгт UMAP болон онцгой оноог тооцоолж байна…", backend="docker")
        umap_neighbors = min(15, len(ids) - 1)
        lof_neighbors = min(20, len(ids) - 1)
        reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.1, metric="cosine", random_state=42)
        coordinates = reducer.fit_transform(vectors)
        lof = LocalOutlierFactor(n_neighbors=lof_neighbors, metric="cosine")
        lof.fit_predict(vectors)
        scores = -lof.negative_outlier_factor_
        score_range = scores.max() - scores.min()
        normalized = (scores - scores.min()) / score_range if score_range > 0 else np.zeros_like(scores)

        update_analytics_job(job_id, "RUNNING", "Үр дүнг Qdrant-д хадгалж байна…", backend="docker")
        for start in range(0, len(ids), 500):
            points = [
                models.PointStruct(
                    id=ids[index],
                    vector=None,
                    payload={
                        "umap_x": float(coordinates[index][0]),
                        "umap_y": float(coordinates[index][1]),
                        "anomaly_score": float(normalized[index]),
                    },
                )
                for index in range(start, min(start + 500, len(ids)))
            ]
            client.upsert(COLLECTION_NAME, points=points, wait=True, update_vectors=False)
        update_analytics_job(job_id, "COMPLETE", f"{len(ids):,} хүсэг дээр аналитик дууссан. Газрын зургийг шинэчилнэ үү.", backend="docker", finished=True)
    except Exception as exc:
        log.exception("Аналитик ажилч алдаа гаргалаа")
        update_analytics_job(job_id, "FAILED", f"Аналитик амжилтгүй: {exc}", backend="docker", finished=True)
    finally:
        client.close()


def run_analytics() -> str:
    global analytics_process
    if qdrant_backend != "docker":
        return "⚠️ Тооллыг түр зогсоосон байна. Олон хэрэглэгчтэй үед локал санг зэрэг ашиглахаас сэргийлж Docker Qdrant шаардлагатай."
    with analytics_process_lock:
        if analytics_process is not None and analytics_process.is_alive():
            return "⏳ Аналитик ажиллаж байна. **Явцыг шалгах**-аас харна уу."
        job_id = str(uuid.uuid4())
        with db_lock, db_connection() as connection:
            connection.execute(
                "INSERT INTO analytics_jobs (id, status, message, backend, started_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, "QUEUED", "Аналитик ажил эхлэхээ хүлээж байна…", "docker", utc_now()),
            )
        analytics_process = mp.get_context("spawn").Process(target=analytics_worker, args=(job_id,), daemon=False)
        analytics_process.start()
    return "🧭 Аналитик тусдаа ажил болж эхэллээ. Та хайлт болон галиг оруулах ажиллагааг үргэлжлүүлж болно."


def analytics_status() -> str:
    job = latest_analytics_job()
    if job is None:
        return "*Аналитик ажил хараахан эхлээгүй байна.*"
    status, message, backend = job
    icon = {"QUEUED": "🕒", "RUNNING": "🧭", "COMPLETE": "✅", "FAILED": "❌"}.get(status, "ℹ️")
    backend_label = "Docker Qdrant" if backend == "docker" else "локал сан"
    return f"{icon} **{status.title()}** — {message}\n\n_Ашигласан сангийн төрөл: {backend_label}_"


def scroll_all_analytics_points() -> list[Any]:
    points: list[Any] = []
    offset = None
    while True:
        records, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="umap_x", range=models.Range(gte=-999999.0))]),
            limit=5_000,
            with_payload=["umap_x", "umap_y", "anomaly_score", "source"],
            with_vectors=False,
        )
        points.extend(records)
        if not records or offset is None:
            return points


def generate_plot() -> tuple[go.Figure, gr.Dropdown]:
    empty = go.Figure().update_layout(
        title="Эхлээд аналитик ажиллуулаад газрын зургийг шинэчилнэ үү.",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    try:
        records = scroll_all_analytics_points()
        if not records:
            return empty, gr.update(choices=[], value=None)
        x_values, y_values, scores, hover, anomalies = [], [], [], [], []
        for record in records:
            payload = record.payload or {}
            if "umap_x" not in payload or "umap_y" not in payload:
                continue
            score = float(payload.get("anomaly_score", 0.0))
            x_values.append(payload["umap_x"])
            y_values.append(payload["umap_y"])
            scores.append(score)
            hover.append(f"Эх сурвалж: {payload.get('source', 'Unknown')}<br>Онцгой оноо: {score:.3f}")
            if score > 0.7:
                anomalies.append(record)
        if not x_values:
            return empty, gr.update(choices=[], value=None)
        figure = go.Figure()
        figure.add_trace(go.Scattergl(
            x=x_values, y=y_values, mode="markers",
            marker=dict(size=[10 if score > 0.7 else 4 for score in scores], color=scores, colorscale="Viridis", showscale=True, opacity=0.85, colorbar=dict(title="Онцгой")),
            text=hover, hoverinfo="text", name="Хүсгүүд",
        ))
        figure.add_trace(go.Scattergl(
            x=[record.payload["umap_x"] for record in anomalies],
            y=[record.payload["umap_y"] for record in anomalies],
            mode="markers", marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=2)),
            hoverinfo="skip", name="Өндөр онцгой (>0.7)",
        ))
        figure.update_layout(template="plotly_dark", title="Чанарын хяналтын газрын зураг", height=500, margin=dict(l=20, r=20, t=45, b=20), paper_bgcolor="rgba(0,0,0,0)")
        choices = [(f"Оноо {record.payload.get('anomaly_score', 0):.3f} · {record.payload.get('source', 'Unknown')}", str(record.id)) for record in anomalies]
        return figure, gr.update(choices=choices, value=choices[0][1] if choices else None)
    except Exception as exc:
        log.exception("Газрын зураг бүтээж чадсангүй")
        raise gr.Error(f"Газрын зургийг шинэчилж чадсангүй: {exc}") from exc


def inspect_node(node_id: str) -> tuple[str | None, str, go.Figure]:
    if not node_id:
        return None, "Өндөр оноотой цэгийг сонгоно уу.", go.Figure()
    try:
        records = qdrant.retrieve(COLLECTION_NAME, ids=[node_id], with_payload=True, with_vectors=True)
        if not records:
            return None, "❌ Энэ цэг боломжгүй болсон байна.", go.Figure()
        record = records[0]
        payload = record.payload or {}
        score = float(payload.get("anomaly_score", 0.0))
        level = "🔴 Өндөр" if score > 0.7 else "🟡 Дунд" if score > 0.4 else "🟢 Бага"
        metadata = (
            f"### {level} онцгой\n\n"
            f"| Талбар | Утга |\n| --- | --- |\n"
            f"| Эх сурвалж | `{payload.get('source', 'Unknown')}` |\n"
            f"| Оноо | `{score:.4f}` |\n"
            f"| Координат | `({payload.get('umap_x', 0):.2f}, {payload.get('umap_y', 0):.2f})` |"
        )
        values = (record.vector or [])[:64]
        embedding_plot = go.Figure(go.Bar(y=values, marker=dict(color=values, colorscale="RdBu", cmid=0)))
        embedding_plot.update_layout(title="Векторын товч (эхний 64)", template="plotly_dark", height=220, margin=dict(l=10, r=10, t=35, b=10), xaxis=dict(showticklabels=False), paper_bgcolor="rgba(0,0,0,0)")
        path = payload.get("local_path")
        return path if path and Path(path).exists() else None, metadata, embedding_plot
    except Exception as exc:
        log.exception("Цэгийн үзлэг амжилтгүй")
        return None, f"❌ Энэ цэгийг шалгаж чадсангүй: {exc}", go.Figure()


# =============================================================================
# 5. ГАЛИГ — ЯГ ТЭНЦҮҮ ХАДГАЛАХ
# =============================================================================
def validate_exact_text(text: str) -> str:
    """Зөвхөн баталгаажуулна. Текстийг огт өөрчилдөггүй."""
    if text is None or text == "":
        raise ValueError("Хадгалахаасаа өмнө галиг оруулна уу.")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Энэ текст UTF-8-д илэрч чадахгүй байна.") from exc
    return text


def token_count_for_reporting(text: str) -> int:
    # Зөвхөн мэдээллийн тоолол; text_content өөрчлөгдөхгүй.
    return len(text.split())


def commit_transliteration(text: str, image_path: str | None) -> str:
    try:
        exact_text = validate_exact_text(text)
    except ValueError as exc:
        return f"⚠️ {exc}"
    source = image_path or "Unknown source"
    try:
        with db_lock, db_connection() as connection:
            connection.execute(
                """
                INSERT INTO transliterations (timestamp, source_image, text_content, token_count, char_count, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (utc_now(), source, exact_text, token_count_for_reporting(exact_text), len(exact_text), "VERIFIED"),
            )
            saved = connection.execute(
                "SELECT text_content FROM transliterations ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
        if saved != exact_text:
            raise RuntimeError("Текст эргэн шалгагдсангүй; хадгалсан гэж үзэх аргагүй.")
        return (
            "## ✅ Галиг хадгалсан\n"
            f"- **Үг (мэдээллийн тоо):** {token_count_for_reporting(exact_text)}\n"
            f"- **Тэмдэгт:** {len(exact_text)}\n"
            f"- **Эх сурвалж:** {Path(source).name if source != 'Unknown source' else source}\n"
            "- **Бүрэн бүтэн:** яг хэвээр SQLite-д хадгалсан"
        )
    except Exception as exc:
        log.exception("Галиг хадгалж чадсангүй")
        return f"❌ Галигийг хадгалж чадсангүй: {exc}"


def get_recent_transliterations() -> list[tuple[Any, ...]]:
    try:
        with db_lock, db_connection() as connection:
            return connection.execute(
                "SELECT timestamp, source_image, text_content, char_count FROM transliterations ORDER BY id DESC LIMIT 5"
            ).fetchall()
    except Exception as exc:
        log.exception("Сүүлийн галиг уншиж чадсангүй")
        return []


def append_suffix(text: str, suffix: str) -> str:
    return (text or "") + suffix


# =============================================================================
# 6. ХЭРЭГЛЭГЧИЙН ХАРИЛЦАХ БҮС
# =============================================================================
CSS = """
.gradio-container { max-width: 1240px !important; padding: 1.25rem !important; }
.header { background: linear-gradient(135deg, #25324d, #5f3f2f); border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; align-items: center !important; }
.header h1 { color: white !important; margin: 0 !important; font-size: 1.7rem !important; }
.header p { color: #f5e6d3 !important; margin: .35rem 0 0 !important; font-size: .98rem !important; }
.header-logo img { border-radius: 10px; object-fit: cover; }
footer { display: none !important; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Монгол Шунхан Ганжуур",
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="slate"),
        css=CSS,
    ) as app:
        with gr.Row(elem_classes=["header"]):
            if LOGO_PATH.exists():
                gr.Image(
                    value=str(LOGO_PATH),
                    show_label=False,
                    container=False,
                    interactive=False,
                    width=78,
                    height=78,
                    elem_classes=["header-logo"],
                )
            with gr.Column():
                gr.Markdown("# Монгол Шунхан Ганжуур")
                gr.Markdown("Эртний судрын дүрсийг хайж, онцгой хэлбэрийг олж, галигийг хадгалах сан.")

        gr.HTML(
            f'<div style="text-align:center; margin-bottom:1rem;">'
            f'<a href="/file={MANUAL_PATH}" target="_blank" style="font-size:1rem; font-weight:600; text-decoration:none; color:#7a3e20;">'
            f'📖 Дэлгэрэнгүй гарын авлага нээх</a></div>'
        )

        with gr.Tabs():
            with gr.Tab("1 · Зураг оруулах"):
                with gr.Row():
                    with gr.Column():
                        zip_input = gr.File(label="Зургийн архив (.zip)", file_types=[".zip"])
                        stride_input = gr.Slider(10, 300, value=DEFAULT_STRIDE, step=10,
                                                 label="Хэсэглэх алхам (пиксел)",
                                                 info="50 пикселээр эхлэхийг зөвлөж байна.")
                        ingest_button = gr.Button("Зургийг сан руу оруулах", variant="primary", size="lg")
                    with gr.Column():
                        ingest_status = gr.Markdown()

            with gr.Tab("2 · Дүрсээр хайх"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.Image(type="numpy", label="Хайх зургийн хэсэг", height=300)
                        top_k = gr.Slider(1, 30, value=8, step=1, label="Харуулах үр дүнгийн тоо")
                        search_button = gr.Button("Ижил дүрс хайх", variant="primary")
                    with gr.Column(scale=3):
                        search_gallery = gr.Gallery(label="Ижил төстэй олдворууд", columns=4, height=430, object_fit="contain")

            with gr.Tab("3 · Онцгой хэлбэр"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analytics_button = gr.Button("Онцгой хэлбэрийг тооцоолох", variant="secondary")
                        check_analytics_button = gr.Button("Явцыг шалгах")
                        refresh_map_button = gr.Button("Газрын зургийг шинэчлэх")
                        analytics_output = gr.Markdown(value=analytics_status())
                    with gr.Column(scale=3):
                        quality_plot = gr.Plot(label="Дүрсийн тархалтын зураг")
                gr.Markdown("### Онцгой олдворыг үзэх")
                with gr.Row():
                    with gr.Column(scale=1):
                        node_dropdown = gr.Dropdown(label="Сонгох онцгой хэсэг", choices=[], interactive=True)
                    with gr.Column(scale=2):
                        detail_image = gr.Image(label="Сонгосон зураг", type="filepath", height=240)
                        detail_metadata = gr.Markdown()
                        send_to_transcription = gr.Button("Галиг оруулах хэсэг рүү илгээх")
                    with gr.Column(scale=2):
                        detail_embedding = gr.Plot(label="Дүрсийн товч харьцуулалт")

            with gr.Tab("4 · Галиг оруулах"):
                with gr.Row():
                    with gr.Column():
                        transcription_image = gr.Image(label="Эх зургийн хэсэг", type="filepath", interactive=True, height=360)
                    with gr.Column():
                        transcription_text = gr.Textbox(label="Галиг текст", lines=11, placeholder="Кирилл галигийг оруулна уу.")
                        with gr.Row():
                            suffix_bugd = gr.Button("бөгөөд", size="sm")
                            suffix_ajguu = gr.Button("ажгуу", size="sm")
                            suffix_mun = gr.Button("мөн", size="sm")
                            suffix_ted = gr.Button("тэд", size="sm")
                        save_transcription = gr.Button("Галигийг хадгалах", variant="primary", size="lg")
                        save_status = gr.Markdown()
                recent_table = gr.Dataframe(
                    headers=["Хадгалсан цаг", "Эх зураг", "Галиг текст", "Тэмдэгтийн тоо"],
                    value=get_recent_transliterations(),
                    interactive=False,
                    height=220,
                )

        ingest_button.click(handle_zip_ingestion, [zip_input, stride_input], ingest_status)
        search_button.click(execute_visual_query, [query_image, top_k], search_gallery)
        analytics_button.click(run_analytics, outputs=analytics_output)
        check_analytics_button.click(analytics_status, outputs=analytics_output)
        refresh_map_button.click(generate_plot, outputs=[quality_plot, node_dropdown])
        node_dropdown.change(inspect_node, node_dropdown, [detail_image, detail_metadata, detail_embedding])
        send_to_transcription.click(lambda path: path, detail_image, transcription_image)
        suffix_bugd.click(lambda value: append_suffix(value, "бөгөөд "), transcription_text, transcription_text)
        suffix_ajguu.click(lambda value: append_suffix(value, "ажгуу "), transcription_text, transcription_text)
        suffix_mun.click(lambda value: append_suffix(value, "мөн "), transcription_text, transcription_text)
        suffix_ted.click(lambda value: append_suffix(value, "тэд "), transcription_text, transcription_text)
        save_transcription.click(
            commit_transliteration,
            [transcription_text, transcription_image],
            save_status,
        ).then(get_recent_transliterations, outputs=recent_table)
    return app


# =============================================================================
# 7. ЭХЛҮҮЛЭХ
# =============================================================================
def verify_startup() -> None:
    global qdrant
    if qdrant is None:
        qdrant = make_qdrant_client()
        atexit.register(qdrant.close)
    if BATCH_SIZE < 1:
        raise RuntimeError("GANJUUR_BATCH_SIZE хамгийн багадаа 1 байх ёстой.")
    init_db()
    try:
        qdrant.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant HTTP холболт амжилтгүй {QDRANT_HOST}:{QDRANT_PORT}. "
            "Docker контейнер эхлүүлж 6333 порт нээгдсэн эсэхийг шалгана уу. "
            "Өөрчлэлт хийгдсэнгүй. Анхны алдаа: " + str(exc)
        ) from exc
    ensure_collection(qdrant)
    info = qdrant.get_collection(COLLECTION_NAME)
    log.info("Qdrant цуглуулга бэлэн: %s", COLLECTION_NAME)
    log.info("Төхөөрөмж: %s | Векторууд: %s", DEVICE.upper(), info.points_count)


if __name__ == "__main__":
    mp.freeze_support()
    verify_startup()
    username = os.getenv("GANJUUR_USER")
    password = os.getenv("GANJUUR_PASS")
    if not username or not password:
        raise RuntimeError("Монгол Шунхан Ганжуур эхлүүлэхийн өмнө GANJUUR_USER болон GANJUUR_PASS орчны хувьсагчиудыг тохируулна уу.")
    build_app().launch(
        server_name=os.getenv("GANJUUR_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GANJUUR_PORT", "7860")),
        show_error=True,
        auth=[(username, password)],
        auth_message="Монгол Шунхан Ганжуур — нэвтрэх",
        favicon_path=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
        allowed_paths=[str(MANUAL_PATH)],
    )
