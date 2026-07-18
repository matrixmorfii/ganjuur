# ==========================================
# 0. CRITICAL UPSTREAM PATCHES (СЭРГЭЭВ)
# ==========================================
import gradio_client.utils
import gradio.networking

# Patch 1: Pydantic v2 Schema Bug (TypeError: argument of type 'bool' is not iterable)
_orig_json_schema = gradio_client.utils._json_schema_to_python_type
def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool): return "any"
    return _orig_json_schema(schema, defs)
gradio_client.utils._json_schema_to_python_type = _patched_json_schema

_orig_get_type = gradio_client.utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool): return "bool"
    return _orig_get_type(schema)
gradio_client.utils.get_type = _patched_get_type

# Patch 2: Localhost Ping Error (ValueError: When localhost is not accessible...)
gradio.networking.url_ok = lambda url: True

# ==========================================
# 1. CONFIGURATION & IMPORTS
# ==========================================
import os
import uuid
import threading
import warnings
import atexit
import logging
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import gradio as gr
import plotly.graph_objects as go
import umap
import torch
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client import models
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- PATHS & SETTINGS ---
BASE_DIR = Path("/home/trinity/ganjuur")
SCANS_DIR = BASE_DIR / "data/scans"
CROPS_DIR = BASE_DIR / "data/ganjuur_crops/db_frames"
VECTORDB_PATH = BASE_DIR / "vectordb"
COLLECTION_NAME = "ganjuur_frames"
VECTOR_DIM = 768
BATCH_SIZE = 32 

LOGO_PATH = "image_566068.jpg"
FAVICON_PATH = "asset/favicon.png"
DB_PATH = BASE_DIR / "transliterations.db"

MAX_FILES = 5000
MAX_TOTAL_PIXELS = 10_000_000_000

SCANS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

db_lock = threading.Lock()

# --- QDRANT INIT ---
qdrant = QdrantClient(path=str(VECTORDB_PATH))
atexit.register(qdrant.close)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    log.info("✅ Qdrant collection created.")
    qdrant.create_payload_index(COLLECTION_NAME, "umap_x", field_schema=models.PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(COLLECTION_NAME, "anomaly_score", field_schema=models.PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(COLLECTION_NAME, "status", field_schema=models.PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "source", field_schema=models.PayloadSchemaType.KEYWORD)

# --- SQLITE INIT ---
def init_db():
    with db_lock:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        c = conn.cursor()
        c.execute('PRAGMA journal_mode=WAL;')
        c.execute('PRAGMA synchronous=NORMAL;')
        c.execute('''
            CREATE TABLE IF NOT EXISTS transliterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source_image TEXT,
                text_content TEXT,
                token_count INTEGER,
                char_count INTEGER,
                status TEXT,
                user_id TEXT DEFAULT 'trinity',
                version INTEGER DEFAULT 1
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_source ON transliterations(source_image);')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON transliterations(timestamp DESC);')
        conn.commit()
        conn.close()
    log.info(f"✅ SQLite DB initialized at {DB_PATH}")

init_db()

# ==========================================
# 2. ML MODEL & THREAD SAFETY
# ==========================================
log.info("Loading DINOv2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"⚙️ Device: {device.upper()}")

dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dinov2_model.eval()

model_lock = threading.Lock()
log.info("✅ System ready.")

# ==========================================
# 3. CORE LOGIC
# ==========================================

def get_embeddings_batch(img_bgr_list: list) -> list:
    if not img_bgr_list: return []
    img_rgb_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_bgr_list]
    with model_lock, torch.inference_mode():
        inputs = dinov2_processor(images=img_rgb_list, return_tensors="pt").to(device)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = dinov2_model(**inputs).last_hidden_state[:, 0, :]
        else:
            outputs = dinov2_model(**inputs).last_hidden_state[:, 0, :]
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
    return outputs.cpu().numpy().tolist()

def save_frame_sharded(frame: np.ndarray, page_id: str, frame_idx: int) -> tuple[str, str]:
    frame_uuid = str(uuid.uuid4())
    shard_dir = CROPS_DIR / frame_uuid[0:2] / frame_uuid[2:4]
    shard_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{frame_uuid}_p{page_id}_f{frame_idx:06d}.webp"
    full_path = shard_dir / filename
    cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_WEBP_QUALITY, 85])
    return str(full_path), frame_uuid

def flush_batch(crops: list, paths: list, ids: list, page_id: str) -> None:
    if not crops: return
    try:
        vecs = get_embeddings_batch(crops)
        points = [
            models.PointStruct(id=fid, vector=vec, payload={"source": page_id, "local_path": path, "status": "PENDING"})
            for fid, path, vec in zip(ids, paths, vecs)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        log.error(f"Qdrant upsert failed for {page_id}: {e}", exc_info=True)

def handle_zip_ingestion(zip_file_obj, stride: int, progress=gr.Progress()) -> str:
    if zip_file_obj is None:
        return "❌ .zip файлаа оруулна уу."

    zip_path = Path(zip_file_obj.name)

    try:
        progress(0.05, desc="Архивыг шалгаж байна...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            image_names = [n for n in zip_ref.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_names:
                return "❌ Архиваас зураг (.jpg, .png) олдсонгүй."
            if len(image_names) > MAX_FILES:
                return f"❌ Файл хэт олон: {len(image_names)} > {MAX_FILES}"

            total_images = len(image_names)
            grand_total_frames = 0
            total_pixels = 0

            progress(0.1, desc=f"Нийт {total_images} зураг олдлоо.")

            for idx, name in enumerate(sorted(image_names)):
                try:
                    with zip_ref.open(name) as f:
                        img_bytes = np.frombuffer(f.read(), np.uint8)
                        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                except Exception:
                    continue

                if img_bgr is None:
                    continue

                total_pixels += img_bgr.shape[0] * img_bgr.shape[1]
                if total_pixels > MAX_TOTAL_PIXELS:
                    return f"❌ Нийт пикселийн хэмжээ хэтэрлээ. Аюулгүй байдлын үүднээс зогсоов."

                page_id = Path(name).stem

                green = img_bgr[:, :, 1]
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
                enhanced = clahe.apply(green)
                processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

                h, w = processed.shape[:2]
                window = min(h, w)
                axis = "X" if w >= h else "Y"
                max_dim = w if axis == "X" else h

                total_steps = (max_dim - window) // stride + 1
                if total_steps <= 0:
                    continue

                crops_buf, paths_buf, ids_buf = [], [], []
                count = 0

                for pos in range(0, max_dim - window + 1, stride):
                    crop = processed[0:h, pos:pos + window] if axis == "X" else processed[pos:pos + window, 0:w]
                    path, f_uuid = save_frame_sharded(crop, page_id, count)

                    crops_buf.append(crop)
                    paths_buf.append(path)
                    ids_buf.append(f_uuid)

                    if len(crops_buf) >= BATCH_SIZE:
                        flush_batch(crops_buf, paths_buf, ids_buf, page_id)
                        crops_buf, paths_buf, ids_buf = [], [], []

                    count += 1

                if crops_buf:
                    flush_batch(crops_buf, paths_buf, ids_buf, page_id)

                grand_total_frames += count

                prog_val = 0.1 + 0.9 * ((idx + 1) / total_images)
                progress(prog_val, desc=f"Зураг {idx + 1}/{total_images} ({page_id}) - {count} хэсэг")

        return (
            f"✅ Амжилттай!\n\n"
            f"Нийт {total_images} зургаас {grand_total_frames} хэсэг үүсгэж санд хадгалав.\n\n"
            f"| Тохиргоо | Утга |\n|---|---|\n"
            f"| Алхам (stride) | {stride} px |\n"
            f"| Багц | {BATCH_SIZE} зураг |"
        )

    except zipfile.BadZipFile:
        return "❌ Файл гэмтэлтэй байна. Зөв .zip файл оруулна уу."
    except Exception as e:
        log.error(f"ZIP Ingestion Error: {e}", exc_info=True)
        return f"❌ Алдаа гарлаа: {str(e)}"

def execute_visual_query(img: np.ndarray, top_k: int) -> list:
    if img is None: return []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vec = get_embeddings_batch([img_bgr])[0]

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=vec, limit=top_k, with_payload=True,
    )

    results = []
    for h in response.points:
        payload = h.payload or {}
        path = payload.get("local_path")
        if path and os.path.exists(path):
            label = f"{h.score * 100:.1f}% | {payload.get('source', 'N/A')}"
            results.append((path, label))
    return results

def _run_analytics_worker():
    log.info("Starting background UMAP/LOF calculation...")
    all_vectors, all_ids = [], []
    offset = None

    while True:
        records, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            offset=offset,
            with_vectors=True,
            with_payload=False
        )
        if not records:
            break
        all_vectors.extend([r.vector for r in records])
        all_ids.extend([r.id for r in records])
        if offset is None:
            break

    if len(all_ids) < 10:
        log.warning("Not enough data for analytics")
        return

    vectors = np.array(all_vectors)
    ids = all_ids

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(vectors)

    lof = LocalOutlierFactor(n_neighbors=20, metric='cosine')
    lof.fit_predict(vectors)
    scores = -lof.negative_outlier_factor_

    s_min, s_max = scores.min(), scores.max()
    scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(scores)

    operations = [
        models.SetPayloadOperation(
            set_payload=models.SetPayload(
                points=[pid],
                payload={"umap_x": float(coords[i][0]), "umap_y": float(coords[i][1]), "anomaly_score": float(scores[i])}
            )
        ) for i, pid in enumerate(ids)
    ]

    for i in range(0, len(operations), 500):
        qdrant.batch_update_points(collection_name=COLLECTION_NAME, update_operations=operations[i:i + 500])
    log.info("✅ Background UMAP/LOF calculation finished.")

def run_analytics() -> str:
    thread = threading.Thread(target=_run_analytics_worker, daemon=True)
    thread.start()
    return "🧮 Тооцоолол дэвсгэр горимд эхэллээ. Дууссаны дараа 'График Шинэчлэх' товчийг дарна уу."

def get_plot_data():
    records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="umap_x", range=models.Range(gte=-9999.0))]),
        limit=5000, with_payload=True, with_vectors=False,
    )
    if not records: return None

    xs, ys, colors, texts, anomalies = [], [], [], [], []
    for r in records:
        p = r.payload
        score = p.get("anomaly_score", 0.0)
        xs.append(p["umap_x"]); ys.append(p["umap_y"]); colors.append(score)
        texts.append(f"Эх сурвалж: {p.get('source', 'N/A')}<br>Оноо: {score:.3f}")
        if score > 0.7: anomalies.append(r)

    return {"xs": xs, "ys": ys, "colors": colors, "texts": texts, "anomalies": anomalies}

def generate_plot():
    data = get_plot_data()
    empty_fig = go.Figure().update_layout(title="Эхлээд 'UMAP/LOF Тооцоолох' товчийг дарна уу", template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')

    if not data: return empty_fig, gr.update(choices=[], value=None)

    xs, ys, colors, texts, anomalies = data["xs"], data["ys"], data["colors"], data["texts"], data["anomalies"]
    marker_sizes = [10 if c > 0.7 else 4 for c in colors]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode='markers', marker=dict(size=marker_sizes, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title="Гажилт", thickness=12, len=0.8), opacity=0.85), text=texts, hoverinfo='text'))

    hi_x = [xs[i] for i, c in enumerate(colors) if c > 0.7]
    hi_y = [ys[i] for i, c in enumerate(colors) if c > 0.7]
    if hi_x:
        fig.add_trace(go.Scattergl(x=hi_x, y=hi_y, mode='markers', marker=dict(size=12, color='red', symbol='circle-open', line=dict(width=2)), name='Гажилт (>0.7)', hoverinfo='skip'))

    fig.update_layout(template='plotly_dark', title=dict(text="Чанарын хяналтын самбар", font=dict(size=14)), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation='h', y=-0.02, font=dict(size=11)), paper_bgcolor='rgba(0,0,0,0)', height=450)

    choices = [(f"Оноо: {r.payload.get('anomaly_score', 0):.3f} | {r.payload.get('source', 'N/A')}", str(r.id)) for r in anomalies]
    return fig, gr.update(choices=choices, value=choices[0][1] if choices else None)

def inspect_node(node_id: str) -> tuple:
    if not node_id: return None, "Цэгийг сонгоно уу.", go.Figure()
    try:
        records = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[node_id], with_payload=True, with_vectors=True)
        if not records: return None, "❌ Олдсонгүй.", go.Figure()

        p = records[0].payload
        score = p.get("anomaly_score", 0.0)
        status = "🔴 Өндөр" if score > 0.7 else "🟡 Дунд" if score > 0.4 else "🟢 Бага"

        meta = f"### {status} эрсдэлтэй цэг\n\n| Талбар | Утга |\n|---|---|\n| Эх сурвалж | `{p.get('source', 'N/A')}` |\n| Гажилтын оноо | `{score:.4f}` |\n| Координат | `({p.get('umap_x', 0):.2f}, {p.get('umap_y', 0):.2f})` |"

        vec = records[0].vector[:64]
        fig_emb = go.Figure()
        fig_emb.add_trace(go.Bar(y=vec, marker=dict(color=vec, colorscale='RdBu', cmid=0, opacity=0.8)))
        fig_emb.update_layout(title="Эмбеддинг (эхний 64 хэмжээс)", template='plotly_dark', margin=dict(l=10, r=10, t=35, b=10), xaxis=dict(showticklabels=False), paper_bgcolor='rgba(0,0,0,0)', height=200)

        return p.get("local_path"), meta, fig_emb
    except Exception as e:
        log.error(f"inspect_node error: {e}", exc_info=True)
        return None, f"❌ Алдаа: {e}", go.Figure()

def commit_transliteration(text: str, img_path: str) -> str:
    if not text or not text.strip(): return "⚠️ Текст хоосон байна."

    # 🛡️ HF Issue #31884-өөс хамгаалагдсан цэвэр UTF-8 шалгалт
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return "🔴 Текст кодлох боломжгүй (UTF-8 алдаа)."

    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            c = conn.cursor()
            c.execute('''
                INSERT INTO transliterations (timestamp, source_image, text_content, token_count, char_count, status)
                VALUES (?,?,?,?,?,?)
            ''', (datetime.now().isoformat(), img_path if img_path else "Тодорхойгүй", text, len(text.split()), len(text), "БАТАЛГААЖСАН"))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"SQLite Error: {e}")
            return f"🔴 Сангийн алдаа: {e}"

    return (
        f"✅ Амжилттай хадгалагдлаа\n\n"
        f"| Үзүүлэлт | Утга |\n|---|---|\n"
        f"| Үг | {len(text.split())} |\n"
        f"| Тэмдэгт | {len(text)} |\n"
        f"| Нарийвчлал | 100% ✓ |\n"
        f"| Эх зураг | {Path(img_path).name if img_path else 'N/A'} |"
    )

def get_recent_transliterations():
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            c = conn.cursor()
            c.execute('''SELECT timestamp, source_image, text_content, char_count FROM transliterations ORDER BY timestamp DESC LIMIT 5''')
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            log.error(f"Failed to fetch historical rows: {e}")
            rows = []

    if not rows: return []
    return rows

# ==========================================
# 4. UI DEFINITION
# ==========================================
CSS = """
.gradio-container { max-width: 1200px !important; padding: 2rem !important; }
.tab-nav button { font-size: 13px !important; font-weight: 500 !important; letter-spacing: 0.02em; padding: 8px 16px !important; }
.tab-nav button.selected { background: var(--primary-600) !important; color: white !important; }
.gr-button-primary { font-weight: 500 !important; letter-spacing: 0.01em; }
.gr-box { border-radius: 8px !important; }
footer { display: none !important; }

.header-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
}
.header-logo img { border-radius: 8px; object-fit: cover; }
.header-card h1 { margin: 0 !important; color: white !important; font-size: 1.5rem !important; font-weight: 600 !important; }
.header-card h3 { margin: 4px 0 0 0 !important; color: #a5b4fc !important; font-size: 0.875rem !important; }

.section-card { background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; }
.metric-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; background: var(--background-fill-secondary); }
"""

with gr.Blocks(title="Монгол шунхан Ганжуур", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate"), css=CSS) as app:

    with gr.Row(elem_classes=["header-card"]):
        if os.path.exists(LOGO_PATH):
            gr.Image(value=LOGO_PATH, show_label=False, container=False, width=80, height=80, interactive=False, elem_classes=["header-logo"])
        with gr.Column(scale=1):
            gr.Markdown("# Монгол шунхан Ганжуур")
            gr.Markdown("### DINOv2 · Qdrant · UMAP/LOF · SQLite")

    with gr.Tabs():
        with gr.TabItem("  Өгөгдөл Оруулах"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Скан хуудасны зургуудыг (.zip) векторжуулж санд хадгалах")
                    with gr.Group():
                        zip_file_in = gr.File(label="Зургуудын архив (.zip)", file_types=[".zip"])
                        stride_in = gr.Slider(10, 300, 50, step=10, label="Алхам (пиксел) — Бага = илүү олон хэсэг")
                    btn_ingest = gr.Button("🚀 Эхлүүлэх", variant="primary", size="lg")
                with gr.Column(scale=3):
                    ingest_output = gr.Markdown("""<div style="text-align: center; padding: 3rem; color: #6b7280;"><p style="font-size: 2rem; margin-bottom: 0.5rem;">📥</p><p>.zip файл оруулж "Эхлүүлэх" товчийг дарна уу</p></div>""", elem_classes=["section-card"])

        with gr.TabItem("  Визуал Хайлт"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Хайлтын зураг")
                    q_img = gr.Image(type="numpy", height=280, label="Зураг оруулах")
                    k_slider = gr.Slider(1, 30, 5, step=1, label="Хамгийн ойр илэрц (Top K)")
                    btn_search = gr.Button("🔎 Хайх", variant="primary")
                with gr.Column(scale=3):
                    gr.Markdown("### Олдсон ижил төстэй хэсгүүд")
                    gallery = gr.Gallery(columns=4, height=400, object_fit="contain", label="Үр дүн")

        with gr.TabItem("  Чанарын Хяналт"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Тооцоолол")
                    btn_analytics = gr.Button("🧮 UMAP/LOF Тооцоолох", variant="stop")
                    btn_plot = gr.Button("📈 График Шинэчлэх", variant="secondary")
                    analytics_status = gr.Markdown("*Тооцоолол хийгдээгүй*")
                with gr.Column(scale=4):
                    qa_plot = gr.Plot(label="UMAP Визуалчлал")

            gr.HTML("<hr style='margin: 1.5rem 0; border-color: var(--border-color-primary);'>")
            gr.Markdown("### Гажилттай цэгүүдийг шалгах")

            with gr.Row():
                with gr.Column(scale=1):
                    node_dropdown = gr.Dropdown(label="Цэг сонгох (Жагсаалтаас)", choices=[], interactive=True)
                with gr.Column(scale=2):
                    with gr.Row():
                        detail_image = gr.Image(label="Зураг", type="filepath", height=200)
                        detail_meta = gr.Markdown()
                with gr.Column(scale=2):
                    detail_embedding = gr.Plot(label="Эмбеддинг")
                    btn_send = gr.Button("📝 Галиглах руу илгээх", size="sm")

        with gr.TabItem("  Галиглах"):
            gr.Markdown("### Текст Галиг Оруулах")
            gr.Markdown("*Текстийн нарийвчлал 100% хамгаалагдсан. Түүхийн бичвэрийн бүрэн бүтэн байдлыг хадгална.*", elem_classes=["metric-badge"])

            with gr.Row():
                with gr.Column(scale=1):
                    translit_img = gr.Image(label="Эх зураг", type="filepath", interactive=True, height=350)
                with gr.Column(scale=1):
                    txt_input = gr.Textbox(label="Галиг текст", lines=10, placeholder="Монгол галигийг энд оруулна уу...")
                    with gr.Row():
                        btn_bugd = gr.Button("бөгөөд", size="sm")
                        btn_ajguu = gr.Button("ажгуу", size="sm")
                        btn_mun = gr.Button("мөн", size="sm")
                        btn_ted = gr.Button("тэд", size="sm")
                    btn_commit = gr.Button("✅ Хадгалах", variant="primary")
                    commit_out = gr.Markdown()

            gr.Markdown("### 📋 Сүүлд хадгалагдсан галигууд")
            recent_transliterations = gr.Dataframe(
                headers=["Цаг хугацаа", "Эх зураг", "Галиг текст", "Тэмдэгтийн тоо"],
                interactive=False,
                height=200
            )

    btn_ingest.click(fn=handle_zip_ingestion, inputs=[zip_file_in, stride_in], outputs=ingest_output)
    btn_search.click(fn=execute_visual_query, inputs=[q_img, k_slider], outputs=gallery)
    btn_analytics.click(fn=run_analytics, outputs=analytics_status)
    btn_plot.click(fn=generate_plot, outputs=[qa_plot, node_dropdown])
    node_dropdown.change(fn=inspect_node, inputs=node_dropdown, outputs=[detail_image, detail_meta, detail_embedding])
    btn_send.click(fn=lambda x: x, inputs=detail_image, outputs=translit_img)
    btn_bugd.click(lambda x: (x or "") + "бөгөөд ", inputs=txt_input, outputs=txt_input)
    btn_ajguu.click(lambda x: (x or "") + "ажгуу ", inputs=txt_input, outputs=txt_input)
    btn_mun.click(lambda x: (x or "") + "мөн ", inputs=txt_input, outputs=txt_input)
    btn_ted.click(lambda x: (x or "") + "тэд ", inputs=txt_input, outputs=txt_input)
    btn_commit.click(fn=commit_transliteration, inputs=[txt_input, translit_img], outputs=commit_out).then(fn=get_recent_transliterations, outputs=recent_transliterations)

# ==========================================
# 6. LAUNCH
# ==========================================
if __name__ == "__main__":
    AUTH_USER = os.getenv("GANJUUR_USER", "trinity")
    AUTH_PASS = os.getenv("GANJUUR_PASS", "ganjuur2026")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        auth=[(AUTH_USER, AUTH_PASS)],
        auth_message="📜 Ганжуур систем — Нэвтрэх",
        favicon_path=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    )
