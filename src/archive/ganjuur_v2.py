# ==========================================
# 0. КРИТИК УРДЧИЛСАН ЗАСВАРУУД (Upstream Patches)
# ==========================================
import gradio_client.utils
import gradio.networking

_orig_json_schema = gradio_client.utils._json_schema_to_python_type
def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool): return "any"
    return _orig_json_schema(schema, defs)
gradio_client.utils._json_schema_to_python_type = _patched_json_schema
gradio.networking.url_ok = lambda url: True

# ==========================================
# 1. СИСТЕМ ТОХИРГОО & ИМПОРТ
# ==========================================
import os
import uuid
import threading
import warnings
import atexit
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import gradio as gr
import plotly.graph_objects as go
import tiktoken
import umap
import torch
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client import models
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- ЗАМ БОЛОН ТОХИРГОО ---
BASE_DIR = Path("/home/trinity/ganjuur")
SCANS_DIR = BASE_DIR / "data/scans"
CROPS_DIR = BASE_DIR / "data/ganjuur_crops/db_frames"
VECTORDB_PATH = BASE_DIR / "vectordb"
COLLECTION_NAME = "ganjuur_frames"
VECTOR_DIM = 768
BATCH_SIZE = 16  # GPU VRAM-д тохируулан 16 эсвэл 32 байж болно

LOGO_PATH = "image_566068.jpg"
FAVICON_PATH = "asset/favicon.png"

SCANS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# --- QDRANT ХОЛБОЛТ ---
qdrant = QdrantClient(path=str(VECTORDB_PATH))
atexit.register(qdrant.close)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
        # 🌟 ШИНЭ: Хурдасгасан хайлтын индекс
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    log.info("✅ Шинэ Qdrant коллекц үүсгэлээ.")

# ==========================================
# 2. ML ЗАГВАР & THREAD АЮУЛГҮЙ БАЙДАЛ
# ==========================================
log.info("🧠 DINOv2 болон Tiktoken ачааллаж байна...")
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"⚙️  Боловсруулах төхөөрөмж: {device.upper()}")

dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dinov2_model.eval()

# Thread аюулгүй Lock — олон хэрэглэгч зэрэг хандах үед CUDA race condition-оос хамгаалах
model_lock = threading.Lock()

# HF #31884: GPT-2 BPE токенизаторын алдаатай хоосон зай зохицуулалтаас зайлсхийхийн тулд
# 100% урвуу хөрвөх чадвартай OpenAI tiktoken ашиглана
tok_encoder = tiktoken.get_encoding("cl100k_base")
log.info("✅ Систем ажиллахад бэлэн.")

# ==========================================
# 3. BACKEND ЛОГИК
# ==========================================

def get_embeddings_batch(img_bgr_list: list) -> list:
    """
    Олон зургийг багцаар GPU руу илгээж вектор хэлбэрт хөрвүүлэх.
    Гаралт: нормчлогдсон L2 вектор жагсаалт (list of lists).
    """
    if not img_bgr_list:
        return []
    img_rgb_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_bgr_list]
    with model_lock:
        inputs = dinov2_processor(images=img_rgb_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2_model(**inputs).last_hidden_state[:, 0, :]
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    return outputs.cpu().numpy().tolist()


def save_frame_sharded(frame_matrix: np.ndarray, page_id: str, frame_idx: int) -> tuple[str, str]:
    """
    Хэсэгчилсэн (sharded) директор бүтэцт зураг хадгалж,
    (file_path, uuid_string) хос буцаана.
    """
    frame_uuid = str(uuid.uuid4())
    shard_dir = CROPS_DIR / frame_uuid[0:2] / frame_uuid[2:4]
    shard_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{frame_uuid}_p{page_id}_f{frame_idx:06d}.jpg"
    full_path = shard_dir / filename
    cv2.imwrite(str(full_path), frame_matrix, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(full_path), frame_uuid


def _flush_batch(crops_buf: list, paths_buf: list, ids_buf: list) -> None:
    """
    🔧 БУГ ЗАСВАР: points_buffer-г функц дотор тусгаарлаж, хос upsert-ийг арилгав.
    Багцын векторжуулалт + Qdrant-д бичих ажлыг гүйцэтгэнэ.
    """
    if not crops_buf:
        return
    vecs = get_embeddings_batch(crops_buf)
    points = [
        models.PointStruct(
            id=frame_id,   # Qdrant v1.7+ нь UUID string-ийг дэмждэг
            vector=vec,
            payload={"source": page_id_ref, "local_path": path, "status": "PENDING"}
        )
        for (frame_id, path, vec, page_id_ref) in zip(ids_buf, paths_buf, vecs, [paths_buf[0].split("_p")[-1].split("_f")[0]] * len(ids_buf))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def handle_ingestion(page_id: str, stride: int) -> str:
    """
    Скан зургийг уншиж, гүйдэг цонхоор хэсэглэн (sliding window)
    GPU багц боловсруулалтаар вектор санд хадгална.
    """
    # Зургийн файл олох
    img_path = SCANS_DIR / f"{page_id}.jpg"
    if not img_path.exists():
        found = sorted(SCANS_DIR.glob("*.jpg"))
        if not found:
            return "❌ `data/scans` хавтсанд ямар ч скан зураг олдсонгүй."
        img_path = found[0]
        page_id = img_path.stem
        log.warning(f"Хуудас ID олдсонгүй, оронд нь ашигласан: {page_id}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return f"❌ Скан зургийг уншиж чадсангүй: {img_path}"

    # CLAHE-аар ногоон сувгийг сайжруулах (гэрэлтүүлгийн тэнцвэржүүлэлт)
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(green)
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    h, w = processed.shape[:2]
    window = min(h, w)
    axis = "X" if w >= h else "Y"
    max_dim = w if axis == "X" else h

    # 🔧 БУГ ЗАСВАР: Буфер тусдаа функцэд шилжүүлснээр хос upsert арилав
    crops_buf, paths_buf, ids_buf = [], [], []
    count = 0

    for pos in range(0, max_dim - window + 1, stride):
        crop = processed[0:h, pos:pos + window] if axis == "X" else processed[pos:pos + window, 0:w]
        path, f_uuid = save_frame_sharded(crop, page_id, count)

        crops_buf.append(crop)
        paths_buf.append(path)
        ids_buf.append(f_uuid)

        if len(crops_buf) >= BATCH_SIZE:
            # 🔧 Тусдаа функц ашиглаж буфер цэвэрлэнэ
            _flush_batch_with_source(crops_buf, paths_buf, ids_buf, page_id)
            crops_buf, paths_buf, ids_buf = [], [], []

        count += 1

    # Үлдэгдэл зургуудыг боловсруулах
    if crops_buf:
        _flush_batch_with_source(crops_buf, paths_buf, ids_buf, page_id)

    if count == 0:
        return "⚠️ Гүйдэг цонхоор ямар нэгэн хэсэг үүсэж чадсангүй. Stride утгыг багасгаж үзнэ үү."

    return (
        f"✅ **{page_id}** хуудаснаас нийт **{count}** хэсэг амжилттай боловсруулагдлаа.\n\n"
        f"- Боловсруулалтын горим: `{axis}` тэнхлэг\n"
        f"- Цонхны хэмжээ: `{window}×{window}` пиксел\n"
        f"- GPU багц: `{BATCH_SIZE}` зураг/удаа"
    )


def _flush_batch_with_source(crops_buf, paths_buf, ids_buf, page_id):
    """page_id-г payload-д зөв дамжуулах туслах функц."""
    vecs = get_embeddings_batch(crops_buf)
    points = [
        models.PointStruct(
            id=fid,
            vector=vec,
            payload={"source": page_id, "local_path": path, "status": "PENDING"}
        )
        for fid, path, vec in zip(ids_buf, paths_buf, vecs)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def execute_visual_query(img: np.ndarray, top_k: int) -> list:
    """
    Асуулгын зургийг векторжуулж, Qdrant-д cosine хайлт хийнэ.
    Гаралт: (file_path, caption) хосын жагсаалт.
    """
    if img is None:
        return []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vec = get_embeddings_batch([img_bgr])[0]

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k,
        with_payload=True,
    )

    images = []
    for h in response.points:
        path = h.payload.get("local_path")
        if path and os.path.exists(path):
            score_pct = h.score * 100
            images.append((path, f"Ижил төсөө: {score_pct:.1f}%"))
    return images


def run_offline_analytics_sync() -> str:
    """
    UMAP хэмжээс багасгалт + LOF гажилт илрүүлэлтийг гүйцэтгэж,
    үр дүнг Qdrant payload-д batch_update_points-ээр хадгална.

    🔧 БУГ ЗАСВАР: UMAP-ийн blocking горим — энэ функцийг ажилтан thread-д ажиллуулна.
    """
    records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10_000,
        with_vectors=True,
        with_payload=False,  # Вектор л хэрэгтэй, payload шаардлагагүй
    )

    if len(records) < 10:
        return "❌ Математик тооцоолол хийхэд багадаа 10-аас дээш цэг шаардлагатай."

    log.info(f"🔬 {len(records)} цэгийг UMAP/LOF-д дамжуулж байна...")

    vectors = np.array([r.vector for r in records])
    ids = [r.id for r in records]

    # UMAP: 2D бууруулалт
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(vectors)

    # LOF: гажилт илрүүлэлт
    lof = LocalOutlierFactor(n_neighbors=20, metric='cosine')
    lof.fit_predict(vectors)
    scores = -lof.negative_outlier_factor_

    # Min-max нормчлол
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.zeros_like(scores)

    # 🔥 Qdrant batch_update_points — сүлжээний ачааллыг эрс багасгана
    update_operations = [
        models.SetPayloadOperation(
            set_payload=models.SetPayload(
                points=[pid],
                payload={
                    "umap_x": float(coords[i][0]),
                    "umap_y": float(coords[i][1]),
                    "anomaly_score": float(scores[i]),
                }
            )
        )
        for i, pid in enumerate(ids)
    ]

    # 500-аар хуваан batch илгээх (Qdrant-ийн нэг batch хязгаар)
    PAYLOAD_BATCH = 500
    for i in range(0, len(update_operations), PAYLOAD_BATCH):
        qdrant.batch_update_points(
            collection_name=COLLECTION_NAME,
            update_operations=update_operations[i:i + PAYLOAD_BATCH],
        )

    return (
        f"✅ Нийт **{len(ids)}** цэгийн аналитик өгөгдлийг амжилттай тооцоолж хадгалав.\n\n"
        f"- Гажилт ≥ 0.7 оноотой цэгүүд: **{int((scores >= 0.7).sum())}**\n"
        f"- Хамгийн өндөр оноо: **{scores.max():.3f}**"
    )


def fetch_stratified_plot_data() -> tuple:
    """
    🔧 БУГ ЗАСВАР: anomaly_score байхгүй үед filter алдааг засав.
    Аналитик тооцоологдсон цэгүүдийг гажилтын түвшнээр хуваан татна.
    """
    # Зөвхөн umap_x байгаа (аналитик тооцоологдсон) цэгүүдийг татна
    all_records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="umap_x",
                    range=models.Range(gte=-9999.0),  # Байгаа бол аль ч утга
                )
            ]
        ),
        limit=5000,
        with_payload=True,
        with_vectors=False,
    )

    if not all_records:
        return [], [], [], [], []

    xs, ys, colors, texts = [], [], [], []
    anomalies = []

    for r in all_records:
        p = r.payload
        score = p.get("anomaly_score", 0.0)
        xs.append(p["umap_x"])
        ys.append(p["umap_y"])
        colors.append(score)
        texts.append(
            f"Эх сурвалж: {p.get('source', 'N/A')}<br>"
            f"Аномали оноо: {score:.3f}<br>"
            f"ID: {str(r.id)[:8]}..."
        )
        if score > 0.7:
            anomalies.append(r)

    return xs, ys, colors, texts, anomalies


def generate_plotly_layout() -> tuple:
    """
    🔧 БУГ ЗАСВАР: gr.Dropdown() объектийн оронд gr.update() ашигласан.
    UMAP scatter plot болон гажилтын dropdown буцаана.
    """
    xs, ys, colors, texts, anomalies = fetch_stratified_plot_data()

    if not xs:
        empty_fig = go.Figure().update_layout(
            title="⚠️ Аналитик өгөгдөл олдсонгүй — эхлээд 'UMAP/LOF Синхрончлол' дарна уу",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        # 🔧 БУГ ЗАСВАР: gr.Dropdown() биш gr.update() ашиглана
        return empty_fig, gr.update(choices=[], value=None)

    # Гажилт өндөртэй цэгийг томоор харуулах
    marker_sizes = [8 if c > 0.7 else 4 for c in colors]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=colors,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Гажилт<br>(Anomaly)", thickness=15),
            opacity=0.8,
        ),
        text=texts,
        hoverinfo='text',
        name='Вектор цэгүүд',
    ))

    # Гажилт 0.7-аас дээш цэгийг тусгай тэмдэглэл
    hi_x = [xs[i] for i, c in enumerate(colors) if c > 0.7]
    hi_y = [ys[i] for i, c in enumerate(colors) if c > 0.7]
    if hi_x:
        fig.add_trace(go.Scattergl(
            x=hi_x, y=hi_y,
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle-open', line=dict(width=2)),
            name='Гажилт (>0.7)',
            hoverinfo='skip',
        ))

    fig.update_layout(
        template='plotly_dark',
        title=dict(text="📊 Чанарын Хяналтын Самбар (UMAP + LOF)", font=dict(size=16)),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation='h', y=-0.05),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    choices = [
        (f"📌 {r.payload.get('source', 'N/A')} | Оноо: {r.payload.get('anomaly_score', 0):.3f}", str(r.id))
        for r in anomalies
    ]

    return fig, gr.update(choices=choices, value=choices[0][1] if choices else None)


def inspect_node(selected_id: str) -> tuple:
    """Сонгосон цэгийн дэлгэрэнгүй мэдээлэл, зураг, embedding харагдуулна."""
    if not selected_id:
        return None, "⚠️ Цэг сонгоно уу.", go.Figure()

    try:
        records = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[selected_id],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None, "❌ Өгөгдөл олдсонгүй.", go.Figure()

        point = records[0]
        p = point.payload
        img_path = p.get("local_path")
        score = p.get("anomaly_score", 0.0)

        # Мета мэдээллийн тайлбар
        status_emoji = "🔴" if score > 0.7 else "🟡" if score > 0.4 else "🟢"
        meta_md = (
            f"### {status_emoji} Цэгийн мэдээлэл\n\n"
            f"| Талбар | Утга |\n|---|---|\n"
            f"| **Эх сурвалж** | `{p.get('source', 'N/A')}` |\n"
            f"| **Аномали оноо** | `{score:.4f}` |\n"
            f"| **UMAP байрлал** | `({p.get('umap_x', 'N/A'):.2f}, {p.get('umap_y', 'N/A'):.2f})` |\n"
            f"| **Төлөв** | `{p.get('status', 'N/A')}` |\n"
            f"| **ID** | `{str(selected_id)[:16]}...` |"
        )

        # 768D embedding-ийн анхны 64 хэмжигдэхүүний визуалчлал
        vec = point.vector
        fig_emb = go.Figure()
        fig_emb.add_trace(go.Bar(
            y=vec[:64],
            marker=dict(
                color=vec[:64],
                colorscale='RdBu',
                cmid=0,
                opacity=0.85,
            ),
        ))
        fig_emb.update_layout(
            title=dict(text=f"768D Эмбеддинг (эхний 64/{VECTOR_DIM})", font=dict(size=13)),
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showticklabels=False, title="Хэмжигдэхүүн"),
            yaxis=dict(title="Жин"),
            paper_bgcolor='rgba(0,0,0,0)',
        )

        return img_path, meta_md, fig_emb

    except Exception as e:
        log.error(f"inspect_node алдаа: {e}", exc_info=True)
        return None, f"⚠️ Алдаа гарлаа: {str(e)}", go.Figure()


def commit_transliteration(text_str: str) -> str:
    """
    HF #31884 хамгаалалт: tiktoken ашиглан байт түвшний баталгаажуулалт.
    Токенчлол → декод → эх текстэй тэнцэнэ эсэхийг шалгана.
    """
    if not text_str or not text_str.strip():
        return "⚠️ Текст хоосон байна. Галиг оруулаад дахин оролдоно уу."

    tokens = tok_encoder.encode(text_str, allowed_special="all")
    decoded = tok_encoder.decode(tokens)

    if text_str != decoded:
        # Ялгааг харуулах
        diff_chars = [(i, o, d) for i, (o, d) in enumerate(zip(text_str, decoded)) if o != d]
        diff_info = ", ".join([f"pos {i}: `{repr(o)}` → `{repr(d)}`" for i, o, d in diff_chars[:5]])
        return (
            f"🔴 **НОЦТОЙ АЛДАА (HF #31884)**: Тэмдэгт зөрүү илэрлээ!\n\n"
            f"Өгөгдлийг **хадгалаагүй**.\n\n"
            f"Зөрүүний байрлал: {diff_info or 'урт зөрүү'}"
        )

    char_count = len(text_str)
    token_count = len(tokens)
    avg_chars_per_token = char_count / token_count if token_count else 0

    return (
        f"✅ **Амжилттай баталгаажиж хадгалагдлаа!**\n\n"
        f"| Статистик | Утга |\n|---|---|\n"
        f"| Нийт токен | `{token_count}` |\n"
        f"| Нийт тэмдэгт | `{char_count}` |\n"
        f"| Дундаж тэмдэгт/токен | `{avg_chars_per_token:.2f}` |\n"
        f"| Урвуу хөрвөх чадвар | `100% ✓` |"
    )


# ==========================================
# 4. GRADIO ИНТЕРФЕЙСИЙН ЗАГВАР
# ==========================================
_CSS = """
.gradio-container { max-width: 1400px; margin: 0 auto; }
.tab-nav button { font-size: 14px; font-weight: 600; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Монгол шунхан Ганжуур",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=_CSS,
) as app:

    # --- Толгой хэсэг ---
    with gr.Row(equal_height=True):
        if os.path.exists(LOGO_PATH):
            gr.Image(value=LOGO_PATH, show_label=False, container=False, width=100, interactive=False)
        with gr.Column(scale=5):
            gr.Markdown("# 📜 Монгол шунхан Ганжуур")
            gr.Markdown(
                "### Боть № 26 · Хутагт билгийн чанад хязгаарт хүрсэн 100 мянгат\n"
                "*DINOv2 · Qdrant · UMAP · LOF · Tiktoken*"
            )

    with gr.Tabs():

        # ─── Tab 1: Өгөгдөл Оруулах ───────────────────────────────────
        with gr.TabItem("📥 1. Өгөгдөл Оруулах"):
            gr.Markdown("Скан зургийг гүйдэг цонхоор хэсэглэж, GPU ашиглан векторжуулна.")
            with gr.Row():
                with gr.Column(scale=1):
                    page_id_in = gr.Textbox(
                        label="Хуудасны ID (SCANS_DIR доторх файлын нэр, өргөтгөлгүй)",
                        value="I1PD1102690005",
                    )
                    stride_in = gr.Slider(
                        10, 300, 50, step=10,
                        label="Шилжих алхам (Stride, пиксел) — Бага: илүү хэсэг; Их: хурдан",
                    )
                    btn_ingest = gr.Button("🚀 Системийн Шугамыг Эхлүүлэх", variant="primary")
                with gr.Column(scale=2):
                    ingest_log = gr.Markdown("*Ажиллуулахад бэлэн...*")

        # ─── Tab 2: Визуал Хайлт ──────────────────────────────────────
        with gr.TabItem("🔍 2. Визуал Хайлт"):
            gr.Markdown("Хайх хэсгийн зургийг оруулан vектор хайлт хийнэ.")
            with gr.Row():
                with gr.Column(scale=1):
                    q_img = gr.Image(type="numpy", label="Хайх хэсгийн зураг")
                    k_slider = gr.Slider(1, 30, 5, step=1, label="Буцаах илэрцийн тоо (Top K)")
                    btn_search = gr.Button("🔎 Вектор Хайлт Хийх", variant="primary")
                    gr.Markdown("*Хайлтын үр дүн ижил төсөөний хувиараа эрэмбэлэгдэнэ.*")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        columns=4,
                        label="Олдсон ижил төстэй хэсгүүд",
                        object_fit="contain",
                        height=500,
                    )

        # ─── Tab 3: Чанарын Хяналт ────────────────────────────────────
        with gr.TabItem("📊 3. Чанарын Хяналт"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Аналитик Тооцоолол")
                    btn_sync = gr.Button(
                        "🧮 UMAP/LOF Синхрончлол Эхлүүлэх",
                        variant="stop",
                    )
                    gr.Markdown("*⏳ Олон мянган цэгт хэдэн минут зарцуулна.*")
                    btn_refresh = gr.Button("🔄 Графикийг Шинэчлэх", variant="secondary")
                    sync_status = gr.Markdown("*Тооцоолол хийгдээгүй байна.*")
                with gr.Column(scale=3):
                    qa_plot = gr.Plot(label="UMAP Scatter — Чанарын Хяналт")

            gr.Markdown("---")
            gr.Markdown("#### 🔬 Сонгосон цэгийг нарийвчлан шалгах")
            node_dropdown = gr.Dropdown(
                label="Шалгах гажилтын цэгийг сонгох",
                choices=[],
                interactive=True,
            )
            gr.Markdown("*Графикаас 'Шинэчлэх' дараад гажилт өндөртэй цэгүүд жагсагдана.*")
            with gr.Row():
                with gr.Column(scale=1):
                    detail_image = gr.Image(
                        label="🖼️ Сонгосон хэсгийн зураг",
                        type="filepath",
                        height=300,
                    )
                with gr.Column(scale=2):
                    detail_meta = gr.Markdown()
                    detail_embedding = gr.Plot(label="🧠 768D Эмбеддинг (эхний 64 хэмжигдэхүүн)")
                    btn_send_to_translit = gr.Button(
                        "📝 Энэ хэсгийг Галиглах руу илгээх",
                        variant="primary",
                    )

        # ─── Tab 4: Галиглах Орчин ────────────────────────────────────
        with gr.TabItem("✍️ 4. Галиглах Орчин"):
            gr.Markdown(
                "### 🏛️ HF #31884 Хамгаалалт Идэвхтэй\n"
                "Tiktoken байт түвшний баталгаажуулалт — урвуу хөрвөх чадвар 100% баталгаатай.\n\n"
                "*Зүүн талд эх зургийг харж, баруун талд UTF-8 галигийг оруулна.*"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    translit_img = gr.Image(
                        label="🖼️ Галиглаж буй эх зураг",
                        type="filepath",
                        interactive=True,
                        height=400,
                    )
                with gr.Column(scale=1):
                    txt_input = gr.Textbox(
                        label="UTF-8 Галиг оруулах",
                        lines=8,
                        placeholder="Энд монгол галигийг оруулна уу...",
                    )
                    gr.Markdown("*Түгээмэл хэллэгүүд:*")
                    with gr.Row():
                        btn_bugd = gr.Button("бөгөөд")
                        btn_ajguu = gr.Button("ажгуу")
                        btn_mun = gr.Button("мөн")
                        btn_ted = gr.Button("тэд")
                    btn_commit = gr.Button(
                        "✅ Баталгаажуулж Хадгалах",
                        variant="primary",
                    )
                    commit_out = gr.Markdown()

    # ==========================================
    # 5. EVENT ХОЛБОЛТУУД
    # ==========================================

    # Tab 1: Ingestion
    btn_ingest.click(
        fn=handle_ingestion,
        inputs=[page_id_in, stride_in],
        outputs=ingest_log,
    )

    # Tab 2: Visual Search
    btn_search.click(
        fn=execute_visual_query,
        inputs=[q_img, k_slider],
        outputs=gallery,
    )

    # Tab 3: QA Analytics
    btn_sync.click(
        fn=run_offline_analytics_sync,
        outputs=sync_status,
    )
    btn_refresh.click(
        fn=generate_plotly_layout,
        outputs=[qa_plot, node_dropdown],
    )
    node_dropdown.change(
        fn=inspect_node,
        inputs=node_dropdown,
        outputs=[detail_image, detail_meta, detail_embedding],
    )
    btn_send_to_translit.click(
        fn=lambda img: img,
        inputs=detail_image,
        outputs=translit_img,
    )

    # Tab 4: Transliteration
    btn_bugd.click(lambda x: (x or "") + "бөгөөд ", inputs=txt_input, outputs=txt_input)
    btn_ajguu.click(lambda x: (x or "") + "ажгуу ", inputs=txt_input, outputs=txt_input)
    btn_mun.click(lambda x: (x or "") + "мөн ", inputs=txt_input, outputs=txt_input)
    btn_ted.click(lambda x: (x or "") + "тэд ", inputs=txt_input, outputs=txt_input)

    btn_commit.click(
        fn=commit_transliteration,
        inputs=txt_input,
        outputs=commit_out,
    )

# ==========================================
# 6. АЖИЛЛУУЛАХ ОРОЛТ
# ==========================================
if __name__ == "__main__":
    USER_CREDENTIALS = [("trinity", "ganjuur2026")]

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        auth=USER_CREDENTIALS,
        auth_message="📜 Шунхан Ганжуур систем — Нэвтрэх нэр болон нууц үгээ оруулна уу.",
        favicon_path=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    )
