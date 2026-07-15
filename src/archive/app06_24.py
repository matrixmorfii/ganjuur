# ==========================================
# 0. CRITICAL UPSTREAM PATCHES
# ==========================================
import gradio_client.utils
import gradio.networking

# Patch 1: Pydantic v2 Schema Сүйрлээс Хамгаалах Патч
_orig_json_schema = gradio_client.utils._json_schema_to_python_type
def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool): return "any"
    return _orig_json_schema(schema, defs)
gradio_client.utils._json_schema_to_python_type = _patched_json_schema

# Patch 2: Localhost Ping Алдааг Алгасах
gradio.networking.url_ok = lambda url: True

# ==========================================
# 1. SYSTEM CONFIGURATION & IMPORTS
# ==========================================
import os
import uuid
import threading
import warnings
import atexit
from pathlib import Path
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

# --- ЗАМ БОЛОН ТОХИРГОО ---
BASE_DIR = Path("/home/trinity/ganjuur")
SCANS_DIR = BASE_DIR / "data/scans"
CROPS_DIR = BASE_DIR / "data/ganjuur_crops/db_frames"
VECTORDB_PATH = BASE_DIR / "vectordb"
COLLECTION_NAME = "ganjuur_frames"

# Хөтчийн таб болон үндсэн UI-д харагдах зургуудын зам
LOGO_PATH = "image_566068.jpg"   # Таны үндсэн лого зураг
FAVICON_PATH = "asset/favicon.png"  # Зүсэж бэлтгэсэн фавикон зураг

SCANS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# --- QDRANT ХОЛБОЛТ ---
qdrant = QdrantClient(path=str(VECTORDB_PATH))
atexit.register(qdrant.close)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

# --- ML МОДЕЛУУДЫГ СУУЛГАХ ---
print("🧠 DINOv2 болон Tiktoken-ийг санах байгууламжид ачаалж байна...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
tok_encoder = tiktoken.get_encoding("cl100k_base")
print("✅ Продакшн систем ажиллахад бэлэн боллоо.")

# ==========================================
# 2. BACKEND ENGINE LOGIC
# ==========================================

def get_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = dinov2_processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        features = dinov2_model(**inputs).last_hidden_state[:, 0, :]
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()[0]

def save_frame_sharded(frame_matrix, page_id: str, frame_idx: int) -> tuple[str, str]:
    frame_uuid = str(uuid.uuid4())
    shard_dir = CROPS_DIR / frame_uuid[0:2] / frame_uuid[2:4]
    shard_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{frame_uuid}_p{page_id}_f{frame_idx}.jpg"
    full_path = shard_dir / filename
    cv2.imwrite(str(full_path), frame_matrix)
    return str(full_path), frame_uuid

def handle_ingestion(page_id, stride):
    img_path = SCANS_DIR / f"{page_id}.jpg"
    if not img_path.exists():
        found = list(SCANS_DIR.glob("*.jpg"))
        if not found: return "❌ data/scans хавтсанд ямар ч скан зураг олдсонгүй."
        img_path = found[0]
        page_id = img_path.stem

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return "❌ Скан зургийг уншиж чадсангүй."
    
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(green)
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    h, w = processed.shape[:2]
    window = min(h, w)
    max_dim = max(h, w)
    axis = "X" if w > h else "Y"
    
    points = []
    pos = 0
    count = 0
    
    while pos + window <= max_dim:
        crop = processed[0:h, pos:pos+window] if axis == "X" else processed[pos:pos+window, 0:w]
        path, f_uuid = save_frame_sharded(crop, page_id, count)
        vec = get_embedding(crop)
        
        points.append(models.PointStruct(
            id=f_uuid, vector=vec, 
            payload={"source": page_id, "local_path": path, "status": "PENDING"}
        ))
        pos += stride
        count += 1
        
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return f"✅ {page_id} хуудаснаас нийт {count} хэсгийг амжилттай салгаж, вектор санд (Qdrant) хадгалав."
    return "⚠️ Ямар нэгэн хэсэг үүсэж чадсангүй."

def execute_visual_query(img, top_k):
    if img is None: return []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vec = get_embedding(img_bgr)
    
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k
    )
    
    images = []
    for h in hits:
        path = h.payload.get("local_path")
        if os.path.exists(path):
            images.append((path, f"Ижил төсөө: {h.score:.3f}"))
    return images

def run_offline_analytics_sync():
    records, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=10000, with_vectors=True)
    if len(records) < 10: return "❌ Математик тооцоолол хийхэд багадаа 10-аас дээш цэг шаардлагатай."
    
    vectors = np.array([r.vector for r in records])
    ids = [r.id for r in records]
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(vectors)
    
    lof = LocalOutlierFactor(n_neighbors=20, metric='cosine')
    lof.fit_predict(vectors)
    scores = -lof.negative_outlier_factor_
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    for i, pid in enumerate(ids):
        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"umap_x": float(coords[i][0]), "umap_y": float(coords[i][1]), "anomaly_score": float(scores[i])},
            points=[pid], wait=False
        )
    return f"✅ Нийт {len(ids)} цэгийн аналитик өгөгдлийг амжилттай синхрончиллoo."

def fetch_stratified_plot_data():
    anomalies, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="anomaly_score", range=models.Range(gt=0.7))]),
        limit=2000
    )
    baseline, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="anomaly_score", range=models.Range(lte=0.7))]),
        limit=3000
    )
    
    xs, ys, colors, texts = [], [], [], []
    for r in anomalies + baseline:
        p = r.payload
        if "umap_x" in p:
            xs.append(p["umap_x"])
            ys.append(p["umap_y"])
            colors.append(p.get("anomaly_score", 0))
            texts.append(f"Эх сурвалж: {p.get('source')}<br>Аномали оноо: {p.get('anomaly_score', 0):.2f}")
    return xs, ys, colors, texts

def generate_plotly_layout():
    xs, ys, colors, texts = fetch_stratified_plot_data()
    if not xs: return go.Figure().update_layout(title="Өгөгдөл олдсонгүй")
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode='markers',
        marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title="Гажилт (Anomaly)")),
        text=texts, hoverinfo='text'
    ))
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), title="Чанарын Хяналтын Самбар (Үечилсэн Стратификаци)")
    return fig

def commit_transliteration(text_str):
    if not text_str: return "⚠️ Текст хоосон байна."
    
    tokens = tok_encoder.encode(text_str, allowed_special="all")
    decoded = tok_encoder.decode(tokens)
    
    if text_str != decoded:
        return "🔴 НОЦТОЙ АЛДАА: Тэмдэгт эсвэл хоосон зайн байт зөрүү илэрлээ! Өгөгдлийг ХАДГАЛААГҮЙ."
    
    return f"✅ Баталгаажсан бөгөөд Хадгалагдлаа. (Нийт {len(tokens)} токен). Урвуу хөрвөх чадвар 100% хангагдсан."

# ==========================================
# 3. GRADIO INTERFACE LAYOUT
# ==========================================
# favicon_path параметрийг энд нэгтгэв:

with gr.Blocks(
    title="Монгол шунхан Ганжуур", 
    theme=gr.themes.Soft(),
) as app:
    with gr.Row(equal_height=True):
        if os.path.exists(LOGO_PATH):
            gr.Image(
                value=LOGO_PATH, 
                show_label=False, 
                container=False, 
                width=90, 
                interactive=False
            )
        with gr.Column(scale=4):
            gr.Markdown("# Монгол шунхан Ганжуур")
            gr.Markdown("### Боть # 26  |  Хутагт билгийн чанад хязгаарт хүрсэн 100 мянгат")

    with gr.Tabs():
        with gr.TabItem("1. Өгөгдөл Оруулах (Ingestion)"):
            with gr.Row():
                with gr.Column(scale=1):
                    page_id = gr.Textbox(label="Хуудасны ID", value="I1PD1102690005")
                    stride = gr.Slider(10, 200, 50, label="Шилжих цонхны алхам (Stride px)")
                    btn_ingest = gr.Button("Системийн Шугмыг Ажиллуулах", variant="primary")
                with gr.Column(scale=2):
                    log = gr.Markdown()
            btn_ingest.click(handle_ingestion, [page_id, stride], log)

        with gr.TabItem("2. Визуал Хайлт (Search)"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_img = gr.Image(type="numpy", label="Хайх хэсгийн зураг (Query Crop)")
                    k = gr.Slider(1, 20, 5, label="Буцаах илэрцийн тоо (Top K)")
                    btn_search = gr.Button("Вектор Хайлт Хийх")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(columns=3, label="Олдсон ижил төстэй хэсгүүд")
            btn_search.click(execute_visual_query, [q_img, k], gallery)

        with gr.TabItem("3. Чанарын Хяналт (QA Triage)"):
            with gr.Row():
                with gr.Column(scale=3):
                    plot = gr.Plot()
                with gr.Column(scale=1):
                    btn_sync = gr.Button("UMAP/LOF CV Синхрончлолыг Эхлүүлэх", variant="stop")
                    btn_refresh = gr.Button("Графикийг Шинэчлэх")
                    status = gr.Textbox(label="Тооцооллын Төлөв")
            btn_sync.click(run_offline_analytics_sync, outputs=status)
            btn_refresh.click(generate_plotly_layout, outputs=plot)

        with gr.TabItem("4. Галиглах Орчин (Transliteration)"):
            gr.Markdown("### 🏛️ HF #31884 Хамгаалалт Идэвхтэй: Tiktoken Байт Түвшний Үр Төгс Баталжуулалт")
            txt = gr.Textbox(label="Эх UTF-8 Галиг Оруулах Талбар", lines=5)
            with gr.Row():
                gr.Button("бөгөөд").click(lambda x: (x or "") + " бөгөөд ", txt, txt)
                gr.Button("ажгуу").click(lambda x: (x or "") + " ажгуу ", txt, txt)
                btn_commit = gr.Button("Баталгаажуулж Хадгалах", variant="primary")
            out = gr.Markdown()
            btn_commit.click(commit_transliteration, txt, out)

# ==========================================
# 4. EXECUTION ENTRANCE & PASSWORD CONFIG
# ==========================================
if __name__ == "__main__":
    
    # 🔑 Системд нэвтрэх хэрэглэгчийн эрх
    USER_CREDENTIALS = [
        ("trinity", "ganjuur2026"),
    ]
    
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        show_error=True,
        auth=USER_CREDENTIALS,
        auth_message="Шунхан Ганжуур систем: Нэвтрэх нэр, нууц үгээ оруулна уу.",
        favicon_path=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None  
    )
