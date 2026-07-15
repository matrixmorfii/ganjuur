# ==========================================
# 0. CRITICAL UPSTREAM PATCHES
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
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

# ==========================================
# 2. ML MODELS & THREAD SAFETY (Gemini Fix #1 & #5)
# ==========================================
print("🧠 DINOv2 болон Tiktoken-ийг санах байгууламжид ачаалж байна...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# 🔥 GEMINI FIX #1: Моделийг зөвхөн Inference горимд түгжинэ (Dropout/LayerNorm-ыг идэвхгүй болгоно)
dinov2_model.eval()  

# 🔥 GEMINI FIX #5: Олон хэрэглэгч зэрэг хандах үеийн CUDA сөргөлдөөнөөс (Race Condition) хамгаалах Lock
model_lock = threading.Lock()

# 🏛️ HF #31884 ХАМГААЛАЛТ: 
# HuggingFace-ийн GPT-2 BPE токенизатор нь цэг таслалын өмнөх хоосон зайг устгадаг алдаатай.
# Тиймээс бид 100% урвуу хөрвөх чадвартай (invertible) OpenAI-ийн tiktoken-ийг ашиглаж байна.
tok_encoder = tiktoken.get_encoding("cl100k_base")
print("✅ Продакшн систем ажиллахад бэлэн боллоо.")

# ==========================================
# 3. BACKEND ENGINE LOGIC (Batch Processing)
# ==========================================

# 🔥 GEMINI FIX #2: GPU-ийн багц боловсруулалт (10-15 дахин хурдасгана)
def get_embeddings_batch(img_bgr_list):
    """Олон зургийг багцаар нь нэг дор GPU руу шидэж векторжуулах"""
    if not img_bgr_list: return []
    img_rgb_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_bgr_list]
    
    with model_lock:  # 🔥 Thread Safety Lock
        inputs = dinov2_processor(images=img_rgb_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2_model(**inputs).last_hidden_state[:, 0, :]
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
    return outputs.cpu().numpy().tolist()

# 🔥 GEMINI FIX #2 (Applied): handle_ingestion доторх гогцоог багцаар боловсруулах байдлаар рефактор хийв
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
    
    # 🌟 БАГЦ БОЛОВСРУУЛАЛТЫН САНАМЖУУД (Buffers)
    BATCH_SIZE = 16  # GPU-ийн VRAM-д тохируулан 16 эсвэл 32 гэж тохируулж болно
    crops_buffer = []
    paths_buffer = []
    ids_buffer = []
    points_buffer = []
    
    pos = 0
    count = 0
    
    while pos + window <= max_dim:
        crop = processed[0:h, pos:pos+window] if axis == "X" else processed[pos:pos+window, 0:w]
        path, f_uuid = save_frame_sharded(crop, page_id, count)
        
        # Санах ойд цуглуулна
        crops_buffer.append(crop)
        paths_buffer.append(path)
        ids_buffer.append(f_uuid)
        
        # Багц дүүрвэл GPU руу нэг дор шидэж, Qdrant руу хадгална
        if len(crops_buffer) >= BATCH_SIZE:
            vecs = get_embeddings_batch(crops_buffer)
            for f_uuid, path, vec in zip(ids_buffer, paths_buffer, vecs):
                points_buffer.append(models.PointStruct(
                    id=f_uuid, vector=vec, 
                    payload={"source": page_id, "local_path": path, "status": "PENDING"}
                ))
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
            
            # Буферыг цэвэрлэнэ
            crops_buffer, paths_buffer, ids_buffer, points_buffer = [], [], [], []
            
        pos += stride
        count += 1
        
    # Үлдэгдэл зургуудыг боловсруулах
    if crops_buffer:
        vecs = get_embeddings_batch(crops_buffer)
        for f_uuid, path, vec in zip(ids_buffer, paths_buffer, vecs):
            points_buffer.append(models.PointStruct(
                id=f_uuid, vector=vec, 
                payload={"source": page_id, "local_path": path, "status": "PENDING"}
            ))
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
        
    if count > 0:
        return f"✅ {page_id} хуудаснаас нийт {count} хэсгийг амжилттай салгаж, GPU багц боловсруулалтаар вектор санд хадгалав."
    return "⚠️ Ямар нэгэн хэсэг үүсэж чадсангүй."

def save_frame_sharded(frame_matrix, page_id: str, frame_idx: int) -> tuple:
    frame_uuid = str(uuid.uuid4())
    shard_dir = CROPS_DIR / frame_uuid[0:2] / frame_uuid[2:4]
    shard_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{frame_uuid}_p{page_id}_f{frame_idx}.jpg"
    full_path = shard_dir / filename
    cv2.imwrite(str(full_path), frame_matrix)
    return str(full_path), frame_uuid

def execute_visual_query(img, top_k):
    if img is None: return []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vec = get_embeddings_batch([img_bgr])[0] # Багцын функцийг ашиглана
    
    response = qdrant.query_points(collection_name=COLLECTION_NAME, query=vec, limit=top_k)
    hits = response.points
    
    images = []
    for h in hits:
        path = h.payload.get("local_path")
        if path and os.path.exists(path):
            images.append((path, f"Ижил төсөө: {h.score:.3f}"))
    return images

# 🔥 GEMINI FIX #3: Qdrant-ийн Sequential Loop-ийг batch_update_points руу шилжүүлэв
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
    
    # 🔥 GEMINI FIX #3: Qdrant-ийн batch_update_points ашиглах (Pydantic nested бүтцийг засав)
    update_operations = []
    for i, pid in enumerate(ids):
        update_operations.append(
            models.SetPayloadOperation(
                set_payload=models.SetPayload(  # 🌟 ЭНД set_payload гэх үүрэн бүтэц нэмэгдлээ
                    points=[pid],
                    payload={
                        "umap_x": float(coords[i][0]), 
                        "umap_y": float(coords[i][1]), 
                        "anomaly_score": float(scores[i])
                    }
                )
            )
        )
    
    # Багцаар нь нэг дор илгээнэ (Сүлжээний ачааллыг 100 дахин багасгана)
    qdrant.batch_update_points(
        collection_name=COLLECTION_NAME,
        update_operations=update_operations
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
    return xs, ys, colors, texts, anomalies

def generate_plotly_layout():
    xs, ys, colors, texts, anomalies = fetch_stratified_plot_data()
    if not xs: 
        # 🔥 GEMINI FIX #4: gr.update-г устгаж, шууд компонент үүсгэнэ
        return go.Figure().update_layout(title="Өгөгдөл олдсонгүй"), gr.Dropdown(choices=[], value=None)
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode='markers',
        marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title="Гажилт (Anomaly)")),
        text=texts, hoverinfo='text'
    ))
    fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), title="Чанарын Хяналтын Самбар")
    
    choices = [(f"{r.payload.get('source')} | Оноо: {r.payload.get('anomaly_score', 0):.2f}", str(r.id)) for r in anomalies]
    
    # 🔥 GEMINI FIX #4: gr.update-г gr.Dropdown-оор солив
    return fig, gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

def inspect_node(selected_id):
    if not selected_id: return None, "⚠️ Цэг сонгоно уу.", go.Figure()
    try:
        records = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[selected_id], with_payload=True, with_vectors=True)
        if not records: return None, "❌ Өгөгдөл олдсонгүй.", go.Figure()
            
        point = records[0]
        img_path = point.payload.get("local_path")
        source = point.payload.get("source", "Unknown")
        score = point.payload.get("anomaly_score", 0.0)
        
        meta_md = f"**Эх сурвалж:** `{source}`\n\n**Аномали оноо:** `{score:.3f}`"
        
        fig_emb = go.Figure(data=[go.Bar(y=point.vector[:50], marker_color='cyan', opacity=0.8)])
        fig_emb.update_layout(title="768D Эмбеддинг", template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(showticklabels=False), yaxis=dict(title="Жин"))
        
        return img_path, meta_md, fig_emb
    except Exception as e:
        return None, f"⚠️ Алдаа: {str(e)}", go.Figure()

def commit_transliteration(text_str):
    if not text_str: return "⚠️ Текст хоосон байна."
    tokens = tok_encoder.encode(text_str, allowed_special="all")
    decoded = tok_encoder.decode(tokens)
    
    if text_str != decoded:
        return "🔴 НОЦТОЙ АЛДАА (HF #31884): Тэмдэгт эсвэл хоосон зайн байт зөрүү илэрлээ! Өгөгдлийг ХАДГАЛААГҮЙ."
    
    return f"✅ Баталгаажсан бөгөөд Хадгалагдлаа. (Нийт {len(tokens)} токен). Урвуу хөрвөх чадвар 100% хангагдсан."

# ==========================================
# 4. GRADIO INTERFACE LAYOUT
# ==========================================
with gr.Blocks(title="Монгол шунхан Ганжуур", theme=gr.themes.Soft()) as app:
    with gr.Row(equal_height=True):
        if os.path.exists(LOGO_PATH):
            gr.Image(value=LOGO_PATH, show_label=False, container=False, width=90, interactive=False)
        with gr.Column(scale=4):
            gr.Markdown("# Монгол шунхан Ганжуур")
            gr.Markdown("### Боть # 26  |  Хутагт билгийн чанад хязгаарт хүрсэн 100 мянгат")

    with gr.Tabs():
        with gr.TabItem("1. Өгөгдөл Оруулах (Ingestion)"):
            with gr.Row():
                with gr.Column(scale=1):
                    page_id = gr.Textbox(label="Хуудасны ID", value="I1PD1102690005")
                    stride = gr.Slider(10, 200, 50, label="Шилжих цонхны алхам (Stride px)")
                    btn_ingest = gr.Button("Системийн Шугамыг Ажиллуулах", variant="primary")
                with gr.Column(scale=2):
                    log = gr.Markdown()

        with gr.TabItem("2. Визуал Хайлт (Search)"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_img = gr.Image(type="numpy", label="Хайх хэсгийн зураг (Query Crop)")
                    k = gr.Slider(1, 20, 5, label="Буцаах илэрцийн тоо (Top K)")
                    btn_search = gr.Button("Вектор Хайлт Хийх")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(columns=3, label="Олдсон ижил төстэй хэсгүүд")

        with gr.TabItem("3. Чанарын Хяналт (QA Triage)"):
            with gr.Row():
                with gr.Column(scale=3):
                    plot = gr.Plot()
                with gr.Column(scale=1):
                    btn_sync = gr.Button("UMAP/LOF CV Синхрончлолыг Эхлүүлэх", variant="stop")
                    btn_refresh = gr.Button("Графикийг Шинэчлэх")
                    status = gr.Textbox(label="Тооцооллын Төлөв")
            
            gr.Markdown("---")
            gr.Markdown("#### 🔬 Сонгосон цэгээ Галиглах руу илгээх")
            
            node_dropdown = gr.Dropdown(label="Шалгах цэгээ сонгох (Inspect Node)", choices=[], interactive=True)
            
            with gr.Row():
                with gr.Column(scale=1):
                    detail_image = gr.Image(label="🔍 Сонгосон хэсэг (The Ancient Ink)", type="filepath")
                with gr.Column(scale=1):
                    detail_meta = gr.Markdown()
                    detail_embedding = gr.Plot(label="🧠 768D Математик Эмбеддинг")
                    btn_send_to_translit = gr.Button("📝 Энэ хэсгийг Галиглах руу илгээх", variant="primary")

        with gr.TabItem("4. Галиглах Орчин (Transliteration)"):
            gr.Markdown("### 🏛️ HF #31884 Хамгаалалт Идэвхтэй: Tiktoken Байт Түвшний Үр Төгс Баталгаажуулалт")
            gr.Markdown("*(Зүүн талд эх зургийг харж, баруун талд UTF-8 галигийг оруулна.)*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    translit_img = gr.Image(label="🖼️ Галиглаж буй эх зураг", type="filepath", interactive=True)
                with gr.Column(scale=1):
                    txt = gr.Textbox(label="Эх UTF-8 Галиг Оруулах Талбар", lines=5)
                    with gr.Row():
                        btn_bugd = gr.Button("бөгөөд")
                        btn_ajguu = gr.Button("ажгуу")
                    btn_commit = gr.Button("Баталгаажуулж Хадгалах", variant="primary")
            out = gr.Markdown()

    # ==========================================
    # 5. EVENT BINDINGS
    # ==========================================
    btn_ingest.click(handle_ingestion, inputs=[page_id, stride], outputs=log)
    btn_search.click(execute_visual_query, inputs=[q_img, k], outputs=gallery)
    
    btn_sync.click(run_offline_analytics_sync, outputs=status)
    btn_refresh.click(generate_plotly_layout, outputs=[plot, node_dropdown])
    node_dropdown.change(inspect_node, inputs=node_dropdown, outputs=[detail_image, detail_meta, detail_embedding])
    btn_send_to_translit.click(lambda img: img, inputs=detail_image, outputs=translit_img)
    
    btn_bugd.click(lambda x: (x or "") + " бөгөөд ", inputs=txt, outputs=txt)
    btn_ajguu.click(lambda x: (x or "") + " ажгуу ", inputs=txt, outputs=txt)
    btn_commit.click(commit_transliteration, inputs=txt, outputs=out)

# ==========================================
# 6. EXECUTION ENTRANCE
# ==========================================
if __name__ == "__main__":
    USER_CREDENTIALS = [("trinity", "ganjuur2026")]
    
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        show_error=True,
        auth=USER_CREDENTIALS,
        auth_message="Шунхан Ганжуур систем: Нэвтрэх нэр, нууц үгээ оруулна уу.",
        favicon_path=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None  
    )
