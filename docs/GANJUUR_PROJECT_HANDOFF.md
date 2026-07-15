# Ganjuur Production Handoff Prompt

## Role and goal

You are maintaining **Ganjuur**, a production Digital Humanities application for finding visually similar, rare, or unusual forms in Mongolian woodblock manuscript scans and recording human transliterations.

The non-negotiable product goal is:

> Help a researcher find forgotten or unusual manuscript forms quickly, then preserve the human-entered transcription exactly.

Treat this document as a production handoff. Inspect the live files and service status before changing anything. Do not replace working architecture with an unverified rewrite.

## Current verified production state

As of the latest verified deployment:

- The web app is running and responds on LAN at `http://192.168.0.55:7860`.
- The app is managed by systemd as `ganjuur.service`, enabled to start after reboot and configured to restart after crashes.
- Qdrant runs in Docker as `qdrant_ganjuur`.
- Qdrant is bound to `127.0.0.1:6333` and `127.0.0.1:6334`, not exposed directly to the LAN/public internet.
- The Python application uses Docker Qdrant over HTTP port 6333.
- CUDA is detected and DINOv2 runs on the NVIDIA RTX 3060.
- The app uses the Mongolian Cyrillic-first interface, restored logo, and in-app user manual.

Verified Docker Qdrant collection counts:

| Collection | Verified point count |
| --- | ---: |
| `ganjuur_frames` | 60,900 |
| `ganjuur_genealogy` | 4,784 |
| `ganjuur_words` | 10,268 |

The `ganjuur_frames` collection has 768-dimensional COSINE vectors and payload indexes for `umap_x`, `anomaly_score`, `status`, and `source`.

## Server and file layout

- Server OS: Xubuntu / Ubuntu family
- Application directory: `/home/trinity/ganjuur`
- Main application: `/home/trinity/ganjuur/gpt.py`
- Virtual environment: `/home/trinity/ganjuur/ganjuur_env`
- Transcription database: `/home/trinity/ganjuur/transliterations.db`
- Scan data: `/home/trinity/ganjuur/data/`
- Sharded WebP frames: `/home/trinity/ganjuur/data/ganjuur_crops/db_frames/`
- Current Docker Qdrant storage: `/home/trinity/ganjuur/qdrant_storage/`
- Original local Qdrant store: `/home/trinity/ganjuur/vectordb/`
- Full safety backup of the original local store: `/home/trinity/ganjuur/vectordb_pre_docker_backup/`
- Protected environment file: `/home/trinity/ganjuur/.env` (mode 0600)
- Mongolian user manual: `/home/trinity/ganjuur/ganjuur_gariin_avlaga.html`
- System service: `/etc/systemd/system/ganjuur.service`

Never put secrets, passwords, API keys, or the contents of `.env` into commits, logs, prompts, screenshots, or chat responses.

## Critical protection rules

1. **Do not delete, reset, overwrite, move, or mount over `vectordb` or `vectordb_pre_docker_backup`.** They are the original local data and safety backup.
2. **Do not create an empty replacement collection under the name `ganjuur_frames`.** Verify counts before any collection-changing action.
3. **Do not mount the old local `vectordb` folder directly into Docker Qdrant.** Local Qdrant storage and server storage must be migrated by API export/import, not by a direct storage-folder swap.
4. **Do not use Qdrant local mode concurrently from multiple processes.** The app may fall back to local mode only if Docker is unavailable; analytics must be disabled in that mode.
5. **Do not remove the compatibility patches at the beginning of `gpt.py`.** They must execute before other application imports.
6. **Do not use Hugging Face text tokenizers for transcription validation or text round-tripping.**
7. **Do not normalize, trim, tokenize, clean punctuation, or alter transcription text before storage or export.**
8. **Do not replace the running systemd service with a screen-only deployment.** Screen is debugging-only; systemd is the production runtime.
9. **Do not upgrade Gradio, Qdrant client/server, Torch, Transformers, or CUDA-related packages casually.** This project uses compatibility patches and must be version-tested first.
10. Before changing production data or deployment configuration, make a backup and state the rollback plan.

## How gpt.py is designed

### Required upstream compatibility patches

At the top of `gpt.py`, before other imports:

- Patch Gradio Pydantic-v2 boolean JSON schema handling.
- Patch the Gradio localhost reachability check for LAN/Nginx deployment.

These patches are version-sensitive. Preserve their location and behavior unless an exact tested dependency upgrade makes them unnecessary.

### Environment settings

`gpt.py` loads simple `KEY=VALUE` settings from `.env` without overriding already-provided system environment variables.

Important settings include:

- `GANJUUR_USER`
- `GANJUUR_PASS`
- `QDRANT_HOST=127.0.0.1`
- `QDRANT_PORT=6333`
- `QDRANT_GRPC_PORT=6334`
- `GRADIO_ANALYTICS_ENABLED=False`

The systemd service reads the same protected `.env` file.

### Qdrant behavior

`make_qdrant_client()` works in this order:

1. Connect to Docker Qdrant over HTTP on port 6333.
2. If Docker is unavailable, safely fall back to the existing local `vectordb` only if it exists.
3. If neither backend is available, fail with a clear error without changing data.

The fallback exists for recovery. It is not the intended normal production mode.

### Image ingestion

The ingestion workflow:

1. Accepts a ZIP of JPG/JPEG/PNG page scans.
2. Uses bounded, one-image-at-a-time processing rather than decoding every image at once.
3. Checks file count, compressed/uncompressed size, compression ratio, per-image pixels, and total decoded pixels.
4. Uses green-channel cinnabar extraction and CLAHE.
5. Produces sliding-window frames.
6. Saves WebP frames in UUID-sharded directories.
7. Embeds frames in bounded DINOv2 batches.
8. Upserts vectors and payloads to Qdrant.
9. Removes newly written frames from a failed batch to avoid orphan files.

Do not remove the resource limits or streaming behavior.

### Embeddings and GPU

- Model: `facebook/dinov2-base`
- Vector size: 768
- Device: CUDA when available
- Uses `model.eval()`, `torch.inference_mode()`, FP16 autocast on CUDA, L2 normalization, and a model lock.
- Initial batch size: 32

Do not silently change model/preprocessing/vector dimension. Any model or preprocessing change requires a new versioned collection and reindexing plan.

### Analytics

UMAP and LocalOutlierFactor analytics run in a spawned multiprocessing worker when Docker Qdrant is active.

This design prevents the web process from running heavy CPU analytics and avoids CUDA/DINOv2 loading in the analytics worker.

Job state is stored in SQLite table `analytics_jobs`.

Important safety behavior:

- Analytics is deliberately blocked when the app is using local Qdrant fallback.
- Do not re-enable multiprocessing analytics against a local Qdrant data folder. That would risk concurrent local-store access.
- UMAP settings: `n_neighbors=15`, `min_dist=0.1`, `metric="cosine"`, `random_state=42`.
- LOF settings: `n_neighbors=20`, `metric="cosine"`.
- Anomaly scores are normalized to 0-1 and stored in Qdrant payload.

### Transliteration integrity

Transliteration text is scholarly data.

The application:

- Validates UTF-8 without mutating the text.
- Does not use tokenizer encode/decode.
- Stores the exact submitted text in SQLite.
- Reads the value back after insert and compares it byte-for-byte as a Python string.
- Refuses to claim preservation if the round-trip check fails.

SQLite requirements:

- `check_same_thread=False`
- WAL mode
- `synchronous=NORMAL`
- global lock around database operations
- indexes for frequent fields

## User interface requirements

The interface is intentionally Mongolian Cyrillic-first and should remain simple for non-technical researchers.

Visible user-facing requirements:

- Show the existing logo from `image_566068.jpg`.
- Do not brand the user interface as “AI”.
- Avoid exposing DINOv2, Qdrant, UMAP, GPU, vector database, or other implementation terms unless in an administrator/developer-only area.
- Use these plain workflow concepts:
  1. Zurg oruulah (upload scans)
  2. Durseeree haih (search visually)
  3. Ontsgoi helber (review unusual forms)
  4. Galig oruulah (enter transliteration)
- Keep the HTML manual link near the top of the page:
  - visible text: `Delgerengui gariin avlaga neeh`
  - target: `ganjuur_gariin_avlaga.html`
- The manual must remain protected by the same app login.
- Preserve the suffix helper buttons: `bogoood`, `ajguu`, `mon`, `ted`.

## Production operations

Use these checks before and after production changes:

```bash
sudo systemctl status ganjuur.service
sudo journalctl -u ganjuur.service -n 100 --no-pager
docker ps
curl -fsS http://127.0.0.1:6333/collections
curl -I http://127.0.0.1:7860/
```

Expected healthy state:

- `ganjuur.service` is active/running.
- `qdrant_ganjuur` is up.
- Qdrant reports all three collections and the verified counts above.
- Port 7860 listens locally/on LAN.
- Nginx, if configured, proxies browser traffic to the application.
- Docker Qdrant ports remain local-only unless there is a deliberate secured networking plan.

## Safe next improvements

Good next work, in order:

1. Test a complete browser workflow: login, ZIP ingestion with a small archive, visual search, analytics, transliteration save, and manual link.
2. Add tested export for research data (CSV/JSON) with exact text-preservation checks.
3. Add database/Qdrant/frame backup and restore scripts with a documented restoration test.
4. Configure Nginx domain routing, TLS, WebSocket proxying, and upload limits only after confirming the intended domain name.
5. Add user management beyond one Gradio login.
6. Add health monitoring for disk space, Docker Qdrant, systemd service, CUDA, and backup age.
7. Pin all Python and Docker versions in a lock file after recording the currently working versions.

## Required change process for another AI

Before changing anything:

1. Read this handoff file.
2. Read the relevant current part of `gpt.py`; do not assume this document replaces the source.
3. Check service and Qdrant status.
4. State exactly what files/services will change and the rollback path.
5. Make the smallest safe change.
6. Verify the app, vector counts, and logs afterwards.
7. Report the result in plain language.

If a requested change could affect vectors, text integrity, auth, the live service, or deployment, ask for explicit approval before applying it.

## Definition of success

The result must continue to provide:

- Mongolian-language, non-technical workflow for researchers.
- Intact visual search over the existing 60,900 manuscript frames.
- Exact preservation of transliteration text.
- Docker Qdrant as the normal production backend.
- Automatic app restart through systemd.
- Existing local vector store and backup kept untouched.
- Clear user manual accessible from the application.


## Approved production follow-ups from code review

The following improvements were identified after the initial production deployment. Apply them carefully, with tests and a service restart verification.

### 1. Analytics worker backend state and isolation

The analytics worker runs in a separate spawned process. Python globals are isolated per process.

Current implication:

- If `analytics_worker()` calls `make_qdrant_client()` and mutates `qdrant_backend`, that mutation exists only in the child worker.
- The parent Gradio process will not receive that updated global value.
- The user interface should not rely on a child-process global to report which backend the worker used.

Required improvement:

- The analytics worker must use Docker Qdrant only.
- Do not allow an analytics worker to silently fall back to local Qdrant, because this can create unsafe concurrent local-store access.
- Store the worker backend and progress/status directly in the `analytics_jobs` SQLite record, for example with a `backend` column or a message such as `Running on Docker Qdrant`.
- Keep the existing parent-side safeguard that blocks analytics if the main application is in local fallback mode.

### 2. SQLite PRAGMA placement

The original suggestion to move both PRAGMAs exclusively into `init_db()` is only partly correct.

Correct rule:

- `PRAGMA journal_mode=WAL;` is persistent for the database file. It should be set during `init_db()`.
- `PRAGMA synchronous=NORMAL;` is connection-specific. Keep it in `db_connection()` so every new connection uses the intended safety/performance setting.
- Add a reasonable `PRAGMA busy_timeout` per connection to reduce temporary lock errors under normal concurrent access.

Do not remove the global database lock or WAL mode without a tested replacement.

### 3. Plot payload minimization

`scroll_all_analytics_points()` currently does not need every payload field for map rendering.

Required improvement:

```python
with_payload=["umap_x", "umap_y", "anomaly_score", "source"]
```

Use this restricted payload list when scrolling points for the Plotly map. The selected point can still be retrieved later by ID through `inspect_node()`, which is where full metadata such as the frame path belongs.

This reduces data transfer and UI refresh work as the collection grows beyond the current 60,900 frame vectors.
