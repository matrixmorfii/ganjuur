"""ГАЛИГ API — сервер хадгалалт + шууд хайлт (FTS5).

longcat.py процессоос daemon thread болж 127.0.0.1:7861-д ажиллана.
Nginx `/api/` location энд proxy хийнэ. `db_lock`/`db_connection`-ийг
эх процессоос хуваалцана тул SQLite WAL + цоожийн дүрэм зөрчигдөхгүй.

Дүрмүүд (docs/GANJUUR_PROJECT_HANDOFF.md):
  * text_content-ийг ОГТ өөрчилдөггүй — зөвхөн UTF-8 баталгаажуулалт,
    INSERT-ийн дараа byte-exact round-trip шалгалт.
  * FTS5 бол зөвхөн хоёрдогч индекс; хадгалалтад нөлөөлөхгүй.
  * FTS5 боломжгүй SQLite байвал LIKE fallback-тэй.

Endpoint-ууд:
  GET  /api/translit/all              — public; {pageId: text} (сүүлийн хувилбар)
  GET  /api/translit/search?q=...     — public; FTS5 prefix хайлт + snippet
  POST /api/translit/save             — X-Translit-token header шаардана
"""

import hmac
import logging
import os
import sqlite3  # noqa: F401  (баримтжуулалт/db_connection-ийн төрөл)
import threading
from datetime import datetime, timezone

log = logging.getLogger("translit_api")

FTS_OK: bool | None = None  # init_fts() шийднэ


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FTS5 индекс
# ---------------------------------------------------------------------------
def init_fts(db_connection, db_lock) -> bool:
    """FTS5 virtual table + sync trigger үүсгэж, байгаа мөрүүдийг rebuild хийнэ.

    Амжилтгүй бол (FTS5гүй SQLite) FTS_OK=False — LIKE fallback ашиглагдана.
    """
    global FTS_OK
    try:
        with db_lock, db_connection() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS translit_fts USING fts5(
                    text_content,
                    source_image UNINDEXED,
                    content='transliterations',
                    content_rowid='id',
                    tokenize='unicode61'
                )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS translit_fts_ai
                AFTER INSERT ON transliterations BEGIN
                    INSERT INTO translit_fts(rowid, text_content, source_image)
                    VALUES (new.id, new.text_content, new.source_image);
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS translit_fts_ad
                AFTER DELETE ON transliterations BEGIN
                    INSERT INTO translit_fts(translit_fts, rowid, text_content, source_image)
                    VALUES ('delete', old.id, old.text_content, old.source_image);
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS translit_fts_au
                AFTER UPDATE ON transliterations BEGIN
                    INSERT INTO translit_fts(translit_fts, rowid, text_content, source_image)
                    VALUES ('delete', old.id, old.text_content, old.source_image);
                    INSERT INTO translit_fts(rowid, text_content, source_image)
                    VALUES (new.id, new.text_content, new.source_image);
                END
                """
            )
            # Идэмпотент: content table-аас индексийг бүрэн дахин барина.
            connection.execute("INSERT INTO translit_fts(translit_fts) VALUES('rebuild')")
        FTS_OK = True
        log.info("FTS5 индекс бэлэн (translit_fts)")
    except Exception as exc:
        FTS_OK = False
        log.warning("FTS5 боломжгүй (%s) — LIKE fallback ашиглана", exc)
    return FTS_OK


def _match_query(raw: str) -> str:
    """Хэрэглэгчийн текстийг FTS5 MATCH query болгох (prefix, escape-тэй)."""
    parts = []
    for token in raw.split():
        escaped = token.replace('"', '""')
        parts.append(f'"{escaped}"*')
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Endpoint-ууд
# ---------------------------------------------------------------------------
def build_api(db_connection, db_lock, token: str | None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    api = FastAPI(title="Ganjuur Translit API", docs_url=None, redoc_url=None)

    @api.get("/api/translit/all")
    def get_all():
        """source_image бүрийн хамгийн сүүлийн галиг (public, унших)."""
        with db_lock, db_connection() as connection:
            rows = connection.execute(
                """
                SELECT t.source_image, t.text_content
                FROM transliterations t
                WHERE t.id = (
                    SELECT MAX(id) FROM transliterations
                    WHERE source_image = t.source_image
                )
                AND t.text_content != ''
                """
            ).fetchall()
        return {"pages": {source: text for source, text in rows}}

    @api.get("/api/translit/search")
    def search(q: str = "", limit: int = 20):
        """Галигаар хайх: FTS5 prefix MATCH, боломжгүй бол LIKE."""
        q = (q or "").strip()
        if not q:
            return {"results": [], "mode": "none"}
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 20
        with db_lock, db_connection() as connection:
            if FTS_OK:
                try:
                    rows = connection.execute(
                        """
                        SELECT source_image,
                               snippet(translit_fts, 0, '', '', '…', 12) AS snip
                        FROM translit_fts
                        WHERE translit_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (_match_query(q), limit),
                    ).fetchall()
                    return {
                        "results": [{"page": page, "snippet": snip} for page, snip in rows],
                        "mode": "fts",
                    }
                except Exception as exc:
                    log.warning("FTS хайлт амжилтгүй (%s) — LIKE fallback", exc)
            like = "%" + q.replace("%", "").replace("_", "") + "%"
            rows = connection.execute(
                """
                SELECT source_image,
                       substr(text_content, max(1, instr(text_content, ?) - 40), 160)
                FROM transliterations
                WHERE id IN (SELECT MAX(id) FROM transliterations GROUP BY source_image)
                  AND text_content LIKE ?
                LIMIT ?
                """,
                (q, like, limit),
            ).fetchall()
        return {
            "results": [{"page": page, "snippet": snip} for page, snip in rows],
            "mode": "like",
        }

    @api.post("/api/translit/save")
    async def save(request: Request):
        """Галиг хадгалах — token шаардлагатай, byte-exact round-trip-тэй."""
        if token is None:
            return JSONResponse({"error": "write disabled (GANJUUR_TRANS_TOKEN unset)"}, status_code=503)
        presented = request.headers.get("x-translit-token", "")
        if not presented or not hmac.compare_digest(presented, token):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400)
        page = str(body.get("page", "")).strip()
        text = body.get("text")
        if not page or len(page) > 200:
            return JSONResponse({"error": "bad page id"}, status_code=400)
        if text is None or text == "":
            return JSONResponse({"error": "empty text"}, status_code=400)
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return JSONResponse({"error": "text is not valid UTF-8"}, status_code=400)

        with db_lock, db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO transliterations
                    (timestamp, source_image, text_content, token_count, char_count, status)
                VALUES (?, ?, ?, ?, ?, 'VERIFIED')
                """,
                (utc_now(), page, text, len(text.split()), len(text)),
            )
            new_id = cursor.lastrowid
            saved = connection.execute(
                "SELECT text_content FROM transliterations WHERE id = ?", (new_id,)
            ).fetchone()[0]
        if saved != text:
            log.error("Round-trip шалгалт амжилтгүй: id=%s", new_id)
            return JSONResponse({"error": "round-trip verification failed"}, status_code=500)
        return {"ok": True, "id": new_id}

    return api


# ---------------------------------------------------------------------------
# Эхлүүлэлт
# ---------------------------------------------------------------------------
def start_api_thread(db_connection, db_lock, host: str = "127.0.0.1", port: int | None = None) -> threading.Thread:
    """longcat.py-ийн __main__-с дуудах ганц орох цэг."""
    import uvicorn

    token = os.getenv("GANJUUR_TRANS_TOKEN") or None
    port = port or int(os.getenv("GANJUUR_API_PORT", "7861"))
    init_fts(db_connection, db_lock)
    api = build_api(db_connection, db_lock, token)
    thread = threading.Thread(
        target=lambda: uvicorn.run(api, host=host, port=port, log_level="warning"),
        name="translit-api",
        daemon=True,
    )
    thread.start()
    write_state = "идэвхтэй (token тохируулсан)" if token else "ИДЭВХГҮЙ (GANJUUR_TRANS_TOKEN алга)"
    log.info("Галиг API: http://%s:%s — бичилт %s", host, port, write_state)
    return thread
