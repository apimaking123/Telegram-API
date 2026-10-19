"""
Telegram Data Search API (Full Dataset + Master DB + Gradio)
Author : @Cosmos_ownerr | Channel: @cosmosxinfo
"""

import asyncio
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import time

import duckdb
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Request, Response

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
PARALLELISM = int(os.environ.get("TG_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TG_THREADS", "16"))
RATE_LIMIT_PER_MIN = int(os.environ.get("TG_RATE_LIMIT", "500"))

DEVELOPER = "@Cosmos_ownerr"
CHANNEL = "@cosmosxinfo"

# Base URL of the dataset
BASE_URL = "https://huggingface.co/datasets/Saksham4540/Telegram-DB/resolve/main"

# ═══════════════════════════════════════════════════════════════════════════
#  CACHE & RATE LIMIT
# ═══════════════════════════════════════════════════════════════════════════
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_hits: dict[str, list[float]] = defaultdict(list)
_hits_lock = threading.Lock()

def _rate_ok(ip: str) -> bool:
    now = time()
    with _hits_lock:
        _hits[ip] = [t for t in _hits[ip] if now - t < 60]
        if len(_hits[ip]) >= RATE_LIMIT_PER_MIN:
            return False
        _hits[ip].append(now)
        return True

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded: return forwarded.split(",")[0].strip()
    return request.headers.get("X-Real-IP", "").strip() or (request.client.host if request.client else "unknown")

# ═══════════════════════════════════════════════════════════════════════════
#  DUCKDB
# ═══════════════════════════════════════════════════════════════════════════
_main_conn: duckdb.DuckDBPyConnection | None = None
_main_lock = threading.Lock()
pool = ThreadPoolExecutor(max_workers=PARALLELISM)

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except: pass
        
    con.execute("SET enable_http_metadata_cache=true")
    con.execute("SET enable_object_cache=true")
    con.execute("SET prefetch_all_parquet_files=true")
    con.execute("SET http_keep_alive=true")
    con.execute("SET http_timeout=300000")
    con.execute(f"SET threads={THREADS_PER_CONN}")
    con.execute("SET memory_limit='6GB'")
    
    # ✅ FIX: Allow wildcards (*) in HTTP paths
    con.execute("SET allow_asterisks_in_http_paths=true")

    if HF_TOKEN:
        try: con.execute(f"CREATE OR REPLACE SECRET hf_secret (TYPE huggingface, PROVIDER config, TOKEN '{HF_TOKEN}')")
        except: pass

    # Double wildcard to get all subfolders and all data files
    parts_pattern = f"{BASE_URL}/TG_DATA_PARTS/*/data_*.parquet"
    master_file = f"{BASE_URL}/TELEGRAM_MASTER_DB.parquet"
    
    con.execute(f"""
        CREATE OR REPLACE VIEW tg_data AS 
        SELECT * FROM read_parquet(['{parts_pattern}', '{master_file}'])
    """)
    return con

def _get_main_conn() -> duckdb.DuckDBPyConnection:
    global _main_conn
    if _main_conn is None:
        with _main_lock:
            if _main_conn is None: _main_conn = _new_conn()
    return _main_conn

def _warmup():
    try:
        con = _get_main_conn()
        print("[warmup] Loading metadata for ALL parts + master DB...")
        con.execute("SELECT COUNT(*) FROM tg_data LIMIT 1").fetchone()
        print("[warmup] ✅ TG Data ready")
    except Exception as e:
        print(f"[warmup error] {e}")

def _search_core(q_val: str, limit: int) -> dict:
    cache_key = f"tg|{q_val}|{limit}"
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    con = _get_main_conn()
    try:
        if q_val.isdigit() and len(q_val) >= 8:
            sql = f"SELECT * FROM tg_data WHERE phone = '{q_val}' LIMIT {limit}"
        else:
            sql = f"""
                SELECT * FROM tg_data 
                WHERE username ILIKE '%{q_val}%' 
                   OR first_name ILIKE '%{q_val}%' 
                   OR last_name ILIKE '%{q_val}%'
                LIMIT {limit}
            """
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        
        results = []
        for r in rows:
            row_dict = dict(zip(cols, r))
            clean_row = {k: v for k, v in row_dict.items() if v is not None and str(v).strip() != ""}
            if clean_row: results.append(clean_row)
    except Exception as e:
        print(f"SQL Error: {e}")
        results = []

    response_data = {
        "success": len(results) > 0,
        "status": "Data found ✅" if results else "Data not available 🙅",
        "query": q_val,
        "results": results,
        "developer": DEVELOPER,
        "channel": CHANNEL
    }
    
    with _cache_lock:
        _cache[cache_key] = response_data
        
    return response_data

# ═══════════════════════════════════════════════════════════════════════════
#  FASTAPI
# ═══════════════════════════════════════════════════════════════════════════
fastapi_app = FastAPI(title="Telegram Data API")

@fastapi_app.get("/")
def root():
    return {"app": "Telegram Data API", "records": "Master + All Parts", "developer": DEVELOPER}

@fastapi_app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "warmup_done": _main_conn is not None}

@fastapi_app.get("/tg")
async def search_tg(
    request: Request,
    q: str = Query(..., description="Search term"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True)
):
    t_start = time()
    ip = _client_ip(request)
    if not _rate_ok(ip):
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT_PER_MIN} req/min")

    q_val = q.strip()
    response_data = _search_core(q_val, limit)
    
    elapsed = f"{round(time() - t_start, 1)} seconds"
    response_data["response_time"] = elapsed

    return Response(
        content=json.dumps(response_data, indent=2 if pretty else None, ensure_ascii=False),
        media_type="application/json"
    )

@fastapi_app.on_event("startup")
async def _startup():
    loop = asyncio.get_running_loop()
    loop.run_in_executor(pool, _warmup)

# ═══════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════
def _tg_ui(query: str, limit: int):
    if not query or not query.strip():
        return "⚠️ Enter a number or name."
    
    t_start = time()
    data = _search_core(query.strip(), int(limit))
    elapsed = f"{round(time() - t_start, 1)} seconds"
    
    if not data["results"]:
        return f"🔍 **Query:** `{query}`\n⏱️ **Response:** {elapsed}\n\n🙅 **Data not available.**"

    header = (
        f"🔍 **Query:** `{query}`  |  **Found:** {len(data['results'])}\n"
        f"✅ **Data found**\n"
        f"⏱️ **Response:** {elapsed}\n"
        f"👨‍💻 **Dev:** {DEVELOPER}  |  📢 **Channel:** {CHANNEL}\n\n---\n\n"
    )
    
    parts = []
    for i, r in enumerate(data["results"], 1):
        lines = [f"### Result {i}"]
        for k, v in r.items():
            lines.append(f"**{k.replace('_', ' ').title()}:** {v}")
        parts.append("\n\n".join(lines))
        
    return header + "\n\n---\n\n".join(parts)

def _build_ui():
    with gr.Blocks(title="Cosmos Telegram Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Telegram Data Search")
        gr.Markdown("Search **859M+ records** by phone number, username, or name")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", placeholder="Phone, username, name...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(1, 50, 10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=_tg_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=_tg_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown(f"---\n<div style='text-align:center'>👨‍💻 <b>{DEVELOPER}</b> | 📢 <b>{CHANNEL}</b></div>")
    return demo

demo = _build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/tg/ui")
