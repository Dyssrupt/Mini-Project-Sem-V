"""
main.py -- FastAPI backend for the operator dashboard.
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn backend.main:app --port 8000          (run from the project folder, next to sangli_nlp.py)
Env vars: CLASSIFIER=tfidf|bert   CAUTIOUS=0.75 (raise urgency when unsure)   WHISPER_MODEL=small
"""
import asyncio, json, os, random, shutil, tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline import Pipeline, ROOT, derive
from .store import Store

CLASSIFIER = os.getenv("CLASSIFIER", "bert"); CAUTIOUS = float(os.getenv("CAUTIOUS", "0.75")); TAGGER = os.getenv("TAGGER", "combo"); WHISPER = os.getenv("WHISPER_MODEL", "small")
state = {}
tasks = set()          # keep references so background jobs aren't garbage-collected


def spawn(coro):
    t = asyncio.create_task(coro); tasks.add(t); t.add_done_callback(tasks.discard); return t


class Hub:
    def __init__(self): self.clients = set()

    async def broadcast(self, event, data):
        msg, dead = json.dumps({"event": event, "data": data}), []
        for ws in list(self.clients):
            try: await ws.send_text(msg)
            except Exception: dead.append(ws)
        for ws in dead: self.clients.discard(ws)


hub = Hub()


@asynccontextmanager
async def lifespan(app):
    state["pipe"] = Pipeline(classifier=CLASSIFIER, tagger=TAGGER, whisper_model=WHISPER); state["store"] = Store(ROOT / "incidents.db")
    print(f"ready: classifier={CLASSIFIER} cautious={CAUTIOUS} tagger={TAGGER} whisper={WHISPER}")
    yield


app = FastAPI(title="AI Emergency Call Understanding System", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def push(iid):
    await hub.broadcast("incident", state["store"].get(iid))


async def process(iid, text=None, audio_path=None):
    pipe, store = state["pipe"], state["store"]
    try:
        if audio_path:
            store.update(iid, {"stage": "transcribing"}); await push(iid)
            text = await asyncio.to_thread(pipe.transcribe, audio_path)
        store.update(iid, {"transcript": text, "stage": "analysing"}); await push(iid)
        result = await asyncio.to_thread(pipe.analyse, text, CAUTIOUS)
        store.update(iid, {**result, "stage": "done"}); await push(iid)
    except Exception as e:
        store.update(iid, {"stage": "error", "error": str(e)}); await push(iid)
    finally:
        if audio_path: Path(audio_path).unlink(missing_ok=True)


class TextIn(BaseModel):
    text: str


class Patch(BaseModel):
    status: Optional[str] = None            # new | dispatched | resolved
    type_override: Optional[str] = None     # "" clears the override
    urgency_override: Optional[str] = None
    notes: Optional[str] = None


@app.get("/api/health")
def health(): return {"ok": True, "classifier": CLASSIFIER, "cautious": CAUTIOUS, "tagger": TAGGER, "whisper": WHISPER}


@app.get("/api/incidents")
def incidents(): return state["store"].list()


@app.get("/api/incidents/{iid}")
def incident(iid: int):
    r = state["store"].get(iid)
    if not r: raise HTTPException(404, "not found")
    return r


@app.post("/api/calls/text")
async def call_text(body: TextIn):
    if not body.text.strip(): raise HTTPException(400, "empty text")
    iid = state["store"].create("text", {"stage": "queued", "transcript": body.text}); await push(iid)
    spawn(process(iid, text=body.text)); return state["store"].get(iid)


@app.post("/api/calls/audio")
async def call_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "call.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp: shutil.copyfileobj(file.file, tmp)
    iid = state["store"].create("audio", {"stage": "queued", "audio_name": file.filename}); await push(iid)
    spawn(process(iid, audio_path=tmp.name)); return state["store"].get(iid)


@app.patch("/api/incidents/{iid}")
async def patch(iid: int, body: Patch):
    store = state["store"]; cur = store.get(iid)
    if not cur: raise HTTPException(404, "not found")
    upd = {k: v for k, v in body.model_dump(exclude_unset=True).items() if k != "status"}
    if body.status and body.status not in ("new", "dispatched", "resolved"): raise HTTPException(400, "bad status")
    merged = derive({**cur, **upd}) if cur.get("type") else {**cur, **upd}
    new = store.update(iid, {**upd, **{k: merged[k] for k in ("final_type", "final_urgency", "priority", "dispatch") if k in merged}}, status=body.status)
    await push(iid); return new


@app.delete("/api/incidents")
@app.delete("/api/incidents/")
@app.post("/api/incidents/clear")
async def delete_all_incidents():
    store = state["store"]
    count = store.delete_all()
    await hub.broadcast("all_incidents_deleted", {})
    return {"deleted": True, "count": count}


@app.delete("/api/incidents/{iid}")
async def delete_incident(iid: int):
    store = state["store"]
    cur = store.get(iid)
    if not cur:
        raise HTTPException(404, "not found")
    store.delete(iid)
    await hub.broadcast("incident_deleted", {"id": iid})
    return {"deleted": True, "id": iid}


@app.post("/api/demo/replay")
async def replay(n: int = 8, gap: float = 4.0):
    """Feed n random held-out TEST calls (ASR-noisy text) into the dashboard, one every `gap` seconds."""
    rows = random.sample(state["pipe"].test_rows, min(n, 50))

    async def run():
        for r in rows:
            iid = state["store"].create("demo", {"stage": "queued", "transcript": r["transcript_asr_noisy"],
                                                  "demo_truth": {"type": r["emergency_type"], "urgency": r["urgency_label"], "area": r["primary_area"]}})
            await push(iid); await process(iid, text=r["transcript_asr_noisy"]); await asyncio.sleep(gap)
    spawn(run()); return {"started": len(rows)}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept(); hub.clients.add(sock)
    try:
        while True: await sock.receive_text()
    except WebSocketDisconnect:
        hub.clients.discard(sock)


dist = ROOT / "frontend" / "dist"
if dist.exists(): app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
