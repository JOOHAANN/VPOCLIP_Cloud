"""FastAPI service for the CLIPGCN action recognizer.

Run it from the CLIPGCN_Cloud directory:
    uvicorn server.main:app --host 0.0.0.0 --port 8000
Docs at http://127.0.0.1:8000/docs
"""

import logging
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from server.action_log import ActionLog
from server.llm import expand_action, write_report
from server.pipeline import ActionRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vpoclip.api")

recognizer = None
action_log = ActionLog(maxlen=500)
latencies = deque(maxlen=200)
requests_served = 0


@asynccontextmanager
async def lifespan(_app):
    # load the (heavy) model once when the server comes up
    global recognizer
    recognizer = ActionRecognizer()
    yield


app = FastAPI(
    title="VPOCLIP action recognition service",
    description=(
        "Zero-shot tri-modal human action recognition (CLIP + CTR-GCN + YOLO) "
        "for eldercare robots, with Gemini-powered describe-to-add of new classes."
    ),
    version="1.0",
    lifespan=lifespan,
)


class TopKEntry(BaseModel):
    action: str
    confidence: float


class RecognizeResponse(BaseModel):
    action: str
    confidence: float
    topk: List[TopKEntry]
    latency_ms: float


class AddActionRequest(BaseModel):
    name: str = Field(..., examples=["taking medicine"])
    casual_description: Optional[str] = Field(
        None, examples=["老人拿药瓶倒出药片放进嘴里喝水吞下"]
    )


class AddActionResponse(BaseModel):
    added: str
    prompts: List[str]
    source: str
    vocabulary_size: int


class ActionsResponse(BaseModel):
    actions: List[str]
    count: int
    unseen: List[str] = []


class ReportRequest(BaseModel):
    minutes: int = Field(10, ge=1, le=120)
    language: str = "en"


class ReportResponse(BaseModel):
    report: str
    events_analyzed: int
    window_minutes: int
    source: str


class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    device: str
    vocabulary_size: int
    requests_served: int
    avg_inference_latency_ms: float
    p95_inference_latency_ms: float


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(files: List[UploadFile] = File(..., description="JPEG frames of one short clip")):
    """Run tri-modal action recognition on the uploaded frames."""
    global requests_served
    frames = []
    for upload in files:
        data = upload.file.read()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image {upload.filename!r}")
        frames.append(frame)
    if not frames:
        raise HTTPException(status_code=400, detail="No frames uploaded")

    result = recognizer.recognize(frames)
    action_log.add(result["action"], result["confidence"])
    latencies.append(result["latency_ms"])
    requests_served += 1
    log.info("/recognize %d frames -> %s (%.2f) in %.1f ms",
             len(frames), result["action"], result["confidence"], result["latency_ms"])
    return result


@app.post("/add_action", response_model=AddActionResponse)
def add_action(request: AddActionRequest):
    """Add a new recognizable action from a casual description (uses the LLM)."""
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Action name must not be empty")
    try:
        canonical, prompts, source = expand_action(request.name, request.casual_description)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    recognizer.add_class(canonical, prompts)
    log.info("/add_action %r -> %r via %s (%d prompts)", request.name, canonical, source, len(prompts))
    return AddActionResponse(
        added=canonical,
        prompts=prompts,
        source=source,
        vocabulary_size=recognizer.vocabulary_size,
    )


@app.get("/actions", response_model=ActionsResponse)
def list_actions():
    """The current action vocabulary, plus which classes are unseen."""
    actions = recognizer.list_classes()
    return ActionsResponse(actions=actions, count=len(actions), unseen=recognizer.unseen_classes())


@app.delete("/actions/{name}", response_model=ActionsResponse)
def delete_action(name: str):
    """Remove an action class by name."""
    if not recognizer.remove_class(name):
        raise HTTPException(status_code=404, detail=f"Unknown action {name!r}")
    log.info("/actions deleted %r", name)
    actions = recognizer.list_classes()
    return ActionsResponse(actions=actions, count=len(actions), unseen=recognizer.unseen_classes())


@app.post("/report", response_model=ReportResponse)
def report(request: ReportRequest = ReportRequest()):
    """Summarize the recent action log for a caregiver (uses the LLM)."""
    events = action_log.recent(request.minutes)
    try:
        text, source = write_report(events, request.minutes, request.language)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    log.info("/report %d events over %d min via %s", len(events), request.minutes, source)
    return ReportResponse(
        report=text,
        events_analyzed=len(events),
        window_minutes=request.minutes,
        source=source,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Status plus inference latency stats."""
    recent = sorted(latencies)
    avg = sum(recent) / len(recent) if recent else 0.0
    p95 = recent[int(0.95 * (len(recent) - 1))] if recent else 0.0
    return HealthResponse(
        status="ok",
        mock_mode=recognizer.mock_mode,
        device=recognizer.device_name,
        vocabulary_size=recognizer.vocabulary_size,
        requests_served=requests_served,
        avg_inference_latency_ms=round(avg, 1),
        p95_inference_latency_ms=round(p95, 1),
    )
