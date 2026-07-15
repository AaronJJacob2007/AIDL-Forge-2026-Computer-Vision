"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

backend/main.py

What this file is
------------------
The FastAPI application that exposes the hand-tracking pipeline
(`vision.py`) as a small HTTP API. A browser-based frontend
(`frontend/`) polls this API to display a live finger count.

Instructor note:
    This file is demonstrated, not live-coded with students. The
    goal is for students to see how the exact detection logic they
    just wrote in `completed/hand_tracking.py` — webcam frame in,
    finger count out — can sit behind a web API instead of an
    OpenCV window, powering any number of different frontends.

Endpoints:
    GET /        — basic service info, useful as a health check
    GET /count   — the latest hand detection result, as JSON

Running:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import get_settings
from vision import HandTracker

settings = get_settings()
tracker = HandTracker(camera_index=settings.camera_index)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the webcam + MediaPipe background thread when the server
    starts, and cleanly release the camera when it shuts down.

    FastAPI's lifespan context manager is the modern replacement for
    the older `@app.on_event("startup")` / `@app.on_event("shutdown")`
    decorators — everything before `yield` runs on startup, and
    everything after `yield` runs on shutdown.
    """
    tracker.start()
    yield
    tracker.stop()


app = FastAPI(
    title="AIDL Forge — Hand Tracking API",
    description="Exposes live finger-counting results from a webcam hand-tracking pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend (served from a different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


class RootResponse(BaseModel):
    """Response shape for the root health-check endpoint."""

    status: str
    message: str


class CountResponse(BaseModel):
    """Response shape for the /count endpoint."""

    hand_detected: bool
    finger_count: Optional[int]
    fps: float
    updated_at: float


@app.get("/", response_model=RootResponse)
async def read_root() -> RootResponse:
    """
    Basic service info endpoint.

    Returns:
        RootResponse: A simple status payload confirming the API is
        running, useful for a quick health check.
    """
    return RootResponse(
        status="ok",
        message="AIDL Forge Hand Tracking API is running. See GET /count for live results.",
    )


@app.get("/count", response_model=CountResponse)
async def read_count() -> CountResponse:
    """
    Return the most recent hand detection result.

    Returns:
        CountResponse: Whether a hand is currently detected, the
        raised finger count (or null if no hand is detected), the
        current detection loop FPS, and when this snapshot was taken.
    """
    state = tracker.get_state()
    return CountResponse(
        hand_detected=state.hand_detected,
        finger_count=state.finger_count,
        fps=round(state.fps, 1),
        updated_at=state.updated_at,
    )