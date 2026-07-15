"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

backend/vision.py

What this file is
------------------
This module contains the same core computer vision logic students
build in `starter/hand_tracking.py` and finish in `completed/`, but
restructured so a web server (`main.py`) can run it continuously in
the background and serve the latest result over HTTP, instead of
displaying it in an OpenCV window.

The `HandTracker` class owns the webcam and the MediaPipe
HandLandmarker, runs detection in a background thread, and exposes
the latest result through a small, thread-safe `get_state()` method.
FastAPI endpoints in `main.py` simply read that state — they never
touch the camera or MediaPipe directly.

Instructor note:
    This is intentionally NOT live-coded with students. It's shown
    and explained as a demo of "here's how the exact same detection
    logic you just wrote can power a real backend service." The
    checkpoint-style functions below (`create_hand_landmarker`,
    `detect_hands`, `count_fingers`) are line-for-line the same
    approach as `completed/hand_tracking.py` — only the surrounding
    application (background thread + shared state, instead of an
    OpenCV display loop) is different.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolved relative to this file's location, so the backend works
# regardless of the working directory it's launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / "hand_landmarker.task")

# Same landmark indices used throughout the workshop.
FINGER_TIPS = [4, 8, 12, 16, 20]        # Thumb, Index, Middle, Ring, Pinky tips
FINGER_PIPS = [3, 6, 10, 14, 18]        # The joint just below each tip


@dataclass
class HandState:
    """
    A snapshot of the latest hand detection result.

    Attributes:
        hand_detected: Whether a hand was visible in the most recent frame.
        finger_count: Number of raised fingers (0-5), or None if no hand
            was detected.
        fps: Smoothed frames-per-second the detection loop is running at.
        updated_at: Unix timestamp of when this snapshot was produced.
    """

    hand_detected: bool = False
    finger_count: Optional[int] = None
    fps: float = 0.0
    updated_at: float = 0.0


def create_hand_landmarker(model_path: str = MODEL_PATH) -> mp_vision.HandLandmarker:
    """
    Create and configure a MediaPipe HandLandmarker instance.

    Args:
        model_path: Path to the `hand_landmarker.task` model file.

    Returns:
        mp_vision.HandLandmarker: A ready-to-use hand landmark detector
        running in VIDEO mode.

    Raises:
        FileNotFoundError: If the model file doesn't exist at `model_path`.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Could not find the hand landmark model at:\n    {model_path}\n\n"
            "Download it into backend/ by following Step 5 of setup.md "
            "(the same model file used by starter/checkpoint-1/completed)."
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(
    landmarker: mp_vision.HandLandmarker,
    frame_bgr,
    timestamp_ms: int,
) -> mp_vision.HandLandmarkerResult:
    """
    Run hand landmark detection on a single frame.

    Args:
        landmarker: The HandLandmarker created by `create_hand_landmarker`.
        frame_bgr: A frame in OpenCV's default BGR format.
        timestamp_ms: Monotonically increasing timestamp, required by
            VIDEO running mode.

    Returns:
        mp_vision.HandLandmarkerResult: The detection result.
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def count_fingers(hand_landmarks) -> int:
    """
    Count how many fingers are raised on a single detected hand.

    Args:
        hand_landmarks: A list of 21 landmarks for ONE hand.

    Returns:
        int: The number of raised fingers (0 to 5).

    Notes:
        Identical logic to `completed/hand_tracking.py` — see that
        file's docstring for the full explanation of the tip/PIP and
        thumb-specific rules used here.
    """
    raised = 0

    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y:
            raised += 1

    thumb_tip = hand_landmarks[FINGER_TIPS[0]]
    thumb_joint = hand_landmarks[FINGER_PIPS[0]]
    wrist = hand_landmarks[0]

    if abs(thumb_tip.x - wrist.x) > abs(thumb_joint.x - wrist.x):
        raised += 1

    return raised


class HandTracker:
    """
    Runs the webcam + MediaPipe detection loop on a background thread
    and exposes the latest result in a thread-safe way.

    Usage:
        tracker = HandTracker(camera_index=0)
        tracker.start()
        ...
        state = tracker.get_state()   # safe to call from any thread
        ...
        tracker.stop()

    Instructor note:
        FastAPI can handle many HTTP requests concurrently, but only
        ONE thread should ever touch the webcam and the MediaPipe
        landmarker at a time. Running detection on a single dedicated
        background thread — and only ever reading a copied snapshot of
        the result from request handlers — avoids race conditions
        without needing a full task queue for a workshop-scale demo.
    """

    def __init__(self, camera_index: int = 0, model_path: str = MODEL_PATH) -> None:
        self._camera_index = camera_index
        self._model_path = model_path

        self._landmarker: Optional[mp_vision.HandLandmarker] = None
        self._capture: Optional[cv2.VideoCapture] = None

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._state = HandState()

    def start(self) -> None:
        """Open the webcam, initialize MediaPipe, and start the background loop."""
        if self._running:
            return

        self._landmarker = create_hand_landmarker(self._model_path)

        self._capture = cv2.VideoCapture(self._camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self._camera_index}. "
                "Check that it is connected and not in use by another application."
            )

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background loop and release the camera and model."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def get_state(self) -> HandState:
        """
        Return a thread-safe snapshot of the latest detection result.

        Returns:
            HandState: A copy of the current state — safe to read from
            any FastAPI request handler without locking concerns.
        """
        with self._lock:
            return HandState(
                hand_detected=self._state.hand_detected,
                finger_count=self._state.finger_count,
                fps=self._state.fps,
                updated_at=self._state.updated_at,
            )

    def _run_loop(self) -> None:
        """
        Background loop: continuously read frames, run detection, and
        update `self._state`. Runs until `self._running` is set to False.
        """
        start_time = time.time()
        prev_frame_time = time.time()
        fps = 0.0

        while self._running:
            success, frame = self._capture.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = detect_hands(self._landmarker, frame, timestamp_ms)

            hand_detected = bool(result.hand_landmarks)
            finger_count = count_fingers(result.hand_landmarks[0]) if hand_detected else None

            current_frame_time = time.time()
            frame_delta = current_frame_time - prev_frame_time
            if frame_delta > 0:
                instantaneous_fps = 1.0 / frame_delta
                fps = fps * 0.9 + instantaneous_fps * 0.1
            prev_frame_time = current_frame_time

            with self._lock:
                self._state = HandState(
                    hand_detected=hand_detected,
                    finger_count=finger_count,
                    fps=fps,
                    updated_at=time.time(),
                )