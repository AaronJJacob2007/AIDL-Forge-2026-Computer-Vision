"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

COMPLETED: hand_tracking.py

What this file is
------------------
This is the fully working reference solution for the workshop
practical. All four checkpoints are implemented, plus a few small
extras that make the demo feel polished:

    - A live FPS (frames per second) counter
    - The full hand "skeleton" (landmarks + connecting lines), not
      just isolated dots
    - A clean on-screen UI panel instead of raw floating text

Use this file to check your own implementation, or to see the
"finished product" the starter script is building toward. If you're
still working through the checkpoints yourself, try not to peek at
the solved logic below until you've given each checkpoint an honest
attempt in `starter/hand_tracking.py`.

Checkpoints (all solved)
-------------------------
    CHECKPOINT 1 — Initialize the MediaPipe HandLandmarker
    CHECKPOINT 2 — Convert the OpenCV frame and run inference
    CHECKPOINT 3 — Draw the detected hand landmarks
    CHECKPOINT 4 — Count raised fingers from landmark positions
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# STEP 0 — Configuration
# ---------------------------------------------------------------------------
# Path to the pre-trained hand landmark model.
# Downloaded once during setup — see setup.md for the download command.
#
# Resolved relative to THIS FILE's location (not the current working
# directory) so the script works correctly no matter where it's run
# from — e.g. `python hand_tracking.py` from inside this folder, or
# `python completed/hand_tracking.py` from the repository root.
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / "hand_landmarker.task")

# Landmark indices for fingertips and their corresponding lower joints.
# These come from MediaPipe's 21-point hand landmark model.
# Reference diagram: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
FINGER_TIPS = [4, 8, 12, 16, 20]        # Thumb, Index, Middle, Ring, Pinky tips
FINGER_PIPS = [3, 6, 10, 14, 18]        # The joint just below each tip

# Pairs of landmark indices to connect with lines, forming the visible
# "skeleton" of the hand. Defined locally so this file has no
# dependency on the deprecated mediapipe.solutions module.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky finger
    (0, 17),                                 # Palm base
]

WINDOW_NAME = "AIDL Forge — Hand Tracking (Completed)"

# UI colors (BGR, since OpenCV uses Blue-Green-Red ordering)
COLOR_LANDMARK = (0, 255, 0)      # Green dots
COLOR_CONNECTION = (255, 200, 0)  # Light blue skeleton lines
COLOR_TEXT = (255, 255, 255)      # White text
COLOR_PANEL_BG = (40, 40, 40)     # Dark gray UI panel background


# ---------------------------------------------------------------------------
# CHECKPOINT 1 — Initialize the MediaPipe HandLandmarker [SOLVED]
# ---------------------------------------------------------------------------
def create_hand_landmarker() -> mp_vision.HandLandmarker:
    """
    Create and configure a MediaPipe HandLandmarker instance.

    Returns:
        mp_vision.HandLandmarker: A ready-to-use hand landmark detector
        running in VIDEO mode (suitable for processing a live sequence
        of webcam frames).
    """
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Could not find the hand landmark model at:\n    {MODEL_PATH}\n\n"
            "Download it by following Step 5 of setup.md, and make sure "
            "hand_landmarker.task ends up in this same folder as "
            "hand_tracking.py."
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp_vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# CHECKPOINT 2 — Convert the frame and run inference [SOLVED]
# ---------------------------------------------------------------------------
def detect_hands(
    landmarker: mp_vision.HandLandmarker,
    frame_bgr,
    timestamp_ms: int,
) -> mp_vision.HandLandmarkerResult:
    """
    Run hand landmark detection on a single webcam frame.

    Args:
        landmarker: The HandLandmarker created in Checkpoint 1.
        frame_bgr: The raw webcam frame, in OpenCV's default BGR format.
        timestamp_ms: Monotonically increasing timestamp for this frame,
            required by VIDEO running mode.

    Returns:
        mp_vision.HandLandmarkerResult: The detection result, containing
        zero or more detected hands and their 21 landmarks each.
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


# ---------------------------------------------------------------------------
# CHECKPOINT 3 — Draw the detected hand landmarks [SOLVED]
# ---------------------------------------------------------------------------
def draw_landmarks(frame_bgr, result: mp_vision.HandLandmarkerResult):
    """
    Draw the detected hand landmarks and connecting lines onto the frame.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        result: The HandLandmarkerResult returned by `detect_hands`.

    Returns:
        The same frame, with a full hand skeleton drawn on top (if any
        hand was detected).
    """
    if not result.hand_landmarks:
        return frame_bgr

    frame_height, frame_width = frame_bgr.shape[:2]

    for hand_landmarks in result.hand_landmarks:
        # Convert every landmark from normalized (0.0-1.0) coordinates
        # to actual pixel coordinates on this frame.
        points = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks
        ]

        # Draw the connecting lines first, so the landmark dots sit
        # visually on top of them.
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start_idx], points[end_idx], COLOR_CONNECTION, 2)

        # Draw each landmark as a filled circle.
        for point in points:
            cv2.circle(frame_bgr, point, 5, COLOR_LANDMARK, -1)

    return frame_bgr


# ---------------------------------------------------------------------------
# CHECKPOINT 4 — Count raised fingers [SOLVED]
# ---------------------------------------------------------------------------
def count_fingers(hand_landmarks) -> int:
    """
    Count how many fingers are raised on a single detected hand.

    Args:
        hand_landmarks: A list of 21 landmarks for ONE hand, as returned
            in `result.hand_landmarks[i]`.

    Returns:
        int: The number of raised fingers (0 to 5).

    Notes:
        - Index/Middle/Ring/Pinky: a finger is "raised" if its tip
          landmark has a smaller y-value (higher on screen) than its
          PIP joint, since image y-coordinates increase downward.
        - Thumb: handled separately by comparing the tip's x-position
          against the joint below it (landmark 3), since the thumb
          moves mostly sideways rather than up/down. This simple rule
          works well for a hand held upright facing the camera, which
          is the expected use case for this workshop.
    """
    raised = 0

    # Four fingers: index, middle, ring, pinky (skip index 0, the thumb pair)
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y:
            raised += 1

    # Thumb: compare x-coordinates instead of y-coordinates.
    thumb_tip = hand_landmarks[FINGER_TIPS[0]]
    thumb_joint = hand_landmarks[FINGER_PIPS[0]]
    wrist = hand_landmarks[0]

    # If the thumb tip is further from the wrist (in x) than the joint
    # below it, treat the thumb as extended outward.
    if abs(thumb_tip.x - wrist.x) > abs(thumb_joint.x - wrist.x):
        raised += 1

    return raised


# ---------------------------------------------------------------------------
# STEP 5 — UI helpers (provided)
# ---------------------------------------------------------------------------
def draw_ui_panel(frame_bgr, finger_count: Optional[int], fps: float):
    """
    Draw a clean status panel in the top-left corner of the frame,
    showing the current finger count and FPS.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        finger_count: The number of raised fingers, or None if no hand
            is currently detected.
        fps: The current, smoothed frames-per-second value.

    Returns:
        The frame with the UI panel drawn on top.
    """
    panel_width, panel_height = 260, 90

    # Semi-transparent dark background panel for readability.
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    finger_label = f"Fingers: {finger_count}" if finger_count is not None else "Fingers: --"
    status_label = "Hand detected" if finger_count is not None else "No hand detected"

    cv2.putText(frame_bgr, finger_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, status_label, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame_bgr


# ---------------------------------------------------------------------------
# STEP 6 — Main application loop
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Open the webcam, run the full detection pipeline on every frame,
    and display the live results in an OpenCV window until the user
    presses 'q'.
    """
    print("Starting AIDL Forge hand tracking (completed)...")
    print("Press 'q' in the video window to quit.\n")

    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check that it is connected and "
            "not being used by another application."
        )

    start_time = time.time()
    prev_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: failed to read frame from webcam. Skipping.")
                continue

            # Flip horizontally for a natural "mirror" selfie view.
            frame = cv2.flip(frame, 1)

            timestamp_ms = int((time.time() - start_time) * 1000)

            result = detect_hands(landmarker, frame, timestamp_ms)
            frame = draw_landmarks(frame, result)

            finger_count: Optional[int] = None
            if result.hand_landmarks:
                finger_count = count_fingers(result.hand_landmarks[0])

            # Smooth FPS calculation using the time delta between frames.
            current_frame_time = time.time()
            frame_delta = current_frame_time - prev_frame_time
            if frame_delta > 0:
                instantaneous_fps = 1.0 / frame_delta
                # Simple exponential smoothing so the number doesn't jitter.
                fps = fps * 0.9 + instantaneous_fps * 0.1
            prev_frame_time = current_frame_time

            frame = draw_ui_panel(frame, finger_count, fps)

            cv2.imshow(WINDOW_NAME, frame)

            # Press 'q' to exit the application.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Always release the camera and destroy windows, even if an
        # error occurs above. This prevents the webcam from staying
        # "locked" after the script exits.
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nWebcam released. Goodbye!")


if __name__ == "__main__":
    main()