"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

BONUS: pose-estimation.py

What this project does
-----------------------
Detects a full body pose in real time using MediaPipe's Tasks API
PoseLandmarker, drawing all 33 body landmarks and the connecting
skeleton lines (shoulders, arms, hips, legs).

Same overall pattern as the hand-tracking practical and the other
bonus projects: create a task, convert frames to RGB, call
detect_for_video(), draw the results — just with a different model
and a larger landmark set (33 points across the whole body instead of
21 points on one hand).

Model file required:
    pose_landmarker_lite.task
    Download (same pattern as hand_landmarker.task in setup.md):

    curl -o bonus/pose_landmarker_lite.task \\
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

    The "lite" variant is used here for speed on typical laptops;
    "full" and "heavy" variants trade speed for accuracy if you want
    to experiment.

Controls:
    - Press 'q' to quit.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / "pose_landmarker_lite.task")

WINDOW_NAME = "AIDL Forge — Bonus: Pose Estimation"

COLOR_LANDMARK = (0, 220, 0)
COLOR_CONNECTION = (255, 180, 0)
COLOR_TEXT = (255, 255, 255)

# Core body connections for the 33-point MediaPipe Pose landmark model.
# Reference diagram: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS = [
    # Shoulders and torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Face (simple outline: eyes, ears, mouth corners, nose)
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
def create_pose_landmarker() -> mp_vision.PoseLandmarker:
    """
    Create a MediaPipe PoseLandmarker configured for live video.

    Returns:
        mp_vision.PoseLandmarker: A ready-to-use pose landmark detector
        running in VIDEO mode.
    """
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Could not find the pose landmark model at:\n    {MODEL_PATH}\n\n"
            "Download it with:\n"
            "  curl -o bonus/pose_landmarker_lite.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def detect_pose(landmarker, frame_bgr, timestamp_ms: int) -> mp_vision.PoseLandmarkerResult:
    """Run pose detection on a single frame."""
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_pose(frame_bgr, result: mp_vision.PoseLandmarkerResult):
    """
    Draw the detected pose's landmarks and skeleton connections.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        result: The PoseLandmarkerResult returned by `detect_pose`.

    Returns:
        The frame with the pose skeleton drawn on top, if a pose was detected.
    """
    if not result.pose_landmarks:
        return frame_bgr

    frame_height, frame_width = frame_bgr.shape[:2]

    for pose_landmarks in result.pose_landmarks:
        points = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in pose_landmarks
        ]

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame_bgr, points[start_idx], points[end_idx], COLOR_CONNECTION, 2)

        for point in points:
            cv2.circle(frame_bgr, point, 4, COLOR_LANDMARK, -1)

    return frame_bgr


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
def main() -> None:
    print("Starting AIDL Forge bonus project: Pose Estimation")
    print("Step back so your full body is visible in frame.")
    print("Press 'q' to quit.\n")

    landmarker = create_pose_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    start_time = time.time()
    prev_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            timestamp_ms = int((time.time() - start_time) * 1000)
            result = detect_pose(landmarker, frame, timestamp_ms)
            frame = draw_pose(frame, result)

            current_frame_time = time.time()
            frame_delta = current_frame_time - prev_frame_time
            if frame_delta > 0:
                fps = fps * 0.9 + (1.0 / frame_delta) * 0.1
            prev_frame_time = current_frame_time

            status = "Pose detected" if result.pose_landmarks else "No pose detected"
            cv2.putText(
                frame,
                f"{status}  FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nWebcam released. Goodbye!")


if __name__ == "__main__":
    main()