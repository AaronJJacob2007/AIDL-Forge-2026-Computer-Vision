"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

BONUS: face-detection.py

What this project does
-----------------------
Detects faces in real time using MediaPipe's Tasks API FaceDetector,
drawing a bounding box and six facial keypoints (eyes, ears, nose tip,
mouth center) around each detected face, along with a confidence score.

This uses a DIFFERENT MediaPipe model than the main practical
(BlazeFace, for face detection, instead of the hand landmark model),
but follows the exact same overall pattern: create a task, convert
frames to RGB, call detect_for_video(), draw the results.

Model file required:
    blaze_face_short_range.task
    Download (same pattern as hand_landmarker.task in setup.md):

    curl -o bonus/blaze_face_short_range.task \\
      https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task

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
MODEL_PATH = str(SCRIPT_DIR / "blaze_face_short_range.task")

WINDOW_NAME = "AIDL Forge — Bonus: Face Detection"

COLOR_BOX = (0, 220, 0)
COLOR_KEYPOINT = (0, 140, 255)
COLOR_TEXT = (255, 255, 255)


# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
def create_face_detector() -> mp_vision.FaceDetector:
    """
    Create a MediaPipe FaceDetector configured for live video.

    Returns:
        mp_vision.FaceDetector: A ready-to-use face detector running
        in VIDEO mode.
    """
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Could not find the face detection model at:\n    {MODEL_PATH}\n\n"
            "Download it with:\n"
            "  curl -o bonus/blaze_face_short_range.task "
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.task"
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def detect_faces(detector, frame_bgr, timestamp_ms: int) -> mp_vision.FaceDetectorResult:
    """Run face detection on a single frame."""
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect_for_video(mp_image, timestamp_ms)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(frame_bgr, result: mp_vision.FaceDetectorResult):
    """
    Draw bounding boxes, keypoints, and confidence scores for each
    detected face.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        result: The FaceDetectorResult returned by `detect_faces`.

    Returns:
        The frame with detections drawn on top.
    """
    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), COLOR_BOX, 2)

        # Each detection has one category (face) with a confidence score.
        if detection.categories:
            score = detection.categories[0].score
            label = f"Face {score * 100:.0f}%"
            cv2.putText(
                frame_bgr,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_BOX,
                2,
                cv2.LINE_AA,
            )

        # Keypoints are normalized (0.0-1.0); convert to pixel coordinates.
        frame_height, frame_width = frame_bgr.shape[:2]
        for keypoint in detection.keypoints:
            kp_x = int(keypoint.x * frame_width)
            kp_y = int(keypoint.y * frame_height)
            cv2.circle(frame_bgr, (kp_x, kp_y), 3, COLOR_KEYPOINT, -1)

    return frame_bgr


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
def main() -> None:
    print("Starting AIDL Forge bonus project: Face Detection")
    print("Press 'q' to quit.\n")

    detector = create_face_detector()

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
            result = detect_faces(detector, frame, timestamp_ms)
            frame = draw_detections(frame, result)

            current_frame_time = time.time()
            frame_delta = current_frame_time - prev_frame_time
            if frame_delta > 0:
                fps = fps * 0.9 + (1.0 / frame_delta) * 0.1
            prev_frame_time = current_frame_time

            cv2.putText(
                frame,
                f"Faces: {len(result.detections)}  FPS: {fps:.1f}",
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
        detector.close()
        print("\nWebcam released. Goodbye!")


if __name__ == "__main__":
    main()