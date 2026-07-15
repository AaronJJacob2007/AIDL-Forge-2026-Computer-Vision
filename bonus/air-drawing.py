"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

BONUS: air-drawing.py

What this project does
-----------------------
Turns your index fingertip into a paintbrush. Raise only your index
finger to draw in the air; raise your index AND middle fingers
together to enter "selection mode" and pick a color (or clear the
canvas) from the palette at the top of the screen, without drawing.

This file is a complete, standalone project — unlike the workshop
checkpoints, there's nothing to fill in here. Read through it to see
how the same hand-tracking building blocks from the main practical
(HandLandmarker, landmark positions, finger-up detection) can be
combined into a different kind of application.

Controls:
    - Raise ONLY your index finger  -> draw
    - Raise index + middle fingers  -> selection mode (touch the
      palette at the top to change color or clear the canvas)
    - Press 'c'                     -> clear the canvas
    - Press 'q'                     -> quit
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / "hand_landmarker.task")

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

WINDOW_NAME = "AIDL Forge — Bonus: Air Drawing"

BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40

# Palette: (label, BGR color). "CLEAR" is a special action, not a color.
PALETTE = [
    ("RED", (0, 0, 255)),
    ("GREEN", (0, 200, 0)),
    ("BLUE", (255, 100, 0)),
    ("YELLOW", (0, 220, 220)),
    ("CLEAR", (60, 60, 60)),
]
PALETTE_SWATCH_WIDTH = 120
PALETTE_HEIGHT = 60


# ---------------------------------------------------------------------------
# MediaPipe setup (same pattern as the main workshop practical)
# ---------------------------------------------------------------------------
def create_hand_landmarker() -> mp_vision.HandLandmarker:
    """Create a MediaPipe HandLandmarker configured for live video."""
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Could not find the hand landmark model at:\n    {MODEL_PATH}\n\n"
            "Download it into bonus/ by following Step 5 of setup.md."
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame_bgr, timestamp_ms: int) -> mp_vision.HandLandmarkerResult:
    """Run detection on a single frame. Same approach as the main practical."""
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def fingers_up(hand_landmarks) -> List[bool]:
    """
    Determine which of the 5 fingers are raised.

    Returns:
        List[bool]: [thumb, index, middle, ring, pinky], True if raised.

    This is the same tip-vs-PIP comparison used in the main practical's
    `count_fingers()`, but returning the per-finger breakdown instead
    of just a total — useful here because we care WHICH fingers are up,
    not just how many.
    """
    states = []

    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        states.append(hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y)

    thumb_tip = hand_landmarks[FINGER_TIPS[0]]
    thumb_joint = hand_landmarks[FINGER_PIPS[0]]
    wrist = hand_landmarks[0]
    thumb_up = abs(thumb_tip.x - wrist.x) > abs(thumb_joint.x - wrist.x)

    return [thumb_up] + states


# ---------------------------------------------------------------------------
# Palette UI
# ---------------------------------------------------------------------------
def draw_palette(frame_bgr, selected_label: str) -> None:
    """
    Draw the color/clear palette bar across the top of the frame.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        selected_label: Label of the currently selected palette entry,
            highlighted with a white border.
    """
    for i, (label, color) in enumerate(PALETTE):
        x_start = i * PALETTE_SWATCH_WIDTH
        x_end = x_start + PALETTE_SWATCH_WIDTH

        cv2.rectangle(frame_bgr, (x_start, 0), (x_end, PALETTE_HEIGHT), color, -1)

        if label == selected_label:
            cv2.rectangle(frame_bgr, (x_start, 0), (x_end, PALETTE_HEIGHT), (255, 255, 255), 3)

        text_color = (255, 255, 255) if label != "YELLOW" else (30, 30, 30)
        cv2.putText(
            frame_bgr,
            label,
            (x_start + 10, PALETTE_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
            cv2.LINE_AA,
        )


def palette_label_at(x: int, y: int) -> Optional[str]:
    """
    Return the palette label under a given pixel position, or None if
    the position isn't over the palette bar at all.
    """
    if y > PALETTE_HEIGHT:
        return None

    index = x // PALETTE_SWATCH_WIDTH
    if 0 <= index < len(PALETTE):
        return PALETTE[index][0]
    return None


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
def main() -> None:
    print("Starting AIDL Forge bonus project: Air Drawing")
    print("Raise ONLY your index finger to draw.")
    print("Raise index + middle fingers together to select a color from the palette.")
    print("Press 'c' to clear, 'q' to quit.\n")

    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # The canvas accumulates strokes across frames — it persists even
    # though each webcam frame is thrown away after being drawn on.
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    current_color = PALETTE[0][1]
    current_label = PALETTE[0][0]
    previous_point: Optional[Tuple[int, int]] = None

    start_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            # Resize the canvas defensively in case the webcam's actual
            # resolution doesn't match what we assumed above.
            if canvas.shape[:2] != frame.shape[:2]:
                canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))

            timestamp_ms = int((time.time() - start_time) * 1000)
            result = detect_hands(landmarker, frame, timestamp_ms)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                thumb_up, index_up, middle_up, ring_up, pinky_up = fingers_up(hand_landmarks)

                # Index fingertip position in pixel coordinates — our brush tip.
                tip = hand_landmarks[FINGER_TIPS[1]]
                tip_x = int(tip.x * frame.shape[1])
                tip_y = int(tip.y * frame.shape[0])

                if index_up and middle_up and not ring_up and not pinky_up:
                    # Selection mode: don't draw, just check if we're
                    # hovering over a palette entry.
                    previous_point = None
                    cv2.circle(frame, (tip_x, tip_y), 12, (255, 255, 255), 2)

                    label = palette_label_at(tip_x, tip_y)
                    if label == "CLEAR":
                        canvas[:] = 0
                    elif label is not None:
                        current_label = label
                        current_color = next(color for name, color in PALETTE if name == label)

                elif index_up and not middle_up and not ring_up and not pinky_up:
                    # Draw mode: connect the previous fingertip position
                    # to the current one with a line, building up strokes
                    # on the persistent canvas.
                    if previous_point is not None and tip_y > PALETTE_HEIGHT:
                        thickness = ERASER_THICKNESS if current_label == "CLEAR" else BRUSH_THICKNESS
                        cv2.line(canvas, previous_point, (tip_x, tip_y), current_color, thickness)

                    previous_point = (tip_x, tip_y) if tip_y > PALETTE_HEIGHT else None

                else:
                    # Any other hand shape (fist, all fingers up, etc.)
                    # pauses drawing without erasing anything.
                    previous_point = None
            else:
                previous_point = None

            # Merge the accumulated canvas onto the live frame using a
            # mask, so drawn strokes appear opaque and clean rather than
            # semi-transparent (as a simple alpha blend would produce).
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, inverse_mask = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
            inverse_mask_bgr = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)

            frame = cv2.bitwise_and(frame, inverse_mask_bgr)
            frame = cv2.bitwise_or(frame, canvas)

            draw_palette(frame, current_label)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                canvas[:] = 0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nWebcam released. Goodbye!")


if __name__ == "__main__":
    main()