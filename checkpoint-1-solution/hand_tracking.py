"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

CHECKPOINT 1 REFERENCE: hand_tracking.py

What this file is
------------------
This is a reference checkpoint for students who fall behind during the
practical. Checkpoints 1 and 2 (initializing MediaPipe and running
inference on each frame) are already solved here. If you're stuck on
Checkpoint 1 or 2 in `starter/hand_tracking.py`, compare your code
against this file — then continue solving Checkpoints 3 and 4 yourself
(they are still marked TODO below, exactly as in the starter).

Do not simply copy this file over your starter script and skip ahead —
you'll miss the learning value of Checkpoints 1 and 2. Use it as a
reference, not a shortcut.

Checkpoints
-----------
    CHECKPOINT 1 — Initialize the MediaPipe HandLandmarker      [SOLVED]
    CHECKPOINT 2 — Convert the OpenCV frame and run inference   [SOLVED]
    CHECKPOINT 3 — Draw the detected hand landmarks             [TODO]
    CHECKPOINT 4 — Count raised fingers from landmark positions [TODO]
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
# `python checkpoint-1/hand_tracking.py` from the repository root.
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / "hand_landmarker.task")

# Landmark indices for fingertips and their corresponding lower joints.
# These come from MediaPipe's 21-point hand landmark model.
# Reference diagram: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
FINGER_TIPS = [4, 8, 12, 16, 20]        # Thumb, Index, Middle, Ring, Pinky tips
FINGER_PIPS = [3, 6, 10, 14, 18]        # The joint just below each tip

WINDOW_NAME = "AIDL Forge — Hand Tracking (Checkpoint 1)"


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

    Instructor note:
        VIDEO mode requires a monotonically increasing timestamp per
        frame, which we already provide in the main loop below via
        `frame_timestamp_ms`. This is different from IMAGE mode, which
        treats every frame as an independent, unrelated picture.
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

    Instructor note:
        MediaPipe expects RGB images, but OpenCV captures frames in BGR.
        Forgetting this conversion is one of the most common beginner
        mistakes in this workshop — colors will look "detected" but
        landmark accuracy will be poor.
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


# ---------------------------------------------------------------------------
# CHECKPOINT 3 — Draw the detected hand landmarks
# ---------------------------------------------------------------------------
def draw_landmarks(frame_bgr, result: mp_vision.HandLandmarkerResult):
    """
    Draw the detected hand landmarks and connections onto the frame.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        result: The HandLandmarkerResult returned by `detect_hands`.

    Returns:
        The same frame, with landmarks drawn on top (if any hand was
        detected).

    Instructor note:
        `result.hand_landmarks` is a list of hands. Each hand is a list
        of 21 landmarks, and each landmark has `.x` and `.y` in
        NORMALIZED coordinates (0.0 to 1.0) relative to image width and
        height. You must multiply by the frame's width/height to get
        pixel coordinates before drawing.
    """
    if not result.hand_landmarks:
        return frame_bgr

    frame_height, frame_width = frame_bgr.shape[:2]

    for hand_landmarks in result.hand_landmarks:
        # TODO (Checkpoint 3):
        # 1. Loop over `hand_landmarks` (21 landmark points).
        # 2. Convert each landmark's normalized (x, y) to pixel
        #    coordinates: px = int(landmark.x * frame_width)
        #                  py = int(landmark.y * frame_height)
        # 3. Draw a small filled circle at each point using
        #    cv2.circle(frame_bgr, (px, py), 5, (0, 255, 0), -1)
        #
        # Bonus (optional): connect the dots using HAND_CONNECTIONS
        # pairs to draw the hand "skeleton" instead of just points.
        pass

    return frame_bgr


# ---------------------------------------------------------------------------
# CHECKPOINT 4 — Count raised fingers
# ---------------------------------------------------------------------------
def count_fingers(hand_landmarks) -> int:
    """
    Count how many fingers are raised on a single detected hand.

    Args:
        hand_landmarks: A list of 21 landmarks for ONE hand, as returned
            in `result.hand_landmarks[i]`.

    Returns:
        int: The number of raised fingers (0 to 5).

    Instructor note:
        The simplest reliable rule for this workshop:
        A finger counts as "raised" if its TIP landmark is above
        (smaller y value) its PIP joint landmark, because image
        y-coordinates increase downward.

        The thumb is a special case because it moves sideways rather
        than up/down — we handle it separately using x-coordinates.
        A simple approach: compare the thumb tip's x position to the
        thumb's lower joint (landmark 3), relative to the wrist (0),
        to infer whether the thumb is extended outward.
    """
    # TODO (Checkpoint 4):
    # 1. Initialize a counter, e.g. `raised = 0`.
    # 2. For the FOUR fingers (index, middle, ring, pinky — skip thumb
    #    for now), compare landmark[tip].y < landmark[pip].y for each
    #    pair in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]). If true, the
    #    finger is raised — increment `raised`.
    # 3. Handle the thumb separately (landmark 4 vs landmark 3),
    #    comparing x-coordinates instead of y-coordinates.
    # 4. Return the total count.
    #
    # Replace the line below with your implementation.
    raise NotImplementedError("Checkpoint 4: count_fingers() is not implemented yet.")


# ---------------------------------------------------------------------------
# STEP 5 — Status overlay helper (provided — do not modify)
# ---------------------------------------------------------------------------
def draw_status_lines(frame_bgr, lines: list) -> None:
    """
    Draw a stack of status text lines in the top-left corner of the frame.

    Args:
        frame_bgr: The frame to draw on (modified in place).
        lines: A list of strings, one per line, drawn top to bottom.
    """
    for i, line in enumerate(lines):
        cv2.putText(
            frame_bgr,
            line,
            (20, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# STEP 6 — Main application loop (provided — do not modify)
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Open the webcam and run the detection pipeline on every frame,
    displaying live results in an OpenCV window until the user
    presses 'q'.

    Checkpoints 1 and 2 are already solved in this file, so detection
    always runs. Checkpoints 3 and 4 are attempted independently: if
    one isn't implemented yet, the window stays open and shows what
    IS working, plus a status message — instead of crashing the whole
    script.
    """
    print("Starting AIDL Forge hand tracking (checkpoint-1)...")
    print("Press 'q' in the video window to quit.\n")

    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check that it is connected and "
            "not being used by another application."
        )

    start_time = time.time()

    # Track which checkpoints we've already printed a console message
    # for, so we warn once instead of spamming the terminal every frame.
    warned_checkpoints = set()

    def warn_once(checkpoint: int, error: Exception) -> None:
        if checkpoint not in warned_checkpoints:
            print(f"[Checkpoint {checkpoint} pending] {error}")
            warned_checkpoints.add(checkpoint)

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

            status_lines = []
            finger_count: Optional[int] = None

            # Checkpoint 3: attempt to draw landmarks, but keep going
            # even if it isn't implemented yet.
            try:
                frame = draw_landmarks(frame, result)
            except NotImplementedError as error:
                warn_once(3, error)
                status_lines.append("Checkpoint 3 pending: landmarks not drawn")

            # Checkpoint 4: attempt to count fingers, but keep going
            # even if it isn't implemented yet.
            if result.hand_landmarks:
                try:
                    finger_count = count_fingers(result.hand_landmarks[0])
                except NotImplementedError as error:
                    warn_once(4, error)
                    status_lines.append("Checkpoint 4 pending: count not shown")

            if finger_count is not None:
                status_lines.append(f"Fingers: {finger_count}")
            elif not result.hand_landmarks:
                status_lines.append("No hand detected")

            draw_status_lines(frame, status_lines)

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