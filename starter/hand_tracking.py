"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

STARTER: hand_tracking.py

Goal of this file
------------------
By the end of this script, your webcam feed will:

    1. Detect a hand in the frame
    2. Draw the 21 hand landmarks on screen
    3. Count how many fingers are raised
    4. Show the result live in an OpenCV window

You will complete FOUR checkpoints, each marked with a "TODO" comment.
Everything else (webcam setup, window management, cleanup, drawing
helpers) is already written for you — do not modify code outside the
TODO blocks unless your instructor tells you to.

Checkpoints
-----------
    CHECKPOINT 1 — Initialize the MediaPipe HandLandmarker
    CHECKPOINT 2 — Convert the OpenCV frame and run inference
    CHECKPOINT 3 — Draw the detected hand landmarks
    CHECKPOINT 4 — Count raised fingers from landmark positions

Instructor note
----------------
This starter uses the modern MediaPipe **Tasks API**
(`mediapipe.tasks.python.vision.HandLandmarker`), which is the
actively maintained way to run hand detection in MediaPipe. It
requires a small pre-trained model file, `hand_landmarker.task`,
which each student downloads once during setup (see `setup.md`).
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
# `python starter/hand_tracking.py` from the repository root.
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
# You will use this list in Checkpoint 3 to draw the connecting lines
# between landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky finger
    (0, 17),                                 # Palm base
]

WINDOW_NAME = "AIDL Forge — Hand Tracking (Starter)"

# UI colors (BGR, since OpenCV uses Blue-Green-Red ordering).
# You will use these in Checkpoint 3 when drawing landmarks and
# connecting lines.
COLOR_LANDMARK = (0, 255, 0)      # Green dots
COLOR_CONNECTION = (255, 200, 0)  # Light blue skeleton lines


# ---------------------------------------------------------------------------
# CHECKPOINT 1 — Initialize the MediaPipe HandLandmarker
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
    # Provided for you: a friendly check so a missing model file gives a
    # clear error instead of a confusing low-level MediaPipe crash.
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Could not find the hand landmark model at:\n    {MODEL_PATH}\n\n"
            "Download it by following Step 5 of setup.md, and make sure "
            "hand_landmarker.task ends up in this same folder as "
            "hand_tracking.py."
        )

    # TODO (Checkpoint 1):
    # 1. Create a `BaseOptions` object pointing at MODEL_PATH.
    #    Hint: mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    #
    # 2. Create a `HandLandmarkerOptions` object using:
    #       - base_options   -> the BaseOptions object from step 1
    #       - running_mode   -> mp_vision.RunningMode.VIDEO
    #       - num_hands      -> 1 (keep it simple for this workshop)
    #       - min_hand_detection_confidence -> 0.5
    #       - min_tracking_confidence       -> 0.5
    #
    # 3. Build and return the landmarker with:
    #       mp_vision.HandLandmarker.create_from_options(options)
    #
    # Replace the line below with your implementation.
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options=mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,)
    return mp_vision.HandLandmarker.create_from_options(options)
    


# ---------------------------------------------------------------------------
# CHECKPOINT 2 — Convert the frame and run inference
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
    # TODO (Checkpoint 2):
    # 1. Convert `frame_bgr` from BGR to RGB using cv2.cvtColor.
    # 2. Wrap the RGB frame in a mp.Image object:
    #       mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # 3. Call `landmarker.detect_for_video(mp_image, timestamp_ms)`
    #    and return the result.
    #
    # Replace the line below with your implementation.
    rgb_frame=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
    return landmarker.detect_for_video(mp_image,timestamp_ms)



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
        # 1. Loop over `hand_landmarks` (21 landmark points) and convert
        #    each landmark's normalized (x, y) to pixel coordinates:
        #       px = int(landmark.x * frame_width)
        #       py = int(landmark.y * frame_height)
        #    Store the results as a list of (px, py) tuples — you'll
        #    need it for both steps below.
        #
        # 2. Draw the connecting "skeleton" lines using the HAND_CONNECTIONS
        #    pairs defined near the top of this file:
        #       cv2.line(frame_bgr, points[start_idx], points[end_idx],
        #                COLOR_CONNECTION, 2)
        #    Draw these BEFORE the circles below, so the dots end up
        #    sitting visually on top of the lines.
        #
        # 3. Draw a small filled circle at each point using
        #    COLOR_LANDMARK:
        #       cv2.circle(frame_bgr, point, 5, COLOR_LANDMARK, -1)
        #
        # Replace the line below with your implementation.
        points=[(int(lm.x*frame_width),int(lm.y*frame_height))
                for lm in hand_landmarks]
        for start_idx,end_idx in HAND_CONNECTIONS:
            cv2.line(frame_bgr,points[start_idx],points[end_idx],COLOR_CONNECTION,2)
        for point in points:
            cv2.circle(frame_bgr,point,5,COLOR_LANDMARK,-1)

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
    raised=0
    for tip_idx,pip_idx in zip(FINGER_TIPS[1:],FINGER_PIPS[1:]):
        if hand_landmarks[tip_idx].y<hand_landmarks[pip_idx].y:
            raised+=1
    thumb_tip=hand_landmarks[FINGER_TIPS[0]]
    thumb_joint=hand_landmarks[FINGER_PIPS[0]]
    wrist=hand_landmarks[0]

    if abs(thumb_tip.x-wrist.x)>abs(thumb_joint.x-wrist.x):
        raised+=1
    return raised


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

    Each checkpoint is attempted independently: if a checkpoint isn't
    implemented yet (or has a bug), the window stays open and shows
    what IS working, plus a status message telling you which
    checkpoint still needs attention — instead of crashing the whole
    script. This means you'll see real progress after finishing each
    checkpoint, rather than only seeing anything once all four are done.
    """
    print("Starting AIDL Forge hand tracking (starter)...")
    print("Press 'q' in the video window to quit.\n")

    # Checkpoint 1: if this isn't implemented yet, we still open the
    # webcam window below, just without any detection running.
    landmarker: Optional[mp_vision.HandLandmarker] = None
    try:
        landmarker = create_hand_landmarker()
    except NotImplementedError as error:
        print(f"[Checkpoint 1 pending] {error}")

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

            status_lines = []
            result = None
            finger_count: Optional[int] = None

            if landmarker is None:
                status_lines.append("Checkpoint 1 pending: detection disabled")
            else:
                # Checkpoint 2: attempt detection, but keep going even
                # if it isn't implemented yet.
                timestamp_ms = int((time.time() - start_time) * 1000)
                try:
                    result = detect_hands(landmarker, frame, timestamp_ms)
                except NotImplementedError as error:
                    warn_once(2, error)
                    status_lines.append("Checkpoint 2 pending: detection disabled")

            if result is not None:
                # Checkpoint 3: attempt to draw landmarks, but keep
                # going even if it isn't implemented yet.
                try:
                    frame = draw_landmarks(frame, result)
                except NotImplementedError as error:
                    warn_once(3, error)
                    status_lines.append("Checkpoint 3 pending: landmarks not drawn")

                # Checkpoint 4: attempt to count fingers, but keep
                # going even if it isn't implemented yet.
                if result.hand_landmarks:
                    try:
                        finger_count = count_fingers(result.hand_landmarks[0])
                    except NotImplementedError as error:
                        warn_once(4, error)
                        status_lines.append("Checkpoint 4 pending: count not shown")

            if finger_count is not None:
                status_lines.append(f"Fingers: {finger_count}")
            elif result is not None and not result.hand_landmarks:
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
        if landmarker is not None:
            landmarker.close()
        print("\nWebcam released. Goodbye!")


if __name__ == "__main__":
    main()