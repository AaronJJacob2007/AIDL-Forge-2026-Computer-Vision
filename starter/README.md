# Starter — Hand Tracking

This is your starting point for the workshop practical. The boilerplate (webcam capture, window management, cleanup, and the main loop) is already written — your job is to complete **four checkpoints** inside [`hand_tracking.py`](hand_tracking.py).

## Before You Begin

1. Make sure you've completed [`setup.md`](../setup.md) at the repository root, including downloading the model file.
2. Copy the model file into this folder if you haven't already:

   ```bash
   # from the repository root
   cp starter/hand_landmarker.task starter/hand_landmarker.task   # already here if you followed setup.md
   ```

   You should see `hand_landmarker.task` sitting next to `hand_tracking.py` in this folder. If it isn't there yet, re-check Step 5 of `setup.md`.

## What's Already Done For You

- Webcam capture loop (`cv2.VideoCapture`)
- Window creation and display (`cv2.imshow`)
- Clean shutdown on pressing `q` (camera release, window cleanup, landmarker close)
- On-screen status overlay, including per-checkpoint pending messages
- All imports and configuration constants

**Note:** the webcam window opens and stays open from the very first run, even before you've implemented anything. Each checkpoint is attempted independently — if one isn't done yet, you'll see an on-screen message like "Checkpoint 3 pending: landmarks not drawn" instead of the whole script crashing. This means you'll see real, visible progress after finishing each checkpoint rather than only seeing something once all four are done.

## Your Task: Four Checkpoints

Open `hand_tracking.py` and look for `TODO` comments. There are four:

### Checkpoint 1 — `create_hand_landmarker()`

Initialize a MediaPipe `HandLandmarker` using the Tasks API, pointed at the downloaded model file, running in `VIDEO` mode.

**You'll know it's working when:** the on-screen message no longer says "Checkpoint 1 pending: detection disabled".

### Checkpoint 2 — `detect_hands()`

Convert the incoming BGR webcam frame to RGB, wrap it in a `mp.Image`, and run detection for the current video timestamp.

**You'll know it's working when:** the on-screen message no longer says "Checkpoint 2 pending: detection disabled" — though you still won't see landmarks drawn until Checkpoint 3.

### Checkpoint 3 — `draw_landmarks()`

Loop over the detected hand's 21 landmarks, convert each from normalized coordinates to pixel coordinates, and draw a circle at each point.

**You'll know it's working when:** you see small green dots tracking your hand in the webcam window, and the "Checkpoint 3 pending" message disappears.

### Checkpoint 4 — `count_fingers()`

Compare fingertip and joint landmark positions to determine how many fingers are raised, handling the thumb as a special case.

**You'll know it's working when:** the on-screen "Fingers: N" counter accurately reflects how many fingers you're holding up, and the "Checkpoint 4 pending" message disappears.

## Running Your Code

```bash
# from this folder, with your virtual environment activated
python hand_tracking.py
```

Press `q` in the video window to quit at any time.

## Stuck?

- Re-read the docstring and inline comments directly above each `TODO` — they contain the exact hints and method names you need.
- Compare your progress against [`checkpoint-1/`](../checkpoint-1/), which has Checkpoints 1 and 2 already solved.
- Ask an instructor or teaching assistant — don't spend more than 5–10 minutes stuck on any single checkpoint.

## Reference Material

- [MediaPipe Hand Landmarker documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [`docs/student-guide.md`](../docs/student-guide.md) for exercises and mini challenges