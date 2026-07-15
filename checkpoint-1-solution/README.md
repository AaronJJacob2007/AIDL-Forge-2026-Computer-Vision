# Checkpoint 1 — Reference Solution (Checkpoints 1 & 2)

This folder is a **reference point**, not a new assignment. It exists for students who fall behind during the 80-minute practical and need to catch up without losing the rest of the session.

## What's Solved Here

- **Checkpoint 1** — `create_hand_landmarker()` — fully implemented
- **Checkpoint 2** — `detect_hands()` — fully implemented

## What's Still Left To Do

- **Checkpoint 3** — `draw_landmarks()` — still a `TODO`
- **Checkpoint 4** — `count_fingers()` — still a `TODO`

If you're behind on Checkpoints 1 or 2 in your own `starter/hand_tracking.py`, copy this folder's `hand_tracking.py` into your working copy (or copy just the two solved functions), then continue on to Checkpoints 3 and 4 yourself.

## Before Running

Make sure `hand_landmarker.task` is present in this folder — see [`setup.md`](../setup.md) Step 5 if it isn't.

## Running

```bash
# from this folder, with your virtual environment activated
python hand_tracking.py
```

The window will open and detect your hand. Since Checkpoints 3 and 4 aren't implemented yet, you won't see landmark dots or a finger count — instead, the window will show an on-screen status message telling you which checkpoint is still pending (and the terminal will print the same thing once, the first time it happens). The app stays running the whole time; nothing crashes. Implement Checkpoint 3 first to confirm detection is working visually, then move on to Checkpoint 4.

## Why We Don't Give You Checkpoints 3 & 4 Solved

Checkpoint 3 (drawing landmarks) and Checkpoint 4 (finger counting) are the core logic of this workshop — the parts where you actually reason about coordinate geometry and hand anatomy. Skipping them would mean leaving the workshop without the main learning outcome. Use the instructor and [`docs/student-guide.md`](../docs/student-guide.md) if you're stuck, rather than jumping straight to [`completed/`](../completed/).

## Next Reference Point

If you finish Checkpoints 3 and 4 and want to compare your final result, see [`completed/`](../completed/) for the fully working reference implementation.