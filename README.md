# Computer Vision Workshop

Workshop materials for AIDL Buildathon Forge 2026 at Fr. Conceicao Rodrigues Institute of Technology. Starting with a webcam feed, participants work through hand detection, landmark drawing and finger counting using Python, OpenCV and the MediaPipe Tasks API.

The repository also includes a FastAPI service and a small browser interface that displays the latest finger count.

## Before you start

You'll need Python 3.12 and a working webcam. Follow [setup.md](setup.md) for the environment, dependencies and model setup.

## Workshop checkpoints

1. Capture frames from the webcam.
2. Detect a hand.
3. Draw its landmarks.
4. Count raised fingers.

Start in `starter/`. The `checkpoint-1-solution/` folder contains the first checkpoint solution, and `completed/` contains the finished hand-tracking exercise.

## Run the examples

After completing the setup, run one of the exercises from its folder:

```sh
cd completed
python hand_tracking.py
```

For the API, open a separate terminal and run:

```sh
cd backend
uvicorn main:app --reload
```

The API exposes `GET /count` with the latest detection and finger count. Open `frontend/index.html` for the browser interface; see [setup.md](setup.md) for the full instructions.

## Repository guide

| Folder | Contents |
| --- | --- |
| `starter/` | Starting point for the workshop |
| `checkpoint-1-solution/` | Webcam capture solution |
| `completed/` | Completed hand-tracking exercise |
| `backend/` | FastAPI service |
| `frontend/` | Browser interface |
| `bonus/` | Optional extension ideas |
| `assets/` | Workshop assets |

The workshop uses Python, OpenCV, MediaPipe, NumPy, FastAPI and HTML/CSS/JavaScript.

## License

[MIT](LICENSE). Maintained by Aaron Jacob for the AIDL Club, FCRIT.
