# assets/diagrams

Place polished, presentation-ready diagrams here — exported PNG/SVG versions of the ASCII architecture diagram in the root `README.md` and `docs/architecture.md`, suitable for embedding in slides or a published blog post about this workshop.

Suggested contents:

- `architecture-overview.png` — the Webcam → OpenCV → MediaPipe → Finger Counter → FastAPI → Browser pipeline
- `hand-landmarks-diagram.png` — the 21-point MediaPipe hand landmark map, annotated with finger tip/PIP indices used in `count_fingers()`

Prefer SVG where possible for crisp rendering at any size; fall back to PNG at 2x resolution for raster diagrams.