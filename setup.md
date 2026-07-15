# Environment Setup

Welcome to **AIDL Buildathon Forge 2026 – Computer Vision Workshop!**

Please complete this setup **before** the workshop begins. The live coding session is only **80 minutes**, so we won't have time to troubleshoot installations during the practical.

**Estimated setup time:** 10–15 minutes

---

# 1. Prerequisites

Before you begin, ensure you have:

- **Python 3.12.x (64-bit)**
- A working webcam (built-in laptop camera is fine)
- VS Code (recommended) or another Python IDE
- Around **250 MB** of free disk space

> **Supported Python Version**
>
> ✅ Python **3.12.x**
>
> ❌ Python 3.13 is **not recommended** for this workshop because MediaPipe may not yet fully support it.

---

# 2. Verify Your Python Version

Open a terminal.

### Windows

```powershell
python --version
```

### macOS / Linux

```bash
python3 --version
```

Expected:

```
Python 3.12.x
```

If your version is **3.13**, install Python 3.12 before continuing.

---

# 3. Extract the Workshop Files

Extract the provided ZIP file.

After extraction your folder should look like:

```
AIDL-Buildathon-Forge-2026/
│
├── assets/
├── backend/
├── bonus/
├── checkpoint-1/
├── completed/
├── frontend/
├── starter/
│
├── .env.example
├── LICENSE
├── README.md
├── requirements.txt
└── setup.md
```

> **Important**
>
> Do **not** move files between folders.
> The repository structure is required for the workshop.

---

# 4. Open the Project

Open **VS Code**.

Select:

**File → Open Folder**

Choose:

```
AIDL-Buildathon-Forge-2026
```

Open a new terminal inside VS Code.

---

# 5. Create a Virtual Environment

A virtual environment keeps the workshop dependencies isolated from other Python projects.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Once activated, your terminal should begin with:

```
(.venv)
```

---

# 6. Install Dependencies

Upgrade pip first:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|----------|----------|
| OpenCV | Webcam access and drawing |
| MediaPipe | Hand tracking (Tasks API) |
| NumPy | Numerical computations |

---

# 7. Model Files

All required MediaPipe model files are **already included** in the workshop package.

No additional downloads are required.

For example:

```
starter/
└── hand_landmarker.task
```

If this file exists, you're ready.

---

# 8. Verify Your Installation

Run:

```bash
python -c "import cv2, mediapipe, numpy; print('OpenCV:', cv2.__version__); print('MediaPipe:', mediapipe.__version__); print('NumPy:', numpy.__version__)"
```

Expected:

```
OpenCV: 4.10.0
MediaPipe: 0.10.35
NumPy: 1.26.4
```

---

# 9. Test the Workshop Project

Move into the starter folder:

```bash
cd starter
```

Run:

```bash
python hand_tracking.py
```

Expected:

- Camera opens
- No error messages
- Hand tracking starts (or "Checkpoint 1 pending", depending on the workshop version)

If this works, your setup is complete.

---

# 10. Troubleshooting

| Problem | Solution |
|----------|----------|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` again. |
| `requirements.txt` not found | Ensure you're in the project root (the folder containing `requirements.txt`). |
| `Activate.ps1` not found | Delete the incomplete `.venv` folder and recreate it. |
| Webcam not detected | Close Zoom, Teams, Discord, or any application using the webcam. |
| `hand_landmarker.task` not found | Ensure the file exists inside the `starter` folder. |
| Camera permission denied (macOS) | Enable camera permission for Terminal or VS Code under System Settings → Privacy & Security → Camera. |

---

# You're Ready!

If all of the following work:

- ✅ Python 3.12
- ✅ Virtual environment activated
- ✅ Dependencies installed
- ✅ `python hand_tracking.py` runs successfully

you're fully prepared for the **AIDL Buildathon Forge 2026 – Computer Vision Workshop**.

We'll begin coding from the **starter** folder during the live session.