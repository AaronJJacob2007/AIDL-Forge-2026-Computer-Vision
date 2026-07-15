"""
AIDL Buildathon Forge 2026 — Day 3: Computer Vision
=====================================================

backend/utils.py

What this file is
------------------
Small utility helpers for the FastAPI backend: loading optional
configuration from a `.env` file and turning it into a typed
`Settings` object.

Instructor note:
    We intentionally avoid adding a dependency like `python-dotenv`
    for something this small — `load_env_file()` below is a ~15-line
    illustration of what such a library does under the hood, which is
    a nice discussion point ("here's the kind of thing a library
    saves you from writing yourself, once your needs grow").
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Resolved relative to this file's location, so `.env` is found at the
# repository root regardless of the working directory the backend is
# launched from.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
ENV_FILE_PATH = REPO_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE_PATH) -> None:
    """
    Load KEY=VALUE pairs from a `.env` file into the process environment.

    Lines starting with `#` and blank lines are ignored. Existing
    environment variables are never overwritten — this lets you
    override any `.env` value by setting it directly in your shell.

    Args:
        path: Path to the `.env` file. Silently does nothing if the
            file doesn't exist, since `.env` is optional (see
            `.env.example` for the template and default values used
            when no `.env` is present).
    """
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key and key not in os.environ:
            os.environ[key] = value


def _parse_cors_origins(raw_value: str) -> List[str]:
    """
    Parse a comma-separated CORS_ORIGINS string into a clean list.

    Args:
        raw_value: Comma-separated origins, e.g.
            "http://127.0.0.1:5500,http://localhost:5500".

    Returns:
        List[str]: Trimmed, non-empty origin strings.
    """
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


@dataclass
class Settings:
    """
    Typed application settings, populated from environment variables
    (with sensible defaults if unset).

    Attributes:
        host: Host address the FastAPI server binds to.
        port: Port the FastAPI server binds to.
        cors_origins: Browser origins allowed to call the API.
        camera_index: Index of the webcam device passed to OpenCV.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["http://127.0.0.1:5500", "http://localhost:5500"])
    camera_index: int = 0


def get_settings() -> Settings:
    """
    Build a `Settings` object from environment variables, loading
    `.env` first if present.

    Returns:
        Settings: The resolved application configuration.
    """
    load_env_file()

    return Settings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        cors_origins=_parse_cors_origins(
            os.getenv("CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500")
        ),
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
    )