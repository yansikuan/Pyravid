"""Shared, portable configuration helpers for CAM.

Configuration values come from environment variables or explicit CLI
arguments. Secrets must never be stored in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: str) -> Path:
    """Return an environment-configured path, relative to the project root."""
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


DATA_DIR = env_path("CAM_DATA_DIR", "data")
OUTPUT_DIR = env_path("CAM_OUTPUT_DIR", "artifacts/outputs")
CACHE_DIR = env_path("CAM_CACHE_DIR", "artifacts/cache")


def api_key(provider: str, key_path: str | None = None) -> str:
    """Load a provider key from the environment, with legacy file support.

    ``key_path`` remains supported for existing experiment commands, but new
    usage should rely on environment variables.
    """
    env_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider.lower())

    if env_name and os.getenv(env_name):
        return os.environ[env_name].strip()
    if key_path:
        path = Path(key_path).expanduser()
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    if env_name:
        raise RuntimeError(
            f"Missing credential: set {env_name}. See .env.example for setup."
        )
    return "EMPTY"
