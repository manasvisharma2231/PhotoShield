"""
Backend package for Photo Shield.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for p in (str(PROJECT_ROOT), str(BASE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.main import app
except (ImportError, ModuleNotFoundError):
    from .main import app

__all__ = ["app"]
