"""
Root-level entrypoint for PhotoShield API on Render.
Binds to 0.0.0.0 and port $PORT (default 10000).
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import FastAPI app from backend.main
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 PhotoShield starting on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
