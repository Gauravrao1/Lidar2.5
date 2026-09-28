"""Vercel entry — ultra-fast cold start."""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["VERCEL"] = "1"
os.environ.setdefault("LIDAR_DATA_DIR", str(PROJECT_ROOT / "data" / "sample"))
os.environ.setdefault("LIDAR_MAX_FRAMES", "5")

from src.viz.standalone_dashboard import app, load_all, state

if state.max_frames == 0:
    load_all(os.environ["LIDAR_DATA_DIR"], 5, None)

app = app.server