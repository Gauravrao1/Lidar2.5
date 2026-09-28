"""
Vercel serverless entry point for the Adaptive LiDAR Dashboard.
Ultra-optimized for Vercel free tier:
  - 10 frames only (fast cold start ~3s)
  - 4K points per frame (low memory)
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ultra-light for Vercel: 10 frames, 4K points
os.environ.setdefault("LIDAR_DATA_DIR", str(PROJECT_ROOT / "data" / "sample"))
os.environ.setdefault("LIDAR_MAX_FRAMES", "10")
os.environ.setdefault("VERCEL", "1")

from src.viz.standalone_dashboard import app, load_all, state

if state.max_frames == 0:
    data_dir = os.environ.get("LIDAR_DATA_DIR", str(PROJECT_ROOT / "data" / "sample"))
    max_frames = int(os.environ.get("LIDAR_MAX_FRAMES", "10"))
    load_all(data_dir, max_frames, None)

app = app.server