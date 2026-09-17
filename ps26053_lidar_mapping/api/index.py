"""
Vercel serverless entry point for the Adaptive LiDAR Dashboard.
Optimized for Vercel's constraints:
  - 10s function timeout (free tier) / 60s (pro)
  - 1024MB memory limit
  - 250MB function size
"""
import os
import sys
from pathlib import Path

# Set project root so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use only 20 frames for Vercel (fast cold start, low memory)
os.environ.setdefault("LIDAR_DATA_DIR", str(PROJECT_ROOT / "data" / "sample"))
os.environ.setdefault("LIDAR_MAX_FRAMES", "20")
os.environ.setdefault("VERCEL", "1")

from src.viz.standalone_dashboard import app, load_all, state

# Pre-process on cold start (only if not already loaded)
if state.max_frames == 0:
    data_dir = os.environ.get("LIDAR_DATA_DIR", str(PROJECT_ROOT / "data" / "sample"))
    max_frames = int(os.environ.get("LIDAR_MAX_FRAMES", "20"))
    load_all(data_dir, max_frames, None)

# Vercel expects 'app' as WSGI application
app = app.server