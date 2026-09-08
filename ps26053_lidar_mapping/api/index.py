import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "LIDAR_DATA_DIR",
    str(PROJECT_ROOT / "data" / "sample")
)
os.environ.setdefault("LIDAR_MAX_FRAMES", "50")

from src.viz.standalone_dashboard import app

app = app.server