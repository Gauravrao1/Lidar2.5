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
from src.viz.standalone_dashboard import load_all

data_dir = Path(os.environ["LIDAR_DATA_DIR"])
if not (data_dir / "velodyne").is_dir():
    data_dir = PROJECT_ROOT / "data" / "sample"
checkpoint = PROJECT_ROOT / "checkpoints" / "pointnet_3class.pth"
load_all(
    data_dir,
    int(os.environ["LIDAR_MAX_FRAMES"]),
    str(checkpoint) if checkpoint.exists() else None,
)

app = app.server