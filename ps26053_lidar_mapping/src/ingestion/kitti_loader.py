import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

@dataclass
class LidarFrame:
    points: np.ndarray      # (N, 4) float32 [x, y, z, intensity]
    labels: Optional[np.ndarray]  # (N,) int32 remapped labels [0=T, 1=S, 2=D, 3=X], None if no .label file
    frame_id: int
    sequence_id: str

REQUIRED_CLASSES = [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99]

def load_class_remap(config_path: str | Path) -> Dict[int, int]:
    """Load class_remap.yaml. Returns dict mapping raw SemanticKITTI ID -> output class ID.
    Raises ValueError if any of the 28 standard classes is missing."""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Support multiple YAML layouts
    if 'remap' in data:
        raw_remap = data['remap']
    elif 'learning_map' in data:
        raw_remap = data['learning_map']
    else:
        raw_remap = data
        
    remap = {int(k): int(v) for k, v in raw_remap.items()}
    
    missing = [c for c in REQUIRED_CLASSES if c not in remap]
    if missing:
        raise ValueError(f"Missing required classes in remap: {missing}")
        
    return remap

def load_bin(bin_path: str | Path) -> np.ndarray:
    """Load .bin file -> (N, 4) float32 array [x, y, z, intensity]."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def load_labels(label_path: str | Path, remap: Dict[int, int]) -> np.ndarray:
    """Load .label file, extract semantic label (lower 16 bits), apply remap -> (N,) int32.
    Raises ValueError if any label ID is not in remap."""
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_ids = (raw_labels & 0xFFFF).astype(np.int32)
    
    unique_labels = np.unique(semantic_ids)
    for lbl in unique_labels:
        if int(lbl) not in remap:
            raise ValueError(f"Label ID {lbl} not found in remap configuration.")
            
    remapped_labels = np.vectorize(remap.get)(semantic_ids).astype(np.int32)
    return remapped_labels

def load_frame(bin_path: str | Path, label_path: Optional[str | Path], remap: Dict[int, int], frame_id: int = 0, sequence_id: str = '08') -> LidarFrame:
    """Load a single frame (bin + optional label)."""
    points = load_bin(bin_path)
    labels = None
    if label_path is not None and Path(label_path).exists():
        labels = load_labels(label_path, remap)
        
    return LidarFrame(points=points, labels=labels, frame_id=frame_id, sequence_id=sequence_id)

def get_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    while current_dir.parent != current_dir:
        if (current_dir / "configs").is_dir():
            return current_dir
        current_dir = current_dir.parent
    return Path(__file__).resolve().parent.parent.parent

class SemanticKITTIDataset:
    def __init__(self, data_dir: str | Path, sequence: str = '08', config_path: Optional[str | Path] = None):
        """Initialize dataset. data_dir should contain sequences/<seq>/velodyne/*.bin and labels/*.label"""
        self.data_dir = Path(data_dir)
        self.sequence = sequence
        
        if config_path is None:
            root = get_project_root()
            self.config_path = root / "configs" / "class_remap.yaml"
        else:
            self.config_path = Path(config_path)
            
        self.remap = load_class_remap(self.config_path)
        
        self.velodyne_dir = self.data_dir / "sequences" / self.sequence / "velodyne"
        self.labels_dir = self.data_dir / "sequences" / self.sequence / "labels"
        
        if not self.velodyne_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.velodyne_dir}")
            
        self.bin_files = sorted(list(self.velodyne_dir.glob("*.bin")))
        if not self.bin_files:
            raise ValueError(f"No .bin files found in {self.velodyne_dir}")

    def __len__(self) -> int:
        return len(self.bin_files)

    def __getitem__(self, idx: int) -> LidarFrame:
        bin_path = self.bin_files[idx]
        frame_id = int(bin_path.stem)
        label_path = self.labels_dir / f"{bin_path.stem}.label"
        
        if not label_path.exists():
            label_path = None
            
        return load_frame(bin_path, label_path, self.remap, frame_id, self.sequence)
