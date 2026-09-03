"""
RandLA-Net inference wrapper using Open3D-ML.

DOCUMENTED FALLBACK — requires Python <= 3.12 and Open3D-ML installed.
Cannot be executed in the current environment (Python 3.13, no Open3D).

To use this backend:
1. Install Python 3.11 or 3.12
2. Create a venv with that Python version
3. pip install open3d torch
4. Download checkpoint:
   wget https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth
5. Place checkpoint in checkpoints/randlanet_semantickitti.pth
6. Set backend='randlanet' in standalone_runner.py config

This file is provided as a ready-to-use implementation — it is NOT a stub.
It simply cannot be tested in the current Python 3.13 environment.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import logging

from .model_interface import SegmentationBackend

logger = logging.getLogger(__name__)

# The SemanticKITTI learning_map used by Open3D-ML RandLA-Net
# Maps the 19-class model output to our 4-class scheme
RANDLANET_19_TO_4CLASS = {
    0: 3,   # unlabeled -> Ignore
    1: 2,   # car -> Dynamic
    2: 2,   # bicycle -> Dynamic
    3: 2,   # motorcycle -> Dynamic
    4: 2,   # truck -> Dynamic
    5: 2,   # other-vehicle -> Dynamic
    6: 2,   # person -> Dynamic
    7: 2,   # bicyclist -> Dynamic
    8: 2,   # motorcyclist -> Dynamic
    9: 0,   # road -> Terrain
    10: 0,  # parking -> Terrain
    11: 0,  # sidewalk -> Terrain
    12: 0,  # other-ground -> Terrain
    13: 1,  # building -> Static
    14: 1,  # fence -> Static
    15: 1,  # vegetation -> Static
    16: 1,  # trunk -> Static
    17: 0,  # terrain -> Terrain
    18: 1,  # pole -> Static
    19: 1,  # traffic-sign -> Static  (if model outputs 20 classes)
}

CHECKPOINT_URL = (
    "https://storage.googleapis.com/open3d-releases/model-zoo/"
    "randlanet_semantickitti_202201071330utc.pth"
)


class RandLANetBackend(SegmentationBackend):
    """Open3D-ML RandLA-Net segmentation backend.

    Requires:
        - Python <= 3.12
        - open3d >= 0.18.0
        - torch (compatible version)
        - Pretrained checkpoint file

    This is the PREFERRED backend when the environment supports it,
    as it provides ~60% mIoU on SemanticKITTI without any training.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
    ):
        """Initialize RandLA-Net backend.

        Args:
            checkpoint_path: Path to .pth checkpoint. If None, attempts to
                find it in checkpoints/randlanet_semantickitti.pth
            device: 'cpu' or 'cuda'
        """
        try:
            import open3d.ml.torch as ml3d
        except ImportError as e:
            raise RuntimeError(
                "Open3D-ML is not installed. This backend requires:\n"
                "  - Python <= 3.12\n"
                "  - pip install open3d torch\n"
                f"Original error: {e}"
            ) from e

        self._ml3d = ml3d
        self._device = device

        # Locate checkpoint
        if checkpoint_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            checkpoint_path = project_root / "checkpoints" / "randlanet_semantickitti.pth"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"RandLA-Net checkpoint not found at {checkpoint_path}\n"
                f"Download it from:\n  {CHECKPOINT_URL}\n"
                f"Place it at: {checkpoint_path}"
            )

        # Load model
        cfg_path = self._find_config()
        self._cfg = ml3d.utils.Config.load_from_file(cfg_path)
        self._model = ml3d.models.RandLANet(**self._cfg.model)
        self._pipeline = ml3d.pipelines.SemanticSegmentation(
            model=self._model, device=device
        )
        self._pipeline.load_ckpt(str(checkpoint_path))
        self._model.eval()

        logger.info(
            "RandLA-Net backend initialized: checkpoint=%s, device=%s",
            checkpoint_path,
            device,
        )

    def _find_config(self) -> str:
        """Find the RandLA-Net SemanticKITTI config file."""
        import open3d.ml as _ml

        ml_root = Path(_ml.__file__).parent
        cfg_candidates = [
            ml_root / "configs" / "randlanet_semantickitti.yml",
            ml_root / "ml3d" / "configs" / "randlanet_semantickitti.yml",
        ]
        for p in cfg_candidates:
            if p.exists():
                return str(p)

        raise FileNotFoundError(
            "Cannot find randlanet_semantickitti.yml config. "
            f"Searched: {[str(p) for p in cfg_candidates]}"
        )

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Run RandLA-Net inference on a point cloud.

        Args:
            points: (N, 4) float32 array [x, y, z, intensity]

        Returns:
            (N,) int32 array with labels in {0=Terrain, 1=Static, 2=Dynamic, 3=Ignore}
        """
        import torch

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected (N, 4) array, got {points.shape}")

        n_points = points.shape[0]

        # Open3D-ML expects specific data format
        data = {
            "point": points[:, :3].astype(np.float32),
            "feat": points[:, 3:4].astype(np.float32) if points.shape[1] >= 4
                    else np.ones((n_points, 1), dtype=np.float32),
        }

        with torch.no_grad():
            result = self._pipeline.run_inference(data)

        # result is per-point 19-class predictions
        raw_labels = np.asarray(result["predict_labels"], dtype=np.int32)

        # Remap 19-class -> 4-class
        remapped = np.full(n_points, 3, dtype=np.int32)  # default Ignore
        for src_class, dst_class in RANDLANET_19_TO_4CLASS.items():
            remapped[raw_labels == src_class] = dst_class

        return remapped

    @property
    def name(self) -> str:
        return f"RandLA-Net (Open3D-ML, {self._device})"

    @property
    def num_classes(self) -> int:
        return 4
