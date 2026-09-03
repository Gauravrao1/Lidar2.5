"""
Smoke test — runs the standalone pipeline end-to-end on sample data.
Asserts on output shapes, label ranges, and that metrics are produced.
"""
import sys
import os
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_sample_data_exists():
    """Verify sample data was generated."""
    velodyne_dir = PROJECT_ROOT / "data" / "sample" / "velodyne"
    labels_dir = PROJECT_ROOT / "data" / "sample" / "labels"
    
    bin_files = list(velodyne_dir.glob("*.bin"))
    label_files = list(labels_dir.glob("*.label"))
    
    assert len(bin_files) >= 1, f"No .bin files found in {velodyne_dir}"
    assert len(label_files) >= 1, f"No .label files found in {labels_dir}"
    assert len(bin_files) == len(label_files), "Mismatched .bin and .label file counts"


def test_kitti_loader():
    """Test that kitti_loader correctly loads and remaps a sample frame."""
    from src.ingestion.kitti_loader import load_class_remap, load_bin, load_labels
    
    remap = load_class_remap(PROJECT_ROOT / "configs" / "class_remap.yaml")
    assert len(remap) >= 26, f"Remap has only {len(remap)} entries"
    
    bin_files = sorted((PROJECT_ROOT / "data" / "sample" / "velodyne").glob("*.bin"))
    assert len(bin_files) > 0
    
    points = load_bin(bin_files[0])
    assert points.ndim == 2
    assert points.shape[1] == 4
    assert points.dtype == np.float32
    
    label_path = PROJECT_ROOT / "data" / "sample" / "labels" / f"{bin_files[0].stem}.label"
    if label_path.exists():
        labels = load_labels(label_path, remap)
        assert labels.shape[0] == points.shape[0], "Point/label count mismatch"
        assert set(np.unique(labels)).issubset({0, 1, 2, 3}), f"Unexpected labels: {np.unique(labels)}"


def test_segmentation_inference():
    """Test that PointNet backend produces valid output shapes and label values."""
    from src.segmentation.pointnet_infer import PointNetBackend
    from src.ingestion.kitti_loader import load_bin
    
    backend = PointNetBackend(device="cpu", num_points=4096)
    
    bin_files = sorted((PROJECT_ROOT / "data" / "sample" / "velodyne").glob("*.bin"))
    assert len(bin_files) > 0
    
    points = load_bin(bin_files[0])
    preds = backend.predict(points)
    
    assert preds.shape[0] == points.shape[0], f"Output shape {preds.shape} != input {points.shape[0]}"
    assert preds.dtype == np.int32
    assert all(l in {0, 1, 2, 3} for l in np.unique(preds)), f"Invalid labels: {np.unique(preds)}"


def test_grid_insert_from_real_data():
    """Test that grid insertion works with real sample data."""
    from src.ingestion.kitti_loader import load_bin, load_labels, load_class_remap
    from src.grid.resolution import GridConfig
    from src.grid.adaptive_grid import AdaptiveGrid
    
    config = GridConfig()
    grid = AdaptiveGrid(config)
    
    remap = load_class_remap(PROJECT_ROOT / "configs" / "class_remap.yaml")
    bin_files = sorted((PROJECT_ROOT / "data" / "sample" / "velodyne").glob("*.bin"))
    
    points = load_bin(bin_files[0])
    label_path = PROJECT_ROOT / "data" / "sample" / "labels" / f"{bin_files[0].stem}.label"
    labels = load_labels(label_path, remap)
    
    grid.insert(points[:, :3], labels, frame_id=0)
    
    assert grid.occupied_cell_count() > 0, "Grid should have cells after insertion"
    assert grid.memory_bytes() > 0, "Memory should be positive"
    
    cells = grid.get_cells()
    assert len(cells) > 0
    for key, cls, center, size, frame in cells:
        assert cls in {0, 1, 2, 3}
        assert size > 0
        assert frame == 0


def test_pipeline_end_to_end():
    """Run the full pipeline on sample data and verify outputs."""
    from src.pipeline.standalone_runner import run_pipeline
    
    results = run_pipeline(
        data_dir=PROJECT_ROOT / "data" / "sample",
        max_frames=3,
        warmup_frames=1,
    )
    
    assert results is not None
    assert results["num_frames"] == 3
    assert results["final_cell_count"] > 0
    
    # Latency
    assert results["latency_report"].mean_ms > 0
    assert results["latency_report"].fps > 0
    
    # Memory
    assert results["memory_report"].adaptive_cells > 0
    assert results["memory_report"].reduction_vs_2d_pct > 0
    assert results["memory_report"].reduction_vs_3d_pct > 0
    
    # Check that report files were created
    reports_dir = PROJECT_ROOT / "reports"
    assert (reports_dir / "latency.csv").exists(), "Latency CSV not generated"
    assert (reports_dir / "per_frame_log.csv").exists(), "Per-frame log not generated"
