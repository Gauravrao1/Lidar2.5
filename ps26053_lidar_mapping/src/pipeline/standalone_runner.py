"""
Standalone pipeline runner — single-process entry point.

This is the PRIMARY way to run the system. It executes:
  1. Load config + sample data
  2. Initialize segmentation backend (PointNet)
  3. Initialize AdaptiveGrid
  4. For each frame: load → segment → grid insert → decay → log metrics
  5. Compute final IoU (using GT labels for evaluation)
  6. Generate metrics report

Usage:
  cd ps26053_lidar_mapping
  .venv\\Scripts\\python.exe -m src.pipeline.standalone_runner --data-dir data/sample --max-frames 10
"""

import argparse
import sys
import time
import logging
import csv
from pathlib import Path
from typing import Optional

import numpy as np

# Setup path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.kitti_loader import (
    load_class_remap, load_bin, load_labels, LidarFrame,
)
from src.segmentation.model_interface import SegmentationBackend
from src.segmentation.pointnet_infer import PointNetBackend
from src.grid.resolution import GridConfig, load_grid_config
from src.grid.adaptive_grid import AdaptiveGrid
from src.metrics.iou import compute_iou, iou_by_distance
from src.metrics.latency import get_hardware_info, LatencyReport, save_latency_csv
from src.metrics.memory import compute_memory_report
from src.metrics.report_generator import generate_metrics_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("standalone_runner")


def discover_frames(data_dir: Path) -> list[tuple[Path, Optional[Path]]]:
    """Find all .bin files and their matching .label files."""
    # Support both flat layout (data/sample/velodyne/) and SemanticKITTI layout
    velodyne_dirs = [
        data_dir / "velodyne",
        data_dir / "sequences" / "08" / "velodyne",
    ]
    
    for vdir in velodyne_dirs:
        if vdir.exists():
            bin_files = sorted(vdir.glob("*.bin"))
            if bin_files:
                label_dir = vdir.parent / "labels"
                frames = []
                for bf in bin_files:
                    lf = label_dir / f"{bf.stem}.label"
                    frames.append((bf, lf if lf.exists() else None))
                return frames
    
    raise FileNotFoundError(
        f"No .bin files found in any of: {[str(d) for d in velodyne_dirs]}"
    )


def run_pipeline(
    data_dir: Path,
    grid_config_path: Optional[Path] = None,
    remap_config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    max_frames: Optional[int] = None,
    output_dir: Optional[Path] = None,
    warmup_frames: int = 3,
    launch_dashboard: bool = False,
) -> dict:
    """Run the full pipeline end-to-end.
    
    Returns a dict with all results: IoU, latency, memory, etc.
    """
    # Defaults
    if grid_config_path is None:
        grid_config_path = PROJECT_ROOT / "configs" / "grid_config.yaml"
    if remap_config_path is None:
        remap_config_path = PROJECT_ROOT / "configs" / "class_remap.yaml"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    logger.info("Loading configs...")
    if grid_config_path.exists():
        grid_config = load_grid_config(str(grid_config_path))
    else:
        logger.warning("Grid config not found at %s, using defaults", grid_config_path)
        grid_config = GridConfig()
    
    remap = load_class_remap(remap_config_path)
    logger.info("Class remap loaded: %d entries", len(remap))

    # Initialize segmentation backend
    logger.info("Initializing segmentation backend...")
    backend: SegmentationBackend = PointNetBackend(
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device="cpu",
        num_points=16384,
    )
    logger.info("Backend: %s", backend.name)

    # Initialize grid
    grid = AdaptiveGrid(grid_config)

    # Discover frames
    frames = discover_frames(data_dir)
    if max_frames is not None:
        frames = frames[:max_frames]
    logger.info("Found %d frames to process", len(frames))

    # Per-frame processing
    all_pred_labels = []
    all_gt_labels = []
    all_points = []
    per_frame_ms = []
    per_frame_cells = []
    per_frame_memory = []

    for frame_idx, (bin_path, label_path) in enumerate(frames):
        t_start = time.perf_counter()

        # 1. Load points
        points = load_bin(bin_path)
        
        # 2. Run segmentation
        pred_labels = backend.predict(points)

        # 3. Load GT labels (if available) for evaluation
        gt_labels = None
        if label_path is not None:
            gt_labels = load_labels(label_path, remap)

        # 4. Insert into grid
        grid.insert(points[:, :3], pred_labels, frame_id=frame_idx)

        # 5. Decay stale dynamic cells
        removed = grid.decay_dynamic_cells(frame_idx)

        t_end = time.perf_counter()
        frame_ms = (t_end - t_start) * 1000.0

        # Log per-frame stats
        cell_count = grid.occupied_cell_count()
        mem_bytes = grid.memory_bytes()
        per_frame_ms.append(frame_ms)
        per_frame_cells.append(cell_count)
        per_frame_memory.append(mem_bytes)

        if gt_labels is not None:
            all_pred_labels.append(pred_labels)
            all_gt_labels.append(gt_labels)
            all_points.append(points)

        logger.info(
            "Frame %03d: %.1f ms | cells=%d | mem=%.1f KB | decayed=%d",
            frame_idx, frame_ms, cell_count, mem_bytes / 1024, removed,
        )

    # ==================== Compute Metrics ====================
    logger.info("Computing metrics...")
    
    # Hardware info
    hw_info = get_hardware_info()
    logger.info("Hardware: %s", hw_info)

    # Latency
    measured_frames = per_frame_ms[warmup_frames:] if len(per_frame_ms) > warmup_frames else per_frame_ms
    arr = np.array(measured_frames)
    latency_report = LatencyReport(
        mean_ms=float(np.mean(arr)) if len(arr) > 0 else 0.0,
        median_ms=float(np.median(arr)) if len(arr) > 0 else 0.0,
        p95_ms=float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
        fps=1000.0 / float(np.mean(arr)) if len(arr) > 0 and np.mean(arr) > 0 else 0.0,
        num_frames=len(measured_frames),
        warmup_frames=warmup_frames,
        hardware_info=hw_info,
        per_frame_ms=measured_frames,
    )
    
    latency_csv_path = output_dir / "latency.csv"
    save_latency_csv(latency_report, latency_csv_path)
    logger.info(
        "Latency: mean=%.1f ms, median=%.1f ms, p95=%.1f ms, FPS=%.1f",
        latency_report.mean_ms, latency_report.median_ms,
        latency_report.p95_ms, latency_report.fps,
    )

    # Memory
    final_cells = grid.occupied_cell_count()
    mem_report = compute_memory_report(
        adaptive_cells=final_cells,
        adaptive_bytes_per_cell=grid_config.BYTES_PER_CELL,
        r_far=grid_config.R_FAR,
        z_range=grid_config.Z_MAX - grid_config.Z_MIN,
        s_near=grid_config.S_NEAR,
    )
    logger.info(
        "Memory: adaptive=%d cells (%.2f KB) | uniform_2D=%d cells | uniform_3D=%d cells",
        mem_report.adaptive_cells, mem_report.adaptive_bytes / 1024,
        mem_report.uniform_2d_cells, mem_report.uniform_3d_cells,
    )
    logger.info(
        "Reduction: vs 2D=%.2f%% | vs 3D=%.2f%%",
        mem_report.reduction_vs_2d_pct, mem_report.reduction_vs_3d_pct,
    )

    # IoU
    iou_results = {}
    per_class_iou_final = None
    miou_final = None
    if all_gt_labels:
        all_pred = np.concatenate(all_pred_labels)
        all_gt = np.concatenate(all_gt_labels)
        all_pts = np.concatenate(all_points)
        
        per_class_iou_final, miou_final = compute_iou(all_pred, all_gt, num_classes=3)
        iou_results = iou_by_distance(all_pred, all_gt, all_pts)
        
        class_names = ["Terrain", "Static", "Dynamic"]
        for i, name in enumerate(class_names):
            logger.info("IoU %s: %.4f", name, per_class_iou_final[i])
        logger.info("mIoU: %.4f", miou_final)
    else:
        logger.warning("No GT labels available — IoU not computed")

    # Generate report
    if iou_results:
        report_path = output_dir / "metrics_table_filled.md"
        generate_metrics_table(
            latency_csv=latency_csv_path,
            memory_report=mem_report,
            iou_results=iou_results,
            hardware_info=hw_info,
            output_path=report_path,
            is_sample_run=True,
        )
        logger.info("Metrics report written to %s", report_path)

    # Save per-frame log
    frame_log_path = output_dir / "per_frame_log.csv"
    with open(frame_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "latency_ms", "cells", "memory_bytes"])
        for i, (ms, cells, mem) in enumerate(zip(per_frame_ms, per_frame_cells, per_frame_memory)):
            writer.writerow([i, f"{ms:.2f}", cells, mem])
    logger.info("Per-frame log written to %s", frame_log_path)

    results = {
        "latency_report": latency_report,
        "memory_report": mem_report,
        "iou_per_class": per_class_iou_final,
        "miou": miou_final,
        "iou_by_distance": iou_results,
        "hardware_info": hw_info,
        "num_frames": len(frames),
        "final_cell_count": final_cells,
    }

    logger.info("Pipeline complete. Processed %d frames.", len(frames))
    return results


def main():
    parser = argparse.ArgumentParser(description="PS-26053 Standalone LiDAR Pipeline")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "sample"),
                       help="Path to data directory (with velodyne/ and labels/ subdirs)")
    parser.add_argument("--grid-config", type=str, default=None,
                       help="Path to grid_config.yaml")
    parser.add_argument("--remap-config", type=str, default=None,
                       help="Path to class_remap.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to PointNet checkpoint .pth file")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for reports")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup frames to discard for latency measurement")
    parser.add_argument("--dashboard", action="store_true",
                       help="Launch Plotly Dash dashboard after processing")
    
    args = parser.parse_args()
    
    results = run_pipeline(
        data_dir=Path(args.data_dir),
        grid_config_path=Path(args.grid_config) if args.grid_config else None,
        remap_config_path=Path(args.remap_config) if args.remap_config else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        max_frames=args.max_frames,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        warmup_frames=args.warmup,
        launch_dashboard=args.dashboard,
    )
    
    print("\n" + "="*60)
    print("PIPELINE RESULTS SUMMARY")
    print("="*60)
    print(f"Frames processed: {results['num_frames']}")
    print(f"Final grid cells: {results['final_cell_count']}")
    print(f"Latency: {results['latency_report'].mean_ms:.1f} ms mean, {results['latency_report'].fps:.1f} FPS")
    print(f"Memory reduction vs 2D: {results['memory_report'].reduction_vs_2d_pct:.2f}%")
    print(f"Memory reduction vs 3D: {results['memory_report'].reduction_vs_3d_pct:.2f}%")
    if results['miou'] is not None:
        print(f"mIoU: {results['miou']:.4f}")
    print("="*60)

    if args.dashboard:
        from src.viz.standalone_dashboard import build_dashboard

        dashboard = build_dashboard(
            Path(args.data_dir),
            grid_config_path=Path(args.grid_config) if args.grid_config else None,
            remap_config_path=Path(args.remap_config) if args.remap_config else None,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            max_frames=args.max_frames,
        )
        print("Dashboard: http://localhost:8050")
        dashboard.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
