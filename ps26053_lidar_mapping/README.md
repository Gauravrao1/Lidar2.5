# PS-26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping

## DRDO Problem Statement 26053
> Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

A complete system that takes raw LiDAR point clouds, performs semantic segmentation into
**Terrain / Static / Dynamic** classes, builds an adaptive variable-resolution 2.5D polar grid,
and provides a real-time browser-based dashboard with live memory-reduction statistics.

---

## Quick Start

### 1. Setup (one-time)

```powershell
# Clone / navigate to project
cd "d:\Lidar 2.5\ps26053_lidar_mapping"

# Create virtual environment (Python 3.13)
py -3.13 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy pyyaml psutil scipy pytest pandas matplotlib plotly dash
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Generate sample data (10 synthetic LiDAR scans)
python scripts\generate_sample_data.py
```

### 2. Run Tests

```powershell
python -m pytest tests/ -v
```

### 3. Run Full Pipeline

```powershell
python -m src.pipeline.standalone_runner --data-dir data\sample --max-frames 10
```

### 4. Launch Dashboard

```powershell
python -m src.viz.standalone_dashboard --data-dir data\sample
# Open http://localhost:8050
```

### 5. One-Command Demo

```powershell
.\scripts\run_full_demo.ps1
```

---

## Component Status

| Component | Status | Notes |
|---|---|---|
| **Grid Engine** | ✅ Working | Log-linear variable resolution, ghost-object decay, all tests pass |
| **Segmentation (PointNet)** | ✅ Working | Pure PyTorch, CPU-only, ~10-50ms/frame |
| **Segmentation (RandLA-Net)** | 📄 Documented | Requires Python ≤3.12 + Open3D-ML (not available in current env) |
| **Data Loader** | ✅ Working | Reads SemanticKITTI binary format, 28-class remap validated |
| **Metrics (IoU)** | ✅ Working | Per-class IoU, mIoU, distance-bucketed IoU |
| **Metrics (Latency)** | ✅ Working | Mean/median/p95/FPS with hardware detection |
| **Metrics (Memory)** | ✅ Working | Live comparison vs uniform 2D and 3D baselines |
| **Dashboard** | ✅ Working | Plotly Dash, 3-color cells, variable-size markers, live stats |
| **ROS2 Nodes** | 📄 Documented | Source provided, cannot build on Windows |
| **Sample Data** | ✅ Working | 10 synthetic scans, 26k points each |

---

## Architecture

```
SemanticKITTI .bin/.label files
        │
        ▼
┌─────────────────┐
│  kitti_loader.py │── reads binary, applies class_remap.yaml (28→4 classes)
└────────┬────────┘
         ▼
┌─────────────────────┐
│  PointNet Backend    │── pure PyTorch, Conv1d architecture, CPU-only
│  model_interface.py  │── abstract base class (swap RandLA-Net if available)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Grid Engine         │── resolution.py: cell_size(r) = log-linear interpolation
│  adaptive_grid.py    │── polar hash grid, majority voting, staleness decay
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Dashboard + Metrics │── Plotly Dash: 3D scatter, live memory stats
│  report_generator.py │── IoU, latency, memory → markdown report
└─────────────────────┘
```

---

## Variable Resolution Formula

Cell size at distance `r` from the sensor:

```
s(r) = S_NEAR                                          if r ≤ R_NEAR (10m)
s(r) = S_NEAR × (S_FAR/S_NEAR)^(ln(r/R_NEAR)/ln(R_FAR/R_NEAR))  if R_NEAR < r < R_FAR
s(r) = S_FAR                                           if r ≥ R_FAR (100m)
```

- **S_NEAR = 0.05m (5cm)** — finest resolution within 10m
- **S_FAR = 0.50m (50cm)** — coarsest resolution at/beyond 100m
- **Log-linear interpolation** between boundaries (continuous, monotonic)

---

## Environment (Actual)

| Item | Value |
|---|---|
| OS | Windows 11 Home, Build 26200 |
| CPU | 11th Gen Intel i3-1115G4 @ 3.00GHz |
| RAM | 8 GB |
| GPU | None (Intel UHD integrated) |
| Python | 3.13.11 |
| PyTorch | CPU-only |
| ROS2 | Not available |

---

## Known Limitations

1. **Segmentation quality**: PointNet trained on 10 synthetic scans has limited accuracy.
   Using the RandLA-Net backend with a pretrained SemanticKITTI checkpoint (requires Python ≤3.12)
   would give ~60% mIoU.

2. **Sample data only**: All reported metrics are from synthetic sample data (10 frames, 26k points each).
   Numbers are clearly labeled as "SAMPLE RUN". Use `download_semantickitti.sh` for real data.

3. **CPU-only inference**: No GPU available. PointNet runs at ~10-50ms/frame on CPU.
   With GPU, latency would be ~1-5ms/frame.

4. **ROS2 path untested**: ROS2 node source code is provided but cannot be built/tested on Windows.
   Documented with exact `colcon build` and `ros2 run` commands for Ubuntu.

5. **8GB RAM constraint**: Point clouds are subsampled to 16,384 points for segmentation.
   Full 100k+ point scans may require more memory.

---

## How to Reproduce Every Reported Number

1. Generate sample data: `python scripts\generate_sample_data.py`
2. Run pipeline: `python -m src.pipeline.standalone_runner --data-dir data\sample`
3. Check outputs in `reports/`:
   - `latency.csv` — per-frame timing
   - `per_frame_log.csv` — cells, memory per frame
   - `metrics_table_filled.md` — complete metrics table

---

## License

This project is for DRDO PS-26053 evaluation purposes.
Dataset (SemanticKITTI): CC BY-NC-SA 4.0.
