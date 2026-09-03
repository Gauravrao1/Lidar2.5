import pandas as pd
from pathlib import Path
from typing import Optional
from .memory import MemoryReport

def generate_metrics_table(latency_csv: str | Path, memory_report: MemoryReport,
                            iou_results: dict, hardware_info: dict,
                            output_path: str | Path,
                            is_sample_run: bool = True) -> None:
    """Write reports/metrics_table_filled.md with all cells filled from measured data.
    RAISES ValueError if any required metric is None or missing.
    If is_sample_run=True, adds prominent 'SAMPLE RUN' banner."""
    
    if memory_report is None:
        raise ValueError("memory_report is missing")
    if not iou_results:
        raise ValueError("iou_results is missing")
    if not hardware_info:
        raise ValueError("hardware_info is missing")
        
    try:
        lat_df = pd.read_csv(latency_csv)
    except Exception as e:
        raise ValueError(f"Failed to read latency CSV: {e}")
        
    mean_ms, fps = None, None
    for _, row in lat_df.iterrows():
        if row['Metric'] == 'Mean (ms)':
            mean_ms = float(row['Value'])
        elif row['Metric'] == 'FPS':
            fps = float(row['Value'])
            
    if mean_ms is None or fps is None:
        raise ValueError("Missing latency metrics in CSV")
        
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, 'w') as f:
        if is_sample_run:
            f.write("# *** SAMPLE RUN ***\n\n")
            
        f.write("# Metrics Report\n\n")
        f.write("## Hardware\n")
        f.write(f"- CPU: {hardware_info.get('cpu', 'N/A')}\n")
        f.write(f"- RAM: {hardware_info.get('ram_gb', 'N/A')} GB\n")
        f.write(f"- GPU: {hardware_info.get('gpu', 'N/A')}\n\n")
        
        f.write("## Performance\n")
        f.write(f"- Mean Latency: {mean_ms:.2f} ms\n")
        f.write(f"- FPS: {fps:.2f}\n\n")
        
        f.write("## Memory\n")
        f.write(f"- Adaptive Cells: {memory_report.adaptive_cells}\n")
        f.write(f"- Adaptive Memory: {memory_report.adaptive_bytes / 1024 / 1024:.2f} MB\n")
        f.write(f"- Reduction vs 2D: {memory_report.reduction_vs_2d_pct:.2f}%\n")
        f.write(f"- Reduction vs 3D: {memory_report.reduction_vs_3d_pct:.2f}%\n\n")
        
        f.write("## Segmentation IoU\n")
        for bucket, metrics in iou_results.items():
            f.write(f"### Range: {bucket}\n")
            f.write(f"- mIoU: {metrics['miou']:.4f}\n")
            f.write(f"- Points: {metrics['num_points']}\n")
