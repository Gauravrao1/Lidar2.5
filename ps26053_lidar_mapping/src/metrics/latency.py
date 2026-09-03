import time
import platform
import psutil
import numpy as np
import csv
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass, field

@dataclass
class LatencyReport:
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float
    num_frames: int
    warmup_frames: int
    hardware_info: dict
    per_frame_ms: list[float]

def get_hardware_info() -> dict:
    """Detect actual hardware: CPU model via platform.processor(), RAM via psutil,
    GPU via torch.cuda if available. Never hardcode."""
    hw_info = {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": "None"
    }
    try:
        import torch
        if torch.cuda.is_available():
            hw_info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return hw_info

def measure_latency(process_fn: Callable, frames: list, warmup: int = 10) -> LatencyReport:
    """Measure latency of process_fn over frames. Discard warmup frames.
    process_fn takes a single frame and processes it (segmentation + grid insert)."""
    
    per_frame_ms = []
    
    for i, frame in enumerate(frames):
        start = time.perf_counter()
        process_fn(frame)
        end = time.perf_counter()
        
        if i >= warmup:
            per_frame_ms.append((end - start) * 1000.0)
            
    if not per_frame_ms:
        return LatencyReport(0.0, 0.0, 0.0, 0.0, len(frames), warmup, get_hardware_info(), [])
        
    arr = np.array(per_frame_ms)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    
    return LatencyReport(
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        fps=fps,
        num_frames=len(frames) - warmup,
        warmup_frames=warmup,
        hardware_info=get_hardware_info(),
        per_frame_ms=per_frame_ms
    )

def save_latency_csv(report: LatencyReport, output_path: str | Path) -> None:
    """Save per-frame timings + summary to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Mean (ms)", report.mean_ms])
        writer.writerow(["Median (ms)", report.median_ms])
        writer.writerow(["P95 (ms)", report.p95_ms])
        writer.writerow(["FPS", report.fps])
        writer.writerow(["CPU", report.hardware_info.get("cpu", "")])
        writer.writerow(["RAM (GB)", report.hardware_info.get("ram_gb", "")])
        writer.writerow(["GPU", report.hardware_info.get("gpu", "")])
        
        writer.writerow([])
        writer.writerow(["Frame", "Latency (ms)"])
        for i, t in enumerate(report.per_frame_ms):
            writer.writerow([i + 1, t])
