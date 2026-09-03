import math
import psutil
import os
from dataclasses import dataclass

@dataclass
class MemoryReport:
    adaptive_cells: int
    adaptive_bytes: int
    uniform_2d_cells: int
    uniform_2d_bytes: int
    uniform_3d_cells: int  
    uniform_3d_bytes: int
    reduction_vs_2d_pct: float
    reduction_vs_3d_pct: float
    process_rss_bytes: int

def uniform_grid_cells(area_m2: float, cell_size_m: float) -> int:
    """Number of cells in a uniform 2D grid covering area_m2 at cell_size_m (use S_NEAR=0.05m).
    area_m2 for a circle of R_FAR=100m = pi * 100^2."""
    return int(math.ceil(area_m2 / (cell_size_m ** 2)))

def uniform_grid_3d_cells(area_m2: float, z_range_m: float, cell_size_m: float) -> int:
    """Number of cells in a uniform 3D voxel grid.
    = uniform_grid_cells * (z_range / cell_size)."""
    return int(math.ceil(uniform_grid_cells(area_m2, cell_size_m) * (z_range_m / cell_size_m)))

def compute_memory_report(adaptive_cells: int, adaptive_bytes_per_cell: int = 48,
                           r_far: float = 100.0, z_range: float = 6.0,
                           s_near: float = 0.05) -> MemoryReport:
    """Compute full memory comparison report.
    Uniform 2D baseline: pi*r_far^2 / s_near^2 cells
    Uniform 3D baseline: pi*r_far^2 * z_range / s_near^3 cells
    Reduction = (1 - adaptive/baseline) * 100"""
    area_m2 = math.pi * (r_far ** 2)
    
    u2d_cells = uniform_grid_cells(area_m2, s_near)
    u3d_cells = uniform_grid_3d_cells(area_m2, z_range, s_near)
    
    u2d_bytes = u2d_cells * adaptive_bytes_per_cell
    u3d_bytes = u3d_cells * adaptive_bytes_per_cell
    
    adaptive_bytes = adaptive_cells * adaptive_bytes_per_cell
    
    red_2d = max(0.0, (1.0 - adaptive_cells / u2d_cells) * 100.0) if u2d_cells > 0 else 0.0
    red_3d = max(0.0, (1.0 - adaptive_cells / u3d_cells) * 100.0) if u3d_cells > 0 else 0.0
    
    process = psutil.Process(os.getpid())
    process_rss_bytes = process.memory_info().rss
    
    return MemoryReport(
        adaptive_cells=adaptive_cells,
        adaptive_bytes=adaptive_bytes,
        uniform_2d_cells=u2d_cells,
        uniform_2d_bytes=u2d_bytes,
        uniform_3d_cells=u3d_cells,
        uniform_3d_bytes=u3d_bytes,
        reduction_vs_2d_pct=red_2d,
        reduction_vs_3d_pct=red_3d,
        process_rss_bytes=process_rss_bytes
    )
