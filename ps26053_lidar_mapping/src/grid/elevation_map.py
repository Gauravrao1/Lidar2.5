from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

@dataclass
class ElevationStats:
    """Per-cell elevation statistics."""
    z_min: float
    z_max: float
    z_mean: float
    z_std: float
    point_count: int
    slope_deg: float  # estimated slope angle in degrees
    roughness: float  # std of z residuals after plane fit
    traversability: float  # 0.0 (impassable) to 1.0 (perfectly flat)

class ElevationMap:
    """Builds per-cell elevation stats from the adaptive grid's cells."""
    
    def __init__(self, max_traversable_slope_deg: float = 25.0, max_traversable_roughness: float = 0.3):
        self.max_slope = max_traversable_slope_deg
        self.max_roughness = max_traversable_roughness
        self.cells: Dict[Tuple[int,int,int], ElevationStats] = {}
    
    def update_from_points(self, cell_keys: np.ndarray, z_values: np.ndarray, points_xyz: np.ndarray) -> None:
        """Given arrays of cell keys and z values, compute elevation stats per cell.
        cell_keys: (N,3) int array of (ring, sector, z_bin)
        z_values: (N,) float array
        points_xyz: (N,3) float array for slope estimation
        Groups points by cell key, then computes min/max/mean/std/slope/roughness/traversability."""
        cell_points = defaultdict(list)
        
        for i in range(len(cell_keys)):
            key = tuple(cell_keys[i])
            cell_points[key].append(points_xyz[i])
            
        for key, pts_list in cell_points.items():
            pts = np.array(pts_list)
            z_vals = pts[:, 2]
            point_count = len(pts)
            
            z_min = float(np.min(z_vals))
            z_max = float(np.max(z_vals))
            z_mean = float(np.mean(z_vals))
            z_std = float(np.std(z_vals)) if point_count > 1 else 0.0
            
            slope_deg = 0.0
            roughness = 0.0
            
            if point_count >= 3:
                # Plane fitting: z = a*x + b*y + c
                # A = [x, y, 1]
                xy = pts[:, :2]
                A = np.c_[xy, np.ones(point_count)]
                b = z_vals
                
                # lstsq returns: x, residuals, rank, s
                coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                a, b_coef, c = coef
                
                # Gradient magnitude
                grad_mag = np.sqrt(a**2 + b_coef**2)
                slope_deg = float(np.degrees(np.arctan(grad_mag)))
                
                # Roughness
                fitted_z = A.dot(coef)
                residuals = z_vals - fitted_z
                roughness = float(np.std(residuals))
            
            # Traversability calculation
            slope_factor = max(0.0, 1.0 - slope_deg / self.max_slope)
            roughness_factor = max(0.0, 1.0 - roughness / self.max_roughness)
            traversability = slope_factor * roughness_factor
            
            self.cells[key] = ElevationStats(
                z_min=z_min,
                z_max=z_max,
                z_mean=z_mean,
                z_std=z_std,
                point_count=point_count,
                slope_deg=slope_deg,
                roughness=roughness,
                traversability=traversability
            )

    def get_traversability_grid(self) -> List[Tuple[Tuple[int,int,int], float, float, float, float]]:
        """Return list of (key, z_mean, slope_deg, roughness, traversability) for all cells."""
        return [
            (key, stats.z_mean, stats.slope_deg, stats.roughness, stats.traversability)
            for key, stats in self.cells.items()
        ]
    
    def get_stats(self, key: Tuple[int,int,int]) -> Optional[ElevationStats]:
        """Get elevation stats for a specific cell."""
        return self.cells.get(key)
    
    def traversable_cell_count(self) -> int:
        """Count cells with traversability > 0.5."""
        return sum(1 for stats in self.cells.values() if stats.traversability > 0.5)
    
    def mean_traversability(self) -> float:
        """Average traversability across all cells."""
        if not self.cells:
            return 0.0
        return sum(stats.traversability for stats in self.cells.values()) / len(self.cells)
