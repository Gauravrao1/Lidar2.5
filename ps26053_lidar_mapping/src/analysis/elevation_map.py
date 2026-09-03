from collections import defaultdict

import numpy as np


class ElevationMap:
    """Grid-size based compatibility adapter for elevation analysis."""

    def __init__(self, grid_size=1.0, traversability_threshold=0.5):
        self.grid_size = float(grid_size)
        self.traversability_threshold = float(traversability_threshold)
        self.cells = {}

    def update(self, points):
        points = np.asarray(points)
        buckets = defaultdict(list)
        for point in points:
            key = (int(np.floor(point[0] / self.grid_size)),
                   int(np.floor(point[1] / self.grid_size)))
            buckets[key].append(point[:3])

        global_slope = 0.0
        if len(points) >= 3:
            design = np.c_[points[:, :2], np.ones(len(points))]
            coefficients, _, _, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
            global_slope = float(np.degrees(np.arctan(np.linalg.norm(coefficients[:2]))))

        for key, values in buckets.items():
            cell_points = np.asarray(values, dtype=float)
            z_values = cell_points[:, 2]
            slope_deg = global_slope if len(cell_points) < 3 else 0.0
            roughness = 0.0
            if len(cell_points) >= 3:
                design = np.c_[cell_points[:, :2], np.ones(len(cell_points))]
                coefficients, _, _, _ = np.linalg.lstsq(design, z_values, rcond=None)
                slope_deg = float(np.degrees(np.arctan(np.linalg.norm(coefficients[:2]))))
                roughness = float(np.std(z_values - design @ coefficients))
            traversability = max(0.0, 1.0 - slope_deg / 25.0) * max(0.0, 1.0 - roughness / 0.3)
            self.cells[key] = {
                "z_min": float(np.min(z_values)),
                "z_max": float(np.max(z_values)),
                "z_mean": float(np.mean(z_values)),
                "slope": slope_deg,
                "roughness": roughness,
                "traversability": traversability,
            }

    def get_cell_info(self, x, y):
        key = (int(np.floor(x / self.grid_size)), int(np.floor(y / self.grid_size)))
        return self.cells.get(key)

    def get_traversable_cells(self):
        return [key for key, info in self.cells.items()
                if info["traversability"] >= self.traversability_threshold]
