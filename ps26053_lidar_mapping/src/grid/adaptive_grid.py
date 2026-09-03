from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from src.grid.resolution import GridConfig, point_to_cell_key, cell_size


@dataclass
class CellData:
    class_label: int
    point_count: int
    last_updated_frame: int
    center_x: float
    center_y: float
    center_z: float
    cell_size: float


class AdaptiveGrid:
    def __init__(self, config: GridConfig):
        self.config = config
        # Internal storage: dict mapping (ring, sector, z_bin) -> CellData
        self.cells: Dict[Tuple[int, int, int], CellData] = {}

    def insert(self, points: np.ndarray, labels: np.ndarray, frame_id: int) -> None:
        """Insert classified points into grid. Points: (N,3+) xyz. Labels: (N,) int.
        For each cell, majority-vote the class label."""
        if len(points) == 0:
            return
            
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        ring_indices, sector_indices, z_bins = point_to_cell_key(x, y, z, self.config)
        cell_sizes = cell_size(np.sqrt(np.square(x) + np.square(y)), self.config)
        
        # Dict mapping cell_key -> [labels_dict, sum_x, sum_y, sum_z, sum_s, count]
        cell_updates = {}
        for i in range(len(points)):
            key = (ring_indices[i], sector_indices[i], z_bins[i])
            label = int(labels[i])
            if key not in cell_updates:
                cell_updates[key] = [{label: 1}, x[i], y[i], z[i], cell_sizes[i], 1]
            else:
                data = cell_updates[key]
                data[0][label] = data[0].get(label, 0) + 1
                data[1] += x[i]
                data[2] += y[i]
                data[3] += z[i]
                data[4] += cell_sizes[i]
                data[5] += 1
                
        # Update self.cells
        for key, data in cell_updates.items():
            labels_dict, sum_x, sum_y, sum_z, sum_s, count = data
            
            # Find majority label
            majority_label = max(labels_dict.items(), key=lambda item: item[1])[0]
            
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            avg_s = sum_s / count
            
            if key in self.cells:
                existing_cell = self.cells[key]
                
                old_count = existing_cell.point_count
                total_count = old_count + count
                
                existing_cell.class_label = majority_label
                existing_cell.center_x = (existing_cell.center_x * old_count + sum_x) / total_count
                existing_cell.center_y = (existing_cell.center_y * old_count + sum_y) / total_count
                existing_cell.center_z = (existing_cell.center_z * old_count + sum_z) / total_count
                existing_cell.cell_size = (existing_cell.cell_size * old_count + sum_s) / total_count
                existing_cell.point_count = total_count
                existing_cell.last_updated_frame = frame_id
            else:
                self.cells[key] = CellData(
                    class_label=majority_label,
                    point_count=count,
                    last_updated_frame=frame_id,
                    center_x=avg_x,
                    center_y=avg_y,
                    center_z=avg_z,
                    cell_size=avg_s
                )

    def decay_dynamic_cells(self, current_frame: int) -> int:
        """Remove dynamic-class (label=2) cells not updated in DECAY_MAX_AGE frames.
        Returns count of removed cells. This prevents ghost objects."""
        removed_count = 0
        keys_to_remove = []
        for key, cell in self.cells.items():
            if cell.class_label == 2:
                if (current_frame - cell.last_updated_frame) > self.config.DECAY_MAX_AGE:
                    keys_to_remove.append(key)
                    
        for key in keys_to_remove:
            del self.cells[key]
            removed_count += 1
            
        return removed_count

    def get_cells(self) -> List[Tuple[Tuple[int, int, int], int, Tuple[float, float, float], float, int]]:
        """Return list of (key, class, center_xyz, cell_size, last_frame) for all active cells."""
        result = []
        for key, cell in self.cells.items():
            center_xyz = (cell.center_x, cell.center_y, cell.center_z)
            result.append((key, cell.class_label, center_xyz, cell.cell_size, cell.last_updated_frame))
        return result

    def occupied_cell_count(self) -> int:
        """Return number of non-empty cells."""
        return len(self.cells)

    def memory_bytes(self) -> int:
        """Return estimated memory in bytes = occupied_cell_count * BYTES_PER_CELL."""
        return len(self.cells) * self.config.BYTES_PER_CELL

    def clear(self) -> None:
        """Reset the grid."""
        self.cells.clear()
