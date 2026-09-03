import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any
import csv

def export_grid_to_ply(cells: list, output_path: str, include_size: bool = True) -> int:
    """Export grid cells to PLY point cloud format.
    Each cell becomes a point at its centroid, colored by class.
    Colors: Terrain=(0,200,0), Static=(0,100,255), Dynamic=(255,50,50), Ignore=(128,128,128)
    If include_size, adds a scalar 'cell_size' property.
    Returns number of points written.
    
    cells format: list of (key, class_label, (cx,cy,cz), cell_size, last_frame)
    """
    colors = {
        0: (128, 128, 128), # Ignore
        1: (0, 200, 0),     # Terrain
        2: (0, 100, 255),   # Static
        3: (255, 50, 50)    # Dynamic
    }
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cells)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if include_size:
            f.write("property float cell_size\n")
        f.write("end_header\n")
        
        for cell in cells:
            key, class_label, (cx, cy, cz), cell_size, last_frame = cell
            r, g, b = colors.get(class_label, (255, 255, 255))
            if include_size:
                f.write(f"{cx} {cy} {cz} {r} {g} {b} {cell_size}\n")
            else:
                f.write(f"{cx} {cy} {cz} {r} {g} {b}\n")
                
    return len(cells)

def export_grid_to_csv(cells: list, output_path: str) -> int:
    """Export grid cells to CSV format with columns:
    ring,sector,z_bin,class,class_name,x,y,z,cell_size,last_frame
    Returns number of rows written."""
    class_names = {
        0: "Ignore",
        1: "Terrain",
        2: "Static",
        3: "Dynamic"
    }
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ring", "sector", "z_bin", "class", "class_name", "x", "y", "z", "cell_size", "last_frame"])
        for cell in cells:
            (ring, sector, z_bin), class_label, (cx, cy, cz), cell_size, last_frame = cell
            class_name = class_names.get(class_label, "Unknown")
            writer.writerow([ring, sector, z_bin, class_label, class_name, cx, cy, cz, cell_size, last_frame])
            
    return len(cells)

def export_point_cloud_to_ply(points: np.ndarray, labels: np.ndarray, output_path: str) -> int:
    """Export raw point cloud with labels to PLY.
    points: (N,3) or (N,4) float array
    labels: (N,) int array
    Returns number of points written."""
    colors = {
        0: (128, 128, 128), # Ignore
        1: (0, 200, 0),     # Terrain
        2: (0, 100, 255),   # Static
        3: (255, 50, 50)    # Dynamic
    }
    
    num_points = points.shape[0]
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(num_points):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            label = int(labels[i]) if i < len(labels) else 0
            r, g, b = colors.get(label, (255, 255, 255))
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
            
    return num_points
