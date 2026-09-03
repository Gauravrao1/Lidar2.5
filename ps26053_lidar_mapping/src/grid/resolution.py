import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union
import numpy as np
import yaml

@dataclass
class GridConfig:
    R_NEAR: float = 10.0
    R_FAR: float = 100.0
    S_NEAR: float = 0.05
    S_FAR: float = 0.50
    N_SECTORS: int = 360
    Z_MIN: float = -3.0
    Z_MAX: float = 3.0
    Z_BIN_SIZE: float = 0.25
    REFRESH_N: int = 5
    DECAY_MAX_AGE: int = 10
    BEYOND_FAR_POLICY: str = "clip"
    BYTES_PER_CELL: int = 48

def load_grid_config(config_path: str) -> GridConfig:
    """Loads YAML config, returns GridConfig with all parameters."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    if data is None:
        return GridConfig()
        
    kwargs = {}
    for k in GridConfig.__dataclass_fields__:
        k_lower = k.lower()
        if k in data:
            kwargs[k] = data[k]
        elif k_lower in data:
            kwargs[k] = data[k_lower]
        # Also check nested dicts, e.g., 'grid'
        elif 'grid' in data and k in data['grid']:
            kwargs[k] = data['grid'][k]
        elif 'grid' in data and k_lower in data['grid']:
            kwargs[k] = data['grid'][k_lower]
            
    return GridConfig(**kwargs)

def cell_size(r: Union[float, np.ndarray], config: GridConfig) -> Union[float, np.ndarray]:
    """Returns cell size at distance r."""
    epsilon = 1e-6
    if isinstance(r, (float, int)):
        r = max(float(r), epsilon)
        if r <= config.R_NEAR:
            return config.S_NEAR
        elif r >= config.R_FAR:
            return config.S_FAR
        else:
            return config.S_NEAR * (config.S_FAR / config.S_NEAR)**(
                math.log(r / config.R_NEAR) / math.log(config.R_FAR / config.R_NEAR)
            )
    else:
        r_arr = np.maximum(r, epsilon)
        sizes = np.empty_like(r_arr, dtype=float)
        
        mask_near = r_arr <= config.R_NEAR
        mask_far = r_arr >= config.R_FAR
        mask_mid = ~(mask_near | mask_far)
        
        sizes[mask_near] = config.S_NEAR
        sizes[mask_far] = config.S_FAR
        if np.any(mask_mid):
            sizes[mask_mid] = config.S_NEAR * (config.S_FAR / config.S_NEAR)**(
                np.log(r_arr[mask_mid] / config.R_NEAR) / np.log(config.R_FAR / config.R_NEAR)
            )
        return sizes

def point_to_cell_key(x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray], config: GridConfig) -> Union[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Converts Cartesian to polar, computes ring_index and sector_index and z_bin."""
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.arctan2(y, x)
    
    cell_s = cell_size(r, config)
    
    if isinstance(r, np.ndarray):
        ring_index = np.floor(r / cell_s).astype(int)
        
        sector_angle = 2 * np.pi / config.N_SECTORS
        sector_index = np.floor((theta + np.pi) / sector_angle).astype(int) % config.N_SECTORS
        
        z_bin = np.floor((z - config.Z_MIN) / config.Z_BIN_SIZE).astype(int)
        max_z_bin = max(0, int(np.ceil((config.Z_MAX - config.Z_MIN) / config.Z_BIN_SIZE)) - 1)
        z_bin = np.clip(z_bin, 0, max_z_bin)
        
        return ring_index, sector_index, z_bin
    else:
        ring_index = int(math.floor(r / cell_s))
        
        sector_angle = 2 * math.pi / config.N_SECTORS
        sector_index = int(math.floor((theta + math.pi) / sector_angle)) % config.N_SECTORS
        
        z_bin = int(math.floor((z - config.Z_MIN) / config.Z_BIN_SIZE))
        max_z_bin = max(0, int(math.ceil((config.Z_MAX - config.Z_MIN) / config.Z_BIN_SIZE)) - 1)
        z_bin = max(0, min(z_bin, max_z_bin))
        
        return ring_index, sector_index, z_bin
