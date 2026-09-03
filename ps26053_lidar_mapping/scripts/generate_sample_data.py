"""
Generate realistic LiDAR simulation data — 50 frames with a driving car trajectory.

Creates synthetic point clouds that simulate:
  - Ego vehicle driving forward along a road
  - Static environment (buildings, trees, poles along the road)
  - Dynamic objects: oncoming car, pedestrian crossing, parked cars
  - Realistic point density and noise
"""
import numpy as np
from pathlib import Path
import struct
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sample"
VELODYNE_DIR = DATA_DIR / "velodyne"
LABEL_DIR = DATA_DIR / "labels"

NUM_FRAMES = 50
POINTS_PER_FRAME = 40000
RNG = np.random.default_rng(42)

# SemanticKITTI label IDs
LABEL_ROAD = 40
LABEL_SIDEWALK = 48
LABEL_BUILDING = 50
LABEL_VEGETATION = 70
LABEL_TRUNK = 71
LABEL_POLE = 80
LABEL_CAR = 10
LABEL_MOVING_CAR = 252
LABEL_PERSON = 30
LABEL_MOVING_PERSON = 254
LABEL_TERRAIN = 72


def make_ground_plane(ego_x, ego_y, n_pts=12000):
    """Generate road + sidewalk + terrain ground plane around ego position."""
    pts = []
    labels = []
    # Road: 8m wide strip along X axis
    road_x = RNG.uniform(ego_x - 30, ego_x + 60, n_pts // 2)
    road_y = RNG.uniform(-4, 4, n_pts // 2)
    road_z = RNG.normal(-1.7, 0.02, n_pts // 2)  # sensor height ~1.7m
    road_i = RNG.uniform(0.1, 0.4, n_pts // 2)
    pts.append(np.column_stack([road_x, road_y, road_z, road_i]))
    labels.extend([LABEL_ROAD] * (n_pts // 2))

    # Sidewalk: strips at y=[-6,-4] and y=[4,6]
    n_side = n_pts // 6
    for y_range in [(-7, -4), (4, 7)]:
        sx = RNG.uniform(ego_x - 25, ego_x + 50, n_side)
        sy = RNG.uniform(*y_range, n_side)
        sz = RNG.normal(-1.65, 0.03, n_side)
        si = RNG.uniform(0.15, 0.35, n_side)
        pts.append(np.column_stack([sx, sy, sz, si]))
        labels.extend([LABEL_SIDEWALK] * n_side)

    # Terrain: patches beyond sidewalk
    n_ter = n_pts // 6
    for y_range in [(-15, -7), (7, 15)]:
        tx = RNG.uniform(ego_x - 20, ego_x + 40, n_ter // 2)
        ty = RNG.uniform(*y_range, n_ter // 2)
        tz = RNG.normal(-1.5, 0.1, n_ter // 2)
        ti = RNG.uniform(0.05, 0.2, n_ter // 2)
        pts.append(np.column_stack([tx, ty, tz, ti]))
        labels.extend([LABEL_TERRAIN] * (n_ter // 2))

    return np.vstack(pts), np.array(labels, dtype=np.uint32)


def make_buildings(ego_x, ego_y, n_per_side=3):
    """Generate buildings along both sides of the road."""
    pts = []
    labels = []
    for side in [-1, 1]:
        for i in range(n_per_side):
            bx = ego_x - 10 + i * 25 + RNG.uniform(-3, 3)
            by = side * RNG.uniform(10, 18)
            bw = RNG.uniform(6, 12)
            bd = RNG.uniform(6, 10)
            bh = RNG.uniform(4, 10)
            n_pts = RNG.integers(300, 600)
            # Front face
            fx = np.full(n_pts, bx)
            fy = RNG.uniform(by, by + bw, n_pts)
            fz = RNG.uniform(-1.7, bh - 1.7, n_pts)
            fi = RNG.uniform(0.2, 0.6, n_pts)
            pts.append(np.column_stack([fx, fy, fz, fi]))
            # Side face
            sx = RNG.uniform(bx, bx + bd, n_pts // 2)
            sy = np.full(n_pts // 2, by)
            sz = RNG.uniform(-1.7, bh - 1.7, n_pts // 2)
            si = RNG.uniform(0.2, 0.5, n_pts // 2)
            pts.append(np.column_stack([sx, sy, sz, si]))
            labels.extend([LABEL_BUILDING] * (n_pts + n_pts // 2))
    return np.vstack(pts), np.array(labels, dtype=np.uint32)


def make_vegetation(ego_x, ego_y, n_trees=8):
    """Generate trees and bushes."""
    pts = []
    labels = []
    for _ in range(n_trees):
        tx = ego_x + RNG.uniform(-15, 40)
        ty = RNG.choice([-1, 1]) * RNG.uniform(8, 20)
        # Trunk
        n_trunk = RNG.integers(30, 60)
        trunk_x = RNG.normal(tx, 0.1, n_trunk)
        trunk_y = RNG.normal(ty, 0.1, n_trunk)
        trunk_z = RNG.uniform(-1.7, 2, n_trunk)
        trunk_i = RNG.uniform(0.1, 0.3, n_trunk)
        pts.append(np.column_stack([trunk_x, trunk_y, trunk_z, trunk_i]))
        labels.extend([LABEL_TRUNK] * n_trunk)
        # Canopy
        n_canopy = RNG.integers(80, 200)
        can_x = RNG.normal(tx, 1.5, n_canopy)
        can_y = RNG.normal(ty, 1.5, n_canopy)
        can_z = RNG.uniform(1, 5, n_canopy)
        can_i = RNG.uniform(0.3, 0.7, n_canopy)
        pts.append(np.column_stack([can_x, can_y, can_z, can_i]))
        labels.extend([LABEL_VEGETATION] * n_canopy)
    return np.vstack(pts), np.array(labels, dtype=np.uint32)


def make_poles(ego_x, ego_y, n_poles=6):
    """Generate street poles."""
    pts = []
    labels = []
    for i in range(n_poles):
        px = ego_x - 5 + i * 12
        py = RNG.choice([-5.5, 5.5])
        n_pts = RNG.integers(20, 50)
        pole_x = RNG.normal(px, 0.05, n_pts)
        pole_y = RNG.normal(py, 0.05, n_pts)
        pole_z = RNG.uniform(-1.7, 3, n_pts)
        pole_i = RNG.uniform(0.4, 0.8, n_pts)
        pts.append(np.column_stack([pole_x, pole_y, pole_z, pole_i]))
        labels.extend([LABEL_POLE] * n_pts)
    return np.vstack(pts), np.array(labels, dtype=np.uint32)


def make_parked_cars(ego_x, ego_y, n_cars=4):
    """Generate parked cars along the road edge."""
    pts = []
    labels = []
    for i in range(n_cars):
        cx = ego_x + 5 + i * 15 + RNG.uniform(-2, 2)
        cy = RNG.choice([-3.5, 3.5])
        n_pts = RNG.integers(100, 250)
        car_x = RNG.uniform(cx - 2.2, cx + 2.2, n_pts)
        car_y = RNG.uniform(cy - 0.9, cy + 0.9, n_pts)
        car_z = RNG.uniform(-1.7, -0.2, n_pts)
        car_i = RNG.uniform(0.3, 0.7, n_pts)
        pts.append(np.column_stack([car_x, car_y, car_z, car_i]))
        labels.extend([LABEL_CAR] * n_pts)
    return np.vstack(pts), np.array(labels, dtype=np.uint32)


def make_oncoming_car(ego_x, ego_y, frame_id, n_pts=200):
    """An oncoming car driving toward ego at ~10 m/s in the opposite lane."""
    start_x = ego_x + 80
    car_x = start_x - frame_id * 3.0  # moving toward ego
    car_y = -2.0  # opposite lane
    if car_x < ego_x - 20:
        return np.empty((0, 4)), np.array([], dtype=np.uint32)
    pts_x = RNG.uniform(car_x - 2.2, car_x + 2.2, n_pts)
    pts_y = RNG.uniform(car_y - 0.9, car_y + 0.9, n_pts)
    pts_z = RNG.uniform(-1.7, -0.2, n_pts)
    pts_i = RNG.uniform(0.4, 0.8, n_pts)
    labels = np.full(n_pts, LABEL_MOVING_CAR, dtype=np.uint32)
    return np.column_stack([pts_x, pts_y, pts_z, pts_i]), labels


def make_pedestrian(ego_x, ego_y, frame_id, n_pts=60):
    """A pedestrian crossing the road at frame 20-35."""
    if frame_id < 20 or frame_id > 35:
        return np.empty((0, 4)), np.array([], dtype=np.uint32)
    progress = (frame_id - 20) / 15.0
    ped_x = ego_x + 15
    ped_y = -6 + progress * 12  # crossing from left to right
    pts_x = RNG.normal(ped_x, 0.15, n_pts)
    pts_y = RNG.normal(ped_y, 0.15, n_pts)
    pts_z = RNG.uniform(-1.7, 0.0, n_pts)
    pts_i = RNG.uniform(0.2, 0.5, n_pts)
    labels = np.full(n_pts, LABEL_MOVING_PERSON, dtype=np.uint32)
    return np.column_stack([pts_x, pts_y, pts_z, pts_i]), labels


def generate_frame(frame_id):
    """Generate one complete frame with ego vehicle at position along trajectory."""
    # Ego drives forward at ~2 m/frame
    ego_x = frame_id * 2.0
    ego_y = 0.0

    all_pts = []
    all_labels = []

    # Static scene (relative to ego)
    for gen in [make_ground_plane, make_buildings, make_vegetation, make_poles, make_parked_cars]:
        pts, labels = gen(ego_x, ego_y)
        all_pts.append(pts)
        all_labels.append(labels)

    # Dynamic objects
    for gen in [make_oncoming_car, make_pedestrian]:
        pts, labels = gen(ego_x, ego_y, frame_id)
        if len(pts) > 0:
            all_pts.append(pts)
            all_labels.append(labels)

    all_pts = np.vstack(all_pts).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.uint32)

    # Transform to ego-centric coordinates (subtract ego position)
    all_pts[:, 0] -= ego_x
    all_pts[:, 1] -= ego_y

    # Add sensor noise
    all_pts[:, :3] += RNG.normal(0, 0.01, all_pts[:, :3].shape).astype(np.float32)

    # Distance-based subsampling (keep more nearby points)
    dist = np.sqrt(all_pts[:, 0]**2 + all_pts[:, 1]**2)
    keep_prob = np.clip(1.0 - dist / 100, 0.3, 1.0)
    mask = RNG.random(len(all_pts)) < keep_prob
    all_pts = all_pts[mask]
    all_labels = all_labels[mask]

    # Cap total points
    if len(all_pts) > POINTS_PER_FRAME:
        idx = RNG.choice(len(all_pts), POINTS_PER_FRAME, replace=False)
        all_pts = all_pts[idx]
        all_labels = all_labels[idx]

    return all_pts, all_labels


def main():
    VELODYNE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    total_points = 0
    for frame_id in range(NUM_FRAMES):
        pts, labels = generate_frame(frame_id)
        total_points += len(pts)

        # Write .bin (float32 x4)
        bin_path = VELODYNE_DIR / f"{frame_id:06d}.bin"
        pts.tofile(str(bin_path))

        # Write .label (uint32)
        label_path = LABEL_DIR / f"{frame_id:06d}.label"
        labels.tofile(str(label_path))

        print(f"Frame {frame_id:03d}: {len(pts):,} points | "
              f"ego=({frame_id*2:.0f}, 0) | "
              f"dynamic={'car+ped' if 20 <= frame_id <= 35 else 'car' if frame_id < 30 else 'none'}")

    print(f"\nGenerated {NUM_FRAMES} frames, {total_points:,} total points")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
