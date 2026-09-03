"""Tests for advanced features: ElevationMap, DynamicObjectTracker, TemporalFusionGrid, PLY export."""
import sys
import os
import numpy as np
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.grid.elevation_map import ElevationMap, ElevationStats
from src.grid.temporal_fusion import TemporalFusionGrid
from src.tracking.object_tracker import DynamicObjectTracker
from src.export.ply_exporter import export_grid_to_ply, export_grid_to_csv, export_point_cloud_to_ply


# ─── ElevationMap ─────────────────────────────────────────────────────
class TestElevationMap:
    def test_flat_ground_high_traversability(self):
        """Flat ground at z=0 → traversability close to 1.0."""
        em = ElevationMap(max_traversable_slope_deg=25.0, max_traversable_roughness=0.3)
        N = 100
        xs = np.random.uniform(0, 5, N)
        ys = np.random.uniform(0, 5, N)
        zs = np.zeros(N)  # perfectly flat
        pts = np.column_stack([xs, ys, zs])
        keys = np.zeros((N, 3), dtype=int)  # all same cell
        em.update_from_points(keys, zs, pts)
        stats = em.get_stats((0, 0, 0))
        assert stats is not None
        assert stats.traversability > 0.8
        assert stats.slope_deg < 5.0

    def test_steep_slope_low_traversability(self):
        """Points on a 60-degree ramp → low traversability."""
        em = ElevationMap(max_traversable_slope_deg=25.0)
        N = 50
        xs = np.linspace(0, 5, N)
        ys = np.zeros(N)
        zs = xs * np.tan(np.radians(60))  # 60-degree slope
        pts = np.column_stack([xs, ys, zs])
        keys = np.zeros((N, 3), dtype=int)
        em.update_from_points(keys, zs, pts)
        stats = em.get_stats((0, 0, 0))
        assert stats is not None
        assert stats.traversability < 0.3
        assert stats.slope_deg > 40

    def test_rough_surface_high_roughness(self):
        """Noisy z values → high roughness."""
        em = ElevationMap()
        N = 100
        xs = np.random.uniform(0, 2, N)
        ys = np.random.uniform(0, 2, N)
        zs = np.random.normal(0, 1.0, N)  # very noisy
        pts = np.column_stack([xs, ys, zs])
        keys = np.zeros((N, 3), dtype=int)
        em.update_from_points(keys, zs, pts)
        stats = em.get_stats((0, 0, 0))
        assert stats is not None
        assert stats.roughness > 0.2

    def test_traversable_cell_count(self):
        """Count cells with traversability > 0.5."""
        em = ElevationMap()
        # Cell A: flat (traversable)
        pts_a = np.column_stack([np.random.uniform(0, 1, 30), np.random.uniform(0, 1, 30), np.zeros(30)])
        keys_a = np.zeros((30, 3), dtype=int)
        em.update_from_points(keys_a, pts_a[:, 2], pts_a)
        # Cell B: steep (not traversable)
        xs = np.linspace(0, 2, 30)
        pts_b = np.column_stack([xs, np.zeros(30), xs * 3.0])
        keys_b = np.ones((30, 3), dtype=int)
        em.update_from_points(keys_b, pts_b[:, 2], pts_b)
        assert em.traversable_cell_count() >= 1
        assert em.mean_traversability() > 0.0

    def test_few_points_handled(self):
        """Cell with < 3 points should still produce stats."""
        em = ElevationMap()
        pts = np.array([[1.0, 2.0, 0.5], [1.1, 2.1, 0.6]])
        keys = np.zeros((2, 3), dtype=int)
        em.update_from_points(keys, pts[:, 2], pts)
        stats = em.get_stats((0, 0, 0))
        assert stats is not None
        assert stats.point_count == 2


# ─── DynamicObjectTracker ─────────────────────────────────────────────
class TestObjectTracker:
    def test_single_cluster_creates_track(self):
        """A tight cluster of points → one track."""
        tracker = DynamicObjectTracker(min_cluster_points=3, cluster_eps=2.0)
        pts = np.array([
            [10.0, 10.0, 0.0], [10.1, 10.1, 0.1],
            [10.2, 10.0, 0.0], [9.9, 10.2, 0.1],
        ])
        tracks = tracker.update(pts, frame_id=0)
        assert len(tracks) >= 1

    def test_velocity_from_motion(self):
        """Object moving 2m between frames → velocity ~2 m/frame."""
        tracker = DynamicObjectTracker(min_cluster_points=1, cluster_eps=2.0, max_association_dist=10.0)
        pts1 = np.array([[10.0, 0.0, 0.0]])
        tracker.update(pts1, frame_id=0)
        pts2 = np.array([[12.0, 0.0, 0.0]])
        tracks = tracker.update(pts2, frame_id=1)
        assert len(tracks) >= 1
        # At least one track should have non-zero velocity
        has_velocity = any(np.linalg.norm(t.velocity) > 0.5 for t in tracks)
        assert has_velocity

    def test_lost_track_removed(self):
        """Track not seen for max_lost_frames gets removed."""
        tracker = DynamicObjectTracker(min_cluster_points=1, cluster_eps=2.0, max_lost_frames=2)
        pts = np.array([[10.0, 0.0, 0.0]])
        tracker.update(pts, frame_id=0)
        assert len(tracker.get_active_tracks()) >= 1
        # 3 empty frames should kill the track
        tracker.update(np.empty((0, 3)), frame_id=1)
        tracker.update(np.empty((0, 3)), frame_id=2)
        tracker.update(np.empty((0, 3)), frame_id=3)
        assert len(tracker.get_active_tracks()) == 0

    def test_two_separate_clusters(self):
        """Two distant clusters → two separate tracks."""
        tracker = DynamicObjectTracker(min_cluster_points=2, cluster_eps=2.0)
        pts = np.array([
            [10.0, 10.0, 0.0], [10.5, 10.5, 0.0],  # cluster A
            [-20.0, -20.0, 0.0], [-20.5, -20.5, 0.0],  # cluster B (far away)
        ])
        tracks = tracker.update(pts, frame_id=0)
        assert len(tracks) >= 2

    def test_summary_stats(self):
        """Summary should report active tracks count."""
        tracker = DynamicObjectTracker(min_cluster_points=1, cluster_eps=2.0)
        pts = np.array([[5.0, 5.0, 0.0], [5.1, 5.1, 0.0]])
        tracker.update(pts, frame_id=0)
        s = tracker.summary()
        assert s["active_tracks"] >= 1
        assert "total_tracks_ever" in s


# ─── TemporalFusionGrid ──────────────────────────────────────────────
class TestTemporalFusion:
    def test_confidence_grows_with_repetition(self):
        """Same class observed 10 times → high confidence."""
        tg = TemporalFusionGrid(num_classes=3)
        key = (5, 10, 2)
        for i in range(10):
            tg.update([key], [1], frame_id=i)
        cell = tg.cells.get(key)
        assert cell is not None
        assert cell.confidence > 0.7
        assert cell.predicted_class == 1

    def test_conflicting_obs_lower_confidence(self):
        """Alternating classes → lower confidence than consistent."""
        tg = TemporalFusionGrid(num_classes=3)
        key = (1, 1, 1)
        for i in range(10):
            cls = 0 if i % 2 == 0 else 1
            tg.update([key], [cls], frame_id=i)
        cell = tg.cells[key]
        # Compare with a consistent cell
        key2 = (2, 2, 2)
        for i in range(10):
            tg.update([key2], [0], frame_id=i)
        cell2 = tg.cells[key2]
        assert cell2.confidence > cell.confidence

    def test_entropy_low_for_certain(self):
        """Consistently observed cell → low entropy."""
        tg = TemporalFusionGrid(num_classes=3)
        key = (3, 3, 3)
        for i in range(20):
            tg.update([key], [2], frame_id=i)
        e = tg.entropy(key)
        assert e < 0.5

    def test_get_confident_cells(self):
        """Confident cells should be retrievable."""
        tg = TemporalFusionGrid(num_classes=3, confidence_threshold=0.6)
        key = (1, 2, 3)
        for i in range(15):
            tg.update([key], [0], frame_id=i)
        confident = tg.get_confident_cells()
        assert len(confident) >= 1
        assert confident[0][1] == 0  # predicted class

    def test_clear_resets(self):
        """Clear empties all cells."""
        tg = TemporalFusionGrid()
        tg.update([(1, 1, 1)], [0], frame_id=0)
        assert len(tg.cells) == 1
        tg.clear()
        assert len(tg.cells) == 0

    def test_overall_confidence(self):
        """Overall confidence should be a valid float."""
        tg = TemporalFusionGrid()
        tg.update([(0, 0, 0), (1, 1, 1)], [0, 1], frame_id=0)
        c = tg.overall_confidence()
        assert 0.0 <= c <= 1.0


# ─── PLY Exporter ────────────────────────────────────────────────────
class TestPLYExporter:
    def test_export_creates_ply(self, tmp_path):
        """Grid export creates a valid PLY file."""
        cells = [
            ((0, 0, 0), 0, (1.0, 2.0, 0.5), 0.05, 5),
            ((1, 1, 0), 2, (5.0, 3.0, 0.0), 0.15, 7),
        ]
        out = tmp_path / "grid.ply"
        n = export_grid_to_ply(cells, str(out), include_size=True)
        assert n == 2
        assert out.exists()
        text = out.read_text()
        assert "ply" in text
        assert "element vertex 2" in text

    def test_export_csv(self, tmp_path):
        """CSV export has correct columns."""
        cells = [((0, 0, 0), 1, (1.0, 2.0, 3.0), 0.5, 1)]
        out = tmp_path / "grid.csv"
        n = export_grid_to_csv(cells, str(out))
        assert n == 1
        text = out.read_text()
        assert "ring" in text
        assert "class" in text

    def test_point_cloud_ply(self, tmp_path):
        """Raw point cloud export produces valid file."""
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        labels = np.array([0, 2])
        out = tmp_path / "cloud.ply"
        n = export_point_cloud_to_ply(pts, labels, str(out))
        assert n == 2
        assert out.exists()
