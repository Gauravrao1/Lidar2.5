"""Tests for src.grid.adaptive_grid — insert, decay, memory."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.grid.resolution import GridConfig
from src.grid.adaptive_grid import AdaptiveGrid, CellData


@pytest.fixture
def config():
    return GridConfig()


@pytest.fixture
def grid(config):
    return AdaptiveGrid(config)


class TestInsert:
    """Test point insertion into the adaptive grid."""

    def test_insert_single_point(self, grid):
        """Inserting a single point should create one cell."""
        points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([0], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        assert grid.occupied_cell_count() == 1

    def test_insert_empty(self, grid):
        """Inserting empty arrays should not crash or create cells."""
        points = np.zeros((0, 3), dtype=np.float32)
        labels = np.zeros(0, dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        assert grid.occupied_cell_count() == 0

    def test_insert_same_cell_majority_vote(self, grid):
        """Multiple points in the same cell should use majority voting for the label."""
        # All at roughly the same location (near origin, close together)
        points = np.array([
            [1.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [1.02, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [1.0, 0.02, 0.0],
        ], dtype=np.float32)
        # 3 terrain, 2 static -> majority should be terrain (0)
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        cells = grid.get_cells()
        # Find the cell(s) and check majority
        class_labels = [c[1] for c in cells]
        # At least one cell should have label 0
        assert 0 in class_labels

    def test_insert_multiple_cells(self, grid):
        """Points at different distances should land in different cells."""
        points = np.array([
            [1.0, 0.0, 0.0],   # near
            [50.0, 0.0, 0.0],  # far
        ], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        assert grid.occupied_cell_count() == 2

    def test_insert_updates_frame(self, grid):
        """Inserting into the same cell at a later frame should update last_updated_frame."""
        points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([2], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        grid.insert(points, labels, frame_id=5)
        
        cells = grid.get_cells()
        assert len(cells) == 1
        _, _, _, _, last_frame = cells[0]
        assert last_frame == 5


class TestDecay:
    """Test dynamic cell staleness and decay."""

    def test_dynamic_cell_decays(self, grid, config):
        """A dynamic cell not updated for DECAY_MAX_AGE+1 frames should be removed."""
        points = np.array([[5.0, 5.0, 0.0]], dtype=np.float32)
        labels = np.array([2], dtype=np.int32)  # Dynamic
        grid.insert(points, labels, frame_id=0)
        
        assert grid.occupied_cell_count() == 1
        
        # Advance past max_age
        removed = grid.decay_dynamic_cells(config.DECAY_MAX_AGE + 1)
        assert removed == 1
        assert grid.occupied_cell_count() == 0

    def test_dynamic_cell_survives_if_refreshed(self, grid, config):
        """A dynamic cell refreshed within DECAY_MAX_AGE should NOT be removed."""
        points = np.array([[5.0, 5.0, 0.0]], dtype=np.float32)
        labels = np.array([2], dtype=np.int32)  # Dynamic
        grid.insert(points, labels, frame_id=0)
        
        # Refresh at frame 5
        grid.insert(points, labels, frame_id=5)
        
        # Try decay at frame 10 (5 frames since last update, max_age=10)
        removed = grid.decay_dynamic_cells(10)
        assert removed == 0
        assert grid.occupied_cell_count() == 1

    def test_static_cell_never_decays(self, grid, config):
        """Static cells (label 0 or 1) should never be decayed regardless of age."""
        points = np.array([
            [5.0, 0.0, 0.0],   # Will be terrain
            [10.0, 5.0, 0.0],  # Will be static
        ], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        
        # Advance way past max_age
        removed = grid.decay_dynamic_cells(1000)
        assert removed == 0
        assert grid.occupied_cell_count() == 2

    def test_ghost_object_removal(self, grid, config):
        """Simulate a dynamic object that appears then disappears — ghost must be cleaned up."""
        # Frame 0: dynamic car appears
        car_points = np.array([[20.0, 10.0, -0.5]], dtype=np.float32)
        car_labels = np.array([2], dtype=np.int32)
        grid.insert(car_points, car_labels, frame_id=0)
        
        # Frames 1-5: car is still there (refreshed)
        for f in range(1, 6):
            grid.insert(car_points, car_labels, frame_id=f)
        
        # Frame 6 onwards: car is gone (no more insertions)
        # Advance to frame 6 + DECAY_MAX_AGE + 1
        grid.decay_dynamic_cells(6 + config.DECAY_MAX_AGE + 1)
        
        # The ghost should be gone
        cells = grid.get_cells()
        dynamic_cells = [c for c in cells if c[1] == 2]
        assert len(dynamic_cells) == 0, "Ghost dynamic cell was not cleaned up!"

    def test_mixed_static_and_dynamic_decay(self, grid, config):
        """Only dynamic cells should be removed; static cells persist."""
        # Insert static building
        building_pts = np.array([[15.0, 0.0, 0.0]], dtype=np.float32)
        grid.insert(building_pts, np.array([1], dtype=np.int32), frame_id=0)
        
        # Insert dynamic car
        car_pts = np.array([[25.0, 5.0, 0.0]], dtype=np.float32)
        grid.insert(car_pts, np.array([2], dtype=np.int32), frame_id=0)
        
        initial_count = grid.occupied_cell_count()
        assert initial_count == 2
        
        # Decay far into the future
        removed = grid.decay_dynamic_cells(config.DECAY_MAX_AGE + 100)
        assert removed == 1  # Only the dynamic cell
        assert grid.occupied_cell_count() == 1  # Static cell remains


class TestMemory:
    """Test memory accounting."""

    def test_memory_bytes_positive(self, grid):
        """Memory should be positive after inserting points."""
        points = np.array([[5.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        
        mem = grid.memory_bytes()
        assert mem > 0
        assert mem == grid.occupied_cell_count() * grid.config.BYTES_PER_CELL

    def test_memory_zero_when_empty(self, grid):
        """Empty grid should use 0 bytes."""
        assert grid.memory_bytes() == 0

    def test_clear_resets(self, grid):
        """Clear should reset all state."""
        points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([0], dtype=np.int32)
        grid.insert(points, labels, frame_id=0)
        assert grid.occupied_cell_count() > 0
        
        grid.clear()
        assert grid.occupied_cell_count() == 0
        assert grid.memory_bytes() == 0


class TestCellDataOutput:
    """Test get_cells output format."""

    def test_get_cells_format(self, grid):
        """Each cell tuple should have (key, class, center_xyz, cell_size, last_frame)."""
        points = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        grid.insert(points, labels, frame_id=42)
        
        cells = grid.get_cells()
        assert len(cells) == 1
        
        key, cls, center, size, frame = cells[0]
        assert isinstance(key, tuple) and len(key) == 3
        assert cls == 1
        assert len(center) == 3
        assert size > 0
        assert frame == 42
