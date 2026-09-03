"""Tests for src.metrics.memory — uniform grid baseline calculations."""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.metrics.memory import uniform_grid_cells, uniform_grid_3d_cells, compute_memory_report


class TestUniformGrid:
    """Sanity checks on uniform grid cell count formulas."""

    def test_uniform_2d_known_area(self):
        """1 m² area at 0.05m cell size should need ceil(1/0.0025) = 400 cells."""
        cells = uniform_grid_cells(area_m2=1.0, cell_size_m=0.05)
        assert cells == 400

    def test_uniform_2d_large_area(self):
        """Circle of R=100m at 5cm cells: pi*10000 / 0.0025 ≈ 12,566,371 cells."""
        area = math.pi * 100.0**2
        cells = uniform_grid_cells(area, 0.05)
        expected = math.ceil(area / 0.05**2)
        assert cells == expected
        assert cells > 12_000_000  # sanity check

    def test_uniform_3d_known_volume(self):
        """3D should be 2D * (z_range / cell_size)."""
        area = 100.0  # m²
        z_range = 6.0  # m
        cell_size = 0.05
        cells_2d = uniform_grid_cells(area, cell_size)
        cells_3d = uniform_grid_3d_cells(area, z_range, cell_size)
        # 3D = ceil(2D * 120) ≈ 2D * 120
        expected = math.ceil(cells_2d * (z_range / cell_size))
        assert cells_3d == expected

    def test_3d_much_larger_than_2d(self):
        """3D uniform grid should always be larger than 2D by z_range/cell_size factor."""
        area = math.pi * 100.0**2
        z_range = 6.0
        cell_size = 0.05
        cells_2d = uniform_grid_cells(area, cell_size)
        cells_3d = uniform_grid_3d_cells(area, z_range, cell_size)
        assert cells_3d > cells_2d * 100  # z_range/cell_size = 120


class TestMemoryReport:
    """Test full memory report computation."""

    def test_reduction_percentage(self):
        """With few adaptive cells, reduction should be very high."""
        report = compute_memory_report(adaptive_cells=10000)
        assert report.reduction_vs_2d_pct > 90.0
        assert report.reduction_vs_3d_pct > 99.0

    def test_adaptive_bytes(self):
        """Adaptive bytes should be cells * bytes_per_cell."""
        report = compute_memory_report(adaptive_cells=5000, adaptive_bytes_per_cell=48)
        assert report.adaptive_bytes == 5000 * 48

    def test_uniform_baselines_positive(self):
        """Both uniform baselines should be positive and large."""
        report = compute_memory_report(adaptive_cells=1000)
        assert report.uniform_2d_cells > 0
        assert report.uniform_3d_cells > 0
        assert report.uniform_2d_bytes > 0
        assert report.uniform_3d_bytes > 0

    def test_process_rss_measured(self):
        """Process RSS should be a real measured value (> 0)."""
        report = compute_memory_report(adaptive_cells=100)
        assert report.process_rss_bytes > 0

    def test_reduction_never_negative(self):
        """Reduction should never be negative (clamped to 0)."""
        # Even with absurdly many cells, reduction is clamped
        report = compute_memory_report(adaptive_cells=10**12)
        assert report.reduction_vs_2d_pct >= 0.0
        assert report.reduction_vs_3d_pct >= 0.0
