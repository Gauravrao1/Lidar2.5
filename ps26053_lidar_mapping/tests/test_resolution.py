"""Tests for src.grid.resolution — log-linear cell size and point-to-cell-key."""
import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.grid.resolution import cell_size, point_to_cell_key, GridConfig


@pytest.fixture
def config():
    return GridConfig()


class TestCellSize:
    """Test the log-linear cell_size function."""

    def test_at_r_zero(self, config):
        """r=0 should be clamped to epsilon and return S_NEAR."""
        s = cell_size(0.0, config)
        assert s == pytest.approx(config.S_NEAR, abs=1e-6)

    def test_at_r_near_boundary(self, config):
        """r exactly at R_NEAR should return S_NEAR."""
        s = cell_size(config.R_NEAR, config)
        assert s == pytest.approx(config.S_NEAR, abs=1e-6)

    def test_below_r_near(self, config):
        """r < R_NEAR should return S_NEAR."""
        for r in [0.01, 1.0, 5.0, 9.99]:
            s = cell_size(r, config)
            assert s == pytest.approx(config.S_NEAR, abs=1e-6), f"Failed for r={r}"

    def test_at_r_far_boundary(self, config):
        """r exactly at R_FAR should return S_FAR."""
        s = cell_size(config.R_FAR, config)
        assert s == pytest.approx(config.S_FAR, abs=1e-6)

    def test_above_r_far(self, config):
        """r > R_FAR should return S_FAR."""
        for r in [100.01, 150.0, 500.0]:
            s = cell_size(r, config)
            assert s == pytest.approx(config.S_FAR, abs=1e-6), f"Failed for r={r}"

    def test_midpoint_interpolation(self, config):
        """Check that midpoint (geometric mean of R_NEAR and R_FAR) gives correct size."""
        r_mid = math.sqrt(config.R_NEAR * config.R_FAR)  # geometric mean = sqrt(10*100) ≈ 31.62
        s = cell_size(r_mid, config)
        # At geometric mean, exponent = 0.5, so s = S_NEAR * (S_FAR/S_NEAR)^0.5
        expected = config.S_NEAR * math.sqrt(config.S_FAR / config.S_NEAR)
        assert s == pytest.approx(expected, rel=1e-6)

    def test_monotonically_increasing(self, config):
        """Cell size should increase monotonically with distance."""
        r_values = np.linspace(0.01, 200.0, 1000)
        sizes = cell_size(r_values, config)
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i-1] - 1e-10, (
                f"Non-monotonic at r={r_values[i]}: s={sizes[i]} < s_prev={sizes[i-1]}"
            )

    def test_continuity_at_r_near(self, config):
        """Cell size should be continuous across the R_NEAR boundary."""
        r_below = config.R_NEAR - 0.001
        r_at = config.R_NEAR
        r_above = config.R_NEAR + 0.001
        s_below = cell_size(r_below, config)
        s_at = cell_size(r_at, config)
        s_above = cell_size(r_above, config)
        assert abs(s_at - s_below) < 0.001, f"Discontinuity at R_NEAR: {s_below} vs {s_at}"
        assert abs(s_above - s_at) < 0.001, f"Discontinuity at R_NEAR: {s_at} vs {s_above}"

    def test_continuity_at_r_far(self, config):
        """Cell size should be continuous across the R_FAR boundary."""
        r_below = config.R_FAR - 0.001
        r_at = config.R_FAR
        s_below = cell_size(r_below, config)
        s_at = cell_size(r_at, config)
        assert abs(s_at - s_below) < 0.001, f"Discontinuity at R_FAR: {s_below} vs {s_at}"

    def test_vectorized_matches_scalar(self, config):
        """Vectorized computation should match scalar."""
        r_values = [0.0, 5.0, 10.0, 31.62, 50.0, 100.0, 200.0]
        scalar_results = [cell_size(r, config) for r in r_values]
        vector_results = cell_size(np.array(r_values), config)
        for i, (s, v) in enumerate(zip(scalar_results, vector_results)):
            assert s == pytest.approx(v, rel=1e-6), f"Mismatch at r={r_values[i]}"

    def test_negative_r_clamped(self, config):
        """Negative r should be clamped to epsilon and return S_NEAR."""
        s = cell_size(-5.0, config)
        assert s == pytest.approx(config.S_NEAR, abs=1e-6)


class TestPointToCellKey:
    """Test Cartesian-to-polar cell key conversion."""

    def test_origin(self, config):
        """Point at origin should get a valid key."""
        key = point_to_cell_key(0.0, 0.0, 0.0, config)
        assert len(key) == 3
        ring, sector, z_bin = key
        assert isinstance(ring, (int, np.integer))
        assert isinstance(sector, (int, np.integer))
        assert isinstance(z_bin, (int, np.integer))

    def test_z_clamping(self, config):
        """Z values outside [Z_MIN, Z_MAX] should be clamped."""
        _, _, z_bin_low = point_to_cell_key(5.0, 0.0, -100.0, config)
        _, _, z_bin_high = point_to_cell_key(5.0, 0.0, 100.0, config)
        assert z_bin_low >= 0
        max_z_bin = int(math.ceil((config.Z_MAX - config.Z_MIN) / config.Z_BIN_SIZE)) - 1
        assert z_bin_high <= max_z_bin

    def test_sector_wraparound_positive_pi(self, config):
        """Points at theta near +pi and -pi should be in adjacent/same sectors."""
        # Point at angle just below pi
        x1, y1 = -10.0, 0.001
        _, s1, _ = point_to_cell_key(x1, y1, 0.0, config)
        # Point at angle just above -pi
        x2, y2 = -10.0, -0.001
        _, s2, _ = point_to_cell_key(x2, y2, 0.0, config)
        # They should be in adjacent or nearly adjacent sectors
        assert abs(s1 - s2) <= 1 or abs(s1 - s2) >= config.N_SECTORS - 1

    def test_vectorized_keys(self, config):
        """Vectorized call should match individual calls."""
        xs = np.array([5.0, 10.0, 50.0])
        ys = np.array([5.0, 0.0, -10.0])
        zs = np.array([0.0, -1.0, 1.0])
        
        rings_v, sectors_v, zbins_v = point_to_cell_key(xs, ys, zs, config)
        
        for i in range(3):
            ring_s, sector_s, zbin_s = point_to_cell_key(xs[i], ys[i], zs[i], config)
            assert ring_s == rings_v[i], f"Ring mismatch at point {i}"
            assert sector_s == sectors_v[i], f"Sector mismatch at point {i}"
            assert zbin_s == zbins_v[i], f"Z-bin mismatch at point {i}"

    def test_different_distances_different_rings(self, config):
        """Points at very different distances should generally land in different rings."""
        _, _, _ = point_to_cell_key(1.0, 0.0, 0.0, config)  # r=1
        r1, _, _ = point_to_cell_key(1.0, 0.0, 0.0, config)
        r2, _, _ = point_to_cell_key(50.0, 0.0, 0.0, config)
        assert r1 != r2
