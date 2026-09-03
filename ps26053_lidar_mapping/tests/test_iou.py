"""Tests for src.metrics.iou — IoU computation with known expected values."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.metrics.iou import compute_iou, compute_confusion_matrix, iou_by_distance


class TestComputeIoU:
    """Test IoU computation against hand-computed expected values."""

    def test_perfect_prediction(self):
        """Perfect prediction should give IoU = 1.0 for all classes."""
        gt   = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        per_class, miou = compute_iou(pred, gt, num_classes=3)
        for i in range(3):
            assert per_class[i] == pytest.approx(1.0, abs=1e-6), f"Class {i} IoU != 1.0"
        assert miou == pytest.approx(1.0, abs=1e-6)

    def test_completely_wrong(self):
        """Completely wrong prediction should give IoU = 0.0."""
        gt   = np.array([0, 0, 0, 0], dtype=np.int32)
        pred = np.array([1, 1, 1, 1], dtype=np.int32)
        per_class, miou = compute_iou(pred, gt, num_classes=3)
        # Class 0: TP=0, FP=0, FN=4 -> IoU=0
        # Class 1: TP=0, FP=4, FN=0 -> IoU=0
        assert per_class[0] == pytest.approx(0.0, abs=1e-6)
        assert per_class[1] == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        """Test with known partial overlap.
        GT:   [0, 0, 1, 1, 2, 2]
        Pred: [0, 1, 1, 2, 2, 0]
        
        Class 0: TP=1, FP=1, FN=1 -> IoU = 1/3
        Class 1: TP=1, FP=1, FN=1 -> IoU = 1/3
        Class 2: TP=1, FP=1, FN=1 -> IoU = 1/3
        mIoU = 1/3
        """
        gt   = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        pred = np.array([0, 1, 1, 2, 2, 0], dtype=np.int32)
        per_class, miou = compute_iou(pred, gt, num_classes=3)
        for i in range(3):
            assert per_class[i] == pytest.approx(1.0/3.0, abs=1e-6), f"Class {i}"
        assert miou == pytest.approx(1.0/3.0, abs=1e-6)

    def test_ignore_class_excluded(self):
        """Class 3 (Ignore) should be excluded from IoU computation."""
        gt   = np.array([0, 0, 3, 3, 1, 1], dtype=np.int32)
        pred = np.array([0, 0, 2, 2, 1, 1], dtype=np.int32)
        per_class, miou = compute_iou(pred, gt, num_classes=3)
        # Only classes 0 and 1 have GT points (both perfectly predicted)
        assert per_class[0] == pytest.approx(1.0, abs=1e-6)
        assert per_class[1] == pytest.approx(1.0, abs=1e-6)

    def test_single_class_present(self):
        """Only one class present — IoU should be 1.0 for that class, NaN for others."""
        gt   = np.array([0, 0, 0], dtype=np.int32)
        pred = np.array([0, 0, 0], dtype=np.int32)
        per_class, miou = compute_iou(pred, gt, num_classes=3)
        assert per_class[0] == pytest.approx(1.0, abs=1e-6)
        assert miou == pytest.approx(1.0, abs=1e-6)  # mIoU over valid classes only


class TestConfusionMatrix:
    """Test confusion matrix construction."""

    def test_shape(self):
        gt   = np.array([0, 1, 2], dtype=np.int32)
        pred = np.array([0, 1, 2], dtype=np.int32)
        cm = compute_confusion_matrix(pred, gt, num_classes=4)
        assert cm.shape == (4, 4)

    def test_diagonal_for_perfect(self):
        gt   = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        cm = compute_confusion_matrix(pred, gt, num_classes=4)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2


class TestIoUByDistance:
    """Test distance-bucketed IoU."""

    def test_basic_buckets(self):
        """IoU computed within distance buckets should have expected format."""
        n = 100
        np.random.seed(42)
        points = np.random.randn(n, 3).astype(np.float32) * 30  # spread out
        gt = np.random.randint(0, 3, n).astype(np.int32)
        pred = gt.copy()  # perfect
        
        results = iou_by_distance(pred, gt, points)
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for key, val in results.items():
            assert 'miou' in val
            assert 'num_points' in val
            assert 'per_class_iou' in val
