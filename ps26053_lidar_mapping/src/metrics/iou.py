import numpy as np
from typing import Optional

def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Returns (num_classes, num_classes) confusion matrix. Ignores class 3 (Ignore)."""
    mask = (gt >= 0) & (gt < num_classes) & (gt != 3) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(num_classes * gt[mask].astype(int) + pred[mask].astype(int), minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes)

def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 3) -> tuple[np.ndarray, float]:
    """Compute per-class IoU and mIoU for classes 0,1,2 (ignoring class 3).
    Returns (per_class_iou: (3,) array, miou: float).
    IoU = TP / (TP + FP + FN) for each class."""
    cm = compute_confusion_matrix(pred, gt, num_classes=4)
    per_class_iou = np.zeros(num_classes, dtype=float)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            per_class_iou[i] = tp / denom
        else:
            per_class_iou[i] = float('nan')
    
    valid_iou = per_class_iou[~np.isnan(per_class_iou)]
    miou = float(np.mean(valid_iou)) if valid_iou.size > 0 else 0.0
    return per_class_iou, miou

def iou_by_distance(pred: np.ndarray, gt: np.ndarray, points: np.ndarray,
                     distance_buckets: Optional[list[float]] = None,
                     num_classes: int = 3) -> dict:
    """Compute IoU bucketed by distance from sensor.
    Default buckets: [0, 10, 20, 40, 60, 80, 100]
    Returns dict with bucket_range -> {per_class_iou, miou, num_points}."""
    if distance_buckets is None:
        distance_buckets = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    
    distances = np.linalg.norm(points[:, :3], axis=1)
    
    results = {}
    for i in range(len(distance_buckets) - 1):
        low = distance_buckets[i]
        high = distance_buckets[i+1]
        
        mask = (distances >= low) & (distances < high)
        bucket_pred = pred[mask]
        bucket_gt = gt[mask]
        
        if len(bucket_gt) == 0:
            results[f"{low}-{high}m"] = {
                "per_class_iou": np.zeros(num_classes),
                "miou": 0.0,
                "num_points": 0
            }
            continue
            
        pc_iou, miou = compute_iou(bucket_pred, bucket_gt, num_classes)
        results[f"{low}-{high}m"] = {
            "per_class_iou": pc_iou,
            "miou": miou,
            "num_points": int(np.sum(mask))
        }
        
    return results
