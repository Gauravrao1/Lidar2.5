"""
Train PointNet segmentation model on sample SemanticKITTI data.

This script:
1. Loads all sample scans from data/sample/
2. Applies class remap (28 -> 4 classes)
3. Subsamples points to 16384 per scan
4. Trains a Vanilla PointNet for N epochs
5. Saves checkpoint to checkpoints/pointnet_3class.pth

Usage:
  .venv\\Scripts\\python.exe scripts\\train_pointnet.py --epochs 50
"""

import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.kitti_loader import load_class_remap, load_bin, load_labels
from src.segmentation.pointnet_infer import PointNetSegModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pointnet")


class LidarSegDataset(Dataset):
    """Dataset that loads sample LiDAR scans for training."""
    
    def __init__(self, data_dir: Path, remap: dict, num_points: int = 16384):
        self.num_points = num_points
        self.samples = []
        
        velodyne_dir = data_dir / "velodyne"
        labels_dir = data_dir / "labels"
        
        for bin_path in sorted(velodyne_dir.glob("*.bin")):
            label_path = labels_dir / f"{bin_path.stem}.label"
            if label_path.exists():
                points = load_bin(bin_path)
                labels = load_labels(label_path, remap)
                self.samples.append((points, labels))
        
        logger.info("Loaded %d samples", len(self.samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        points, labels = self.samples[idx]
        n = points.shape[0]
        
        if n > self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
        elif n < self.num_points:
            indices = np.random.choice(n, self.num_points, replace=True)
        else:
            indices = np.arange(n)
        
        pts = points[indices].astype(np.float32)  # (num_points, 4)
        lbl = labels[indices].astype(np.int64)      # (num_points,)
        
        # Transpose for Conv1d: (4, num_points)
        pts_t = pts.T
        
        return torch.from_numpy(pts_t), torch.from_numpy(lbl)


def train(data_dir: Path, remap_path: Path, output_path: Path,
          epochs: int = 50, lr: float = 0.001, batch_size: int = 2,
          num_points: int = 16384):
    """Train PointNet on sample data."""
    
    remap = load_class_remap(remap_path)
    dataset = LidarSegDataset(data_dir, remap, num_points=num_points)
    
    if len(dataset) == 0:
        logger.error("No training samples found in %s", data_dir)
        return
    
    # Use all samples for training (small dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model = PointNetSegModel(in_channels=4, num_classes=4)
    model.train()
    
    # Class weights (Ignore class gets 0 weight)
    # Roughly balance terrain (many points) vs dynamic (few points)
    class_weights = torch.tensor([1.0, 2.0, 5.0, 0.0], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    logger.info("Training PointNet: %d samples, %d epochs, lr=%.4f", len(dataset), epochs, lr)
    
    t_start = time.time()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for pts_batch, lbl_batch in loader:
            # pts_batch: (B, 4, N), lbl_batch: (B, N)
            optimizer.zero_grad()
            output = model(pts_batch)  # (B, 4, N)
            
            # Reshape for cross entropy: (B*N, 4) vs (B*N,)
            B, C, N = output.shape
            output_flat = output.permute(0, 2, 1).reshape(-1, C)
            labels_flat = lbl_batch.reshape(-1)
            
            loss = criterion(output_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            preds = output.argmax(dim=1)  # (B, N)
            mask = lbl_batch != 3  # Ignore class 3
            epoch_correct += (preds[mask] == lbl_batch[mask]).sum().item()
            epoch_total += mask.sum().item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(loader)
        accuracy = epoch_correct / max(1, epoch_total)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %3d/%d: loss=%.4f, accuracy=%.4f, lr=%.6f",
                       epoch + 1, epochs, avg_loss, accuracy, scheduler.get_last_lr()[0])
    
    elapsed = time.time() - t_start
    logger.info("Training complete in %.1f seconds", elapsed)
    
    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info("Checkpoint saved to %s", output_path)
    
    # Save training log
    log_path = output_path.parent / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(loss_history):
            f.write(f"{i+1},{loss:.6f}\n")
    logger.info("Training log saved to %s", log_path)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PointNet on sample data")
    parser.add_argument("--data-dir", type=str,
                       default=str(PROJECT_ROOT / "data" / "sample"))
    parser.add_argument("--remap-config", type=str,
                       default=str(PROJECT_ROOT / "configs" / "class_remap.yaml"))
    parser.add_argument("--output", type=str,
                       default=str(PROJECT_ROOT / "checkpoints" / "pointnet_3class.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=16384)
    
    args = parser.parse_args()
    
    train(
        data_dir=Path(args.data_dir),
        remap_path=Path(args.remap_config),
        output_path=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_points=args.num_points,
    )


if __name__ == "__main__":
    main()
