import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial import cKDTree
from .model_interface import SegmentationBackend

class PointNetSegModel(nn.Module):
    """Vanilla PointNet segmentation network.
    Architecture: Conv1d(in_ch -> 64 -> 128 -> 512) + global max pool + 
    concat local(64) with global(512) -> Conv1d(576 -> 256 -> 128 -> num_classes)
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.conv4 = nn.Conv1d(576, 256, 1)
        self.conv5 = nn.Conv1d(256, 128, 1)
        self.conv6 = nn.Conv1d(128, num_classes, 1)
        
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        B, C, N = x.size()
        
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        
        # Global max pool
        global_feat = torch.max(out3, 2, keepdim=True)[0]  # (B, 512, 1)
        global_feat_expanded = global_feat.view(B, 512, 1).repeat(1, 1, N) # (B, 512, N)
        
        # Concat local and global
        concat_feat = torch.cat([out1, global_feat_expanded], dim=1) # (B, 576, N)
        
        out4 = F.relu(self.bn4(self.conv4(concat_feat)))
        out5 = F.relu(self.bn5(self.conv5(out4)))
        out6 = self.conv6(out5) # (B, num_classes, N)
        
        return out6

class PointNetBackend(SegmentationBackend):
    def __init__(self, checkpoint_path: Optional[str | Path] = None, device: str = 'cpu', num_points: int = 16384):
        """Load model. If checkpoint exists, load weights. If not, use random init (will predict poorly but won't crash)."""
        self.device = torch.device(device)
        self.num_points = num_points
        self.model = PointNetSegModel(in_channels=4, num_classes=4).to(self.device)
        
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        
        self.model.eval()
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """Subsample to num_points if needed, run inference, map back to original point count.
        Uses nearest-neighbor mapping if subsampled."""
        N = points.shape[0]
        if N == 0:
            return np.array([], dtype=np.int32)
            
        if N > self.num_points:
            # Subsample
            indices = np.random.choice(N, self.num_points, replace=False)
            sub_points = points[indices]
            needs_mapping = True
        elif N < self.num_points:
            # Pad
            pad_size = self.num_points - N
            pad_points = np.zeros((pad_size, 4), dtype=np.float32)
            sub_points = np.vstack([points, pad_points])
            needs_mapping = False
        else:
            sub_points = points
            needs_mapping = False
            
        # Inference
        tensor_points = torch.from_numpy(sub_points).float().transpose(0, 1).unsqueeze(0).to(self.device) # (1, 4, num_points)
        
        with torch.no_grad():
            preds = self.model(tensor_points) # (1, 4, num_points)
            preds_labels = preds.argmax(dim=1).squeeze(0).cpu().numpy() # (num_points,)
            
        if needs_mapping:
            # KDTree for nearest neighbor mapping
            tree = cKDTree(sub_points[:, :3])
            _, nn_indices = tree.query(points[:, :3], k=1)
            final_preds = preds_labels[nn_indices]
        else:
            final_preds = preds_labels[:N]
            
        return final_preds.astype(np.int32)
    
    @property
    def name(self) -> str: 
        return 'PointNet (Pure PyTorch, CPU)'
    
    @property
    def num_classes(self) -> int: 
        return 4
