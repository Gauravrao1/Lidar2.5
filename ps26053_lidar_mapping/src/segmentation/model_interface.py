from abc import ABC, abstractmethod
import numpy as np

class SegmentationBackend(ABC):
    @abstractmethod
    def predict(self, points: np.ndarray) -> np.ndarray:
        """Input: (N, 4) float32 [x,y,z,intensity]. Output: (N,) int32 [0=T, 1=S, 2=D, 3=X]"""
        ...
    
    @abstractmethod
    def name(self) -> str:
        """Return human-readable backend name."""
        ...
    
    @property
    @abstractmethod  
    def num_classes(self) -> int:
        ...
