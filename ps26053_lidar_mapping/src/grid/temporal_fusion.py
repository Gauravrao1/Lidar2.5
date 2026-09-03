from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, List, Optional

@dataclass
class ConfidenceCell:
    """A cell with Bayesian confidence tracking."""
    class_probabilities: np.ndarray  # (num_classes,) probability distribution
    observation_count: int
    last_frame: int
    confidence: float  # max probability = how confident we are
    predicted_class: int  # argmax of probabilities
    pseudo_counts: np.ndarray  # Store pseudo-counts for updates

class TemporalFusionGrid:
    """Maintains a confidence grid that fuses observations across multiple frames.
    
    Instead of simple majority voting (which only uses the current frame),
    this accumulates evidence over time using Bayesian updating:
    
    For each cell, we maintain P(class | all observations so far).
    Each new observation updates the posterior using:
        P(class | obs) ∝ P(obs | class) * P(class | prev_obs)
    
    This means:
    - A cell seen as 'terrain' in 10 consecutive frames has HIGH confidence
    - A cell that flickers between classes has LOW confidence
    - New cells start with uniform prior (equal probability for each class)
    """
    
    def __init__(self, num_classes: int = 3, prior_strength: float = 1.0,
                 observation_weight: float = 2.0, decay_rate: float = 0.95,
                 confidence_threshold: float = 0.6):
        self.num_classes = num_classes
        self.prior_strength = prior_strength  # Dirichlet prior pseudo-count
        self.obs_weight = observation_weight
        self.decay_rate = decay_rate  # temporal decay for old observations
        self.conf_threshold = confidence_threshold
        self.cells: Dict[Tuple[int,int,int], ConfidenceCell] = {}
    
    def update(self, cell_keys: List[Tuple[int,int,int]], labels: List[int], frame_id: int) -> None:
        """Update cells with new observations.
        For each cell:
        1. If cell doesn't exist, create with uniform Dirichlet prior
        2. Apply temporal decay to existing pseudo-counts (multiply by decay_rate)
        3. Add observation_weight to the observed class's pseudo-count
        4. Normalize to get posterior probabilities
        5. Update confidence = max(probabilities)
        6. Update predicted_class = argmax(probabilities)
        """
        for key, label in zip(cell_keys, labels):
            if key not in self.cells:
                pseudo_counts = np.full(self.num_classes, self.prior_strength, dtype=np.float64)
                # Uniform prior distribution
                initial_probs = pseudo_counts / np.sum(pseudo_counts)
                cell = ConfidenceCell(
                    class_probabilities=initial_probs,
                    observation_count=0,
                    last_frame=frame_id,
                    confidence=float(np.max(initial_probs)),
                    predicted_class=int(np.argmax(initial_probs)),
                    pseudo_counts=pseudo_counts
                )
                self.cells[key] = cell
            
            cell = self.cells[key]
            
            # Apply temporal decay to existing pseudo-counts
            cell.pseudo_counts *= self.decay_rate
            
            # Add observation_weight to the observed class's pseudo-count
            cell.pseudo_counts[label] += self.obs_weight
            
            # Normalize to get posterior probabilities
            total_counts = np.sum(cell.pseudo_counts)
            cell.class_probabilities = cell.pseudo_counts / total_counts
            
            # Update observation stats
            cell.observation_count += 1
            cell.last_frame = frame_id
            
            # Update confidence and predicted class
            cell.confidence = float(np.max(cell.class_probabilities))
            cell.predicted_class = int(np.argmax(cell.class_probabilities))
    
    def get_confident_cells(self) -> List[Tuple[Tuple[int,int,int], int, float]]:
        """Return cells where confidence > threshold: [(key, class, confidence), ...]"""
        return [(k, v.predicted_class, v.confidence) for k, v in self.cells.items() if v.confidence > self.conf_threshold]
    
    def get_uncertain_cells(self) -> List[Tuple[Tuple[int,int,int], float]]:
        """Return cells where confidence < threshold: [(key, confidence), ...]"""
        return [(k, v.confidence) for k, v in self.cells.items() if v.confidence < self.conf_threshold]
    
    def get_class_distribution(self, key: Tuple[int,int,int]) -> Optional[np.ndarray]:
        """Return probability distribution for a cell, or None if not observed."""
        cell = self.cells.get(key)
        return cell.class_probabilities if cell is not None else None
    
    def overall_confidence(self) -> float:
        """Mean confidence across all cells."""
        if not self.cells:
            return 0.0
        return float(np.mean([cell.confidence for cell in self.cells.values()]))
    
    def entropy(self, key: Tuple[int,int,int]) -> float:
        """Shannon entropy of the class distribution for a cell. Lower = more certain."""
        cell = self.cells.get(key)
        if cell is None:
            # If cell is unobserved, return entropy of uniform distribution
            return float(np.log2(self.num_classes))
        
        p = cell.class_probabilities
        p_nonzero = p[p > 0]
        return float(-np.sum(p_nonzero * np.log2(p_nonzero)))
    
    def summary(self) -> dict:
        """Return summary: total_cells, confident_cells, uncertain_cells, mean_confidence, mean_entropy."""
        if not self.cells:
            return {
                "total_cells": 0,
                "confident_cells": 0,
                "uncertain_cells": 0,
                "mean_confidence": 0.0,
                "mean_entropy": 0.0
            }
        
        total_cells = len(self.cells)
        confident_cells = sum(1 for v in self.cells.values() if v.confidence > self.conf_threshold)
        uncertain_cells = sum(1 for v in self.cells.values() if v.confidence < self.conf_threshold)
        mean_confidence = self.overall_confidence()
        mean_entropy = float(np.mean([self.entropy(k) for k in self.cells.keys()]))
        
        return {
            "total_cells": total_cells,
            "confident_cells": confident_cells,
            "uncertain_cells": uncertain_cells,
            "mean_confidence": mean_confidence,
            "mean_entropy": mean_entropy
        }
    
    def clear(self) -> None:
        """Reset all state."""
        self.cells.clear()
