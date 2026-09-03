import numpy as np


class TemporalGridFusion:
    """Compatibility adapter for the original single-cell fusion API."""

    def __init__(self, alpha=0.5, num_classes=3):
        self.alpha = float(alpha)
        self.num_classes = int(num_classes)
        self.cells = {}

    def update_cell(self, key, label):
        if key not in self.cells:
            probabilities = np.full(self.num_classes, 1.0 / self.num_classes)
        else:
            probabilities = self.cells[key]["probabilities"]
        observation = np.zeros(self.num_classes)
        observation[int(label)] = 1.0
        probabilities = (1.0 - self.alpha) * probabilities + self.alpha * observation
        self.cells[key] = {"probabilities": probabilities}

    def get_fused_class(self, key):
        return int(np.argmax(self.cells[key]["probabilities"]))

    def get_entropy(self, key):
        probabilities = self.cells[key]["probabilities"]
        nonzero = probabilities[probabilities > 0]
        return float(-np.sum(nonzero * np.log2(nonzero)))

    def clear(self):
        self.cells.clear()
