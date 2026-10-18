import math
from collections import Counter
from typing import List, Any
from .classifier import Classifier
from .dataset import Dataset
from .instance import Instance


class KNNClassifier(Classifier):
    """k-Nearest Neighbours classifier supporting Euclidean and Manhattan metrics."""

    def __init__(self, k: int = 3, metric: str = "euclidean"):
        super().__init__()
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.k: int = k
        self.metric: str = metric
        self.train_data: Dataset = Dataset()
        self.classes: List[Any] = []

    def fit(self, train_data: Dataset) -> None:
        """Stores training instances and extracts unique class labels."""
        if len(train_data) == 0:
            raise ValueError("Cannot fit on an empty Dataset.")
        self.train_data = train_data
        
        # Sort unique class labels for consistent probability indexing
        unique_labels = set(train_data.get_labels())
        self.classes = sorted(list(unique_labels))
        self.is_fitted = True

    def _calc_distance(self, inst1: Instance, inst2: Instance) -> float:
        """Calculates L1 or L2 distance between two instances."""
        if self.metric == "euclidean":
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(inst1.features, inst2.features)))
        return sum(abs(a - b) for a, b in zip(inst1.features, inst2.features))

    def _get_k_neighbors(self, instance: Instance) -> List[Instance]:
        """Returns the k closest training instances to the query instance."""
        distances = [
            (self._calc_distance(instance, train_inst), train_inst)
            for train_inst in self.train_data
        ]
        distances.sort(key=lambda x: x[0])
        return [inst for _, inst in distances[:self.k]]

    def predict_instance(self, instance: Instance) -> Any:
        """Predicts label via majority vote among k nearest neighbours."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_instance().")

        neighbors = self._get_k_neighbors(instance)
        votes = Counter(n.label for n in neighbors)
        return votes.most_common(1)[0][0]

    def predict_proba_instance(self, instance: Instance) -> List[float]:
        """Returns class probabilities based on neighbour label proportions."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_proba_instance().")

        neighbors = self._get_k_neighbors(instance)
        votes = Counter(n.label for n in neighbors)
        total = len(neighbors)
        return [votes[cls] / total for cls in self.classes]