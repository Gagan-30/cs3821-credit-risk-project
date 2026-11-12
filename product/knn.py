import math
from collections import Counter
from typing import List, Any, Dict, Tuple
from .classifier import Classifier
from .dataset import Dataset
from .instance import Instance


class KNNClassifier(Classifier):
    """
    k-Nearest Neighbours classifier supporting Euclidean/Manhattan metrics
    and multiple tie-breaking policies (majority, distance_weighted, neighbor_reduction, class_prior).
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", tie_breaker: str = "majority"):
        super().__init__()
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")
        if tie_breaker not in ("majority", "distance_weighted", "neighbor_reduction", "class_prior"):
            raise ValueError("tie_breaker must be 'majority', 'distance_weighted', 'neighbor_reduction', or 'class_prior'.")

        self.k: int = k
        self.metric: str = metric
        self.tie_breaker: str = tie_breaker
        self.train_data: Dataset = Dataset()
        self.classes: List[Any] = []
        self.class_priors: Dict[Any, int] = {}

    def fit(self, train_data: Dataset) -> None:
        """Stores training dataset, extracts unique classes, and calculates class priors."""
        if len(train_data) == 0:
            raise ValueError("Cannot fit on an empty Dataset.")
        self.train_data = train_data
        unique_labels = set(train_data.get_labels())
        self.classes = sorted(list(unique_labels))
        self.class_priors = Counter(train_data.get_labels())
        self.is_fitted = True

    def _calc_distance(self, inst1: Instance, inst2: Instance) -> float:
        """Calculates L1 (Manhattan) or L2 (Euclidean) distance between two instances."""
        if self.metric == "euclidean":
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(inst1.features, inst2.features)))
        return sum(abs(a - b) for a, b in zip(inst1.features, inst2.features))

    def _get_k_neighbors(self, instance: Instance) -> List[Tuple[float, Instance]]:
        """Returns top k nearest training instances paired with their computed distance."""
        distances = [
            (self._calc_distance(instance, train_inst), train_inst)
            for train_inst in self.train_data
        ]
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def _break_tie_class_prior(self, tied_classes: List[Any]) -> Any:
        """Resolves ties deterministically by selecting the class with highest prior count in training data."""
        return max(tied_classes, key=lambda c: (self.class_priors.get(c, 0), -self.classes.index(c)))

    def predict_instance(self, instance: Instance) -> Any:
        """Predicts class label for a single query Instance according to selected tie-breaking policy."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_instance().")

        neighbors_with_dist = self._get_k_neighbors(instance)

        if self.tie_breaker == "distance_weighted":
            weights: Dict[Any, float] = {cls: 0.0 for cls in self.classes}
            for dist, inst in neighbors_with_dist:
                w = 1.0 / (dist + 1e-9)
                weights[inst.label] += w
            max_w = max(weights.values())
            tied = [cls for cls, w in weights.items() if math.isclose(w, max_w, rel_tol=1e-7)]
            if len(tied) == 1:
                return tied[0]
            return self._break_tie_class_prior(tied)

        elif self.tie_breaker == "neighbor_reduction":
            current_neighbors = list(neighbors_with_dist)
            while len(current_neighbors) > 0:
                votes = Counter(inst.label for _, inst in current_neighbors)
                max_votes = max(votes.values())
                top_classes = [cls for cls, count in votes.items() if count == max_votes]
                if len(top_classes) == 1:
                    return top_classes[0]
                if len(current_neighbors) == 1:
                    return self._break_tie_class_prior(top_classes)
                current_neighbors.pop()
            return self._break_tie_class_prior(self.classes)

        elif self.tie_breaker == "class_prior":
            votes = Counter(inst.label for _, inst in neighbors_with_dist)
            max_votes = max(votes.values())
            top_classes = [cls for cls, count in votes.items() if count == max_votes]
            if len(top_classes) == 1:
                return top_classes[0]
            return self._break_tie_class_prior(top_classes)

        else:  # "majority"
            votes = Counter(inst.label for _, inst in neighbors_with_dist)
            max_votes = max(votes.values())
            top_classes = [cls for cls, count in votes.items() if count == max_votes]
            if len(top_classes) == 1:
                return top_classes[0]
            return self._break_tie_class_prior(top_classes)

    def predict_proba_instance(self, instance: Instance) -> List[float]:
        """Returns class probabilities based on neighbour proportions or distance weights."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_proba_instance().")

        neighbors_with_dist = self._get_k_neighbors(instance)
        if self.tie_breaker == "distance_weighted":
            weights = {cls: 0.0 for cls in self.classes}
            for dist, inst in neighbors_with_dist:
                w = 1.0 / (dist + 1e-9)
                weights[inst.label] += w
            total_w = sum(weights.values())
            if total_w == 0:
                return [1.0 / len(self.classes)] * len(self.classes)
            return [weights[cls] / total_w for cls in self.classes]
        else:
            votes = Counter(inst.label for _, inst in neighbors_with_dist)
            total = len(neighbors_with_dist)
            return [votes.get(cls, 0) / total for cls in self.classes]