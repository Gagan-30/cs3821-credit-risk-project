import math
from collections import Counter
from typing import List, Any, Optional
from .classifier import Classifier
from .dataset import Dataset
from .instance import Instance


class TreeNode:
    """Represents an internal decision node or leaf node within a DecisionTree."""

    __slots__ = (
        "is_leaf",
        "prediction",
        "probabilities",
        "feature_idx",
        "threshold",
        "left",
        "right"
    )

    def __init__(
        self,
        is_leaf: bool = False,
        prediction: Optional[Any] = None,
        probabilities: Optional[List[float]] = None,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None
    ):
        self.is_leaf: bool = is_leaf
        self.prediction: Optional[Any] = prediction
        self.probabilities: List[float] = probabilities if probabilities is not None else []
        self.feature_idx: Optional[int] = feature_idx
        self.threshold: Optional[float] = threshold
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


class DecisionTreeClassifier(Classifier):
    """
    Recursive Decision Tree Classifier supporting continuous features,
    Information Gain (Shannon Entropy), and Gini Impurity heuristics.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 1e-7
    ):
        super().__init__()
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1 or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        self.criterion: str = criterion
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_impurity_decrease: float = min_impurity_decrease
        self.root: Optional[TreeNode] = None
        self.classes: List[Any] = []

    def fit(self, train_data: Dataset) -> None:
        """Constructs the recursive decision tree from the provided training dataset."""
        if len(train_data) == 0:
            raise ValueError("Cannot fit on an empty Dataset.")
        self.classes = sorted(list(set(train_data.get_labels())))
        self.root = self._build_tree(train_data.instances, depth=0)
        self.is_fitted = True
