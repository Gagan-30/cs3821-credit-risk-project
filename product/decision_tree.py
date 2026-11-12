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


