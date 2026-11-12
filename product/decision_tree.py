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

    def _calc_impurity(self, instances: List[Instance]) -> float:
        """Calculates node impurity using Gini Index or Shannon Entropy."""
        if not instances:
            return 0.0
        n = len(instances)
        counts = Counter(inst.label for inst in instances)
        if self.criterion == "entropy":
            entropy = 0.0
            for count in counts.values():
                p = count / n
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy
        else:  # "gini"
            gini = 1.0
            for count in counts.values():
                p = count / n
                gini -= p ** 2
            return gini

    def _calc_probabilities(self, instances: List[Instance]) -> List[float]:
        """Calculates class probability distribution for a subset of instances."""
        n = len(instances)
        if n == 0:
            return [1.0 / len(self.classes)] * len(self.classes)
        counts = Counter(inst.label for inst in instances)
        return [counts.get(cls, 0) / n for cls in self.classes]

    def _get_majority_label(self, instances: List[Instance]) -> Any:
        """Determines majority label with deterministic fallback ordering."""
        counts = Counter(inst.label for inst in instances)
        return max(counts.keys(), key=lambda c: (counts[c], -self.classes.index(c)))

    def _build_tree(self, instances: List[Instance], depth: int) -> TreeNode:
        """Recursively splits data to build tree nodes until stopping criteria are met."""
        num_samples = len(instances)
        probs = self._calc_probabilities(instances)
        maj_label = self._get_majority_label(instances)

        current_impurity = self._calc_impurity(instances)
        if (
            current_impurity == 0.0 or
            num_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)
        ):
            return TreeNode(is_leaf=True, prediction=maj_label, probabilities=probs)

        num_features = len(instances[0].features) if instances else 0
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        for feat_idx in range(num_features):
            vals = sorted(list(set(inst.features[feat_idx] for inst in instances)))
            if len(vals) < 2:
                continue

            thresholds = [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
            for thresh in thresholds:
                left_insts = [inst for inst in instances if inst.features[feat_idx] <= thresh]
                right_insts = [inst for inst in instances if inst.features[feat_idx] > thresh]

                if not left_insts or not right_insts:
                    continue

                w_left = len(left_insts) / num_samples
                w_right = len(right_insts) / num_samples
                split_impurity = w_left * self._calc_impurity(left_insts) + w_right * self._calc_impurity(right_insts)
                gain = current_impurity - split_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thresh
                    best_left = left_insts
                    best_right = right_insts

        if best_gain < self.min_impurity_decrease or best_feature is None:
            return TreeNode(is_leaf=True, prediction=maj_label, probabilities=probs)

        left_child = self._build_tree(best_left, depth + 1)
        right_child = self._build_tree(best_right, depth + 1)

        return TreeNode(
            is_leaf=False,
            prediction=maj_label,
            probabilities=probs,
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    