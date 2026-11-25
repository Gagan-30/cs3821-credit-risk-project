import time
import tracemalloc
from typing import Optional, Tuple, Dict, Any, List
from product.decision_tree import DecisionTreeClassifier, TreeNode
from product.knn import KNNClassifier
from product.tests.synthetic_data import generate_linear_clusters, generate_quadrant_clusters


def inspect_tree_structure(node: Optional[TreeNode]) -> Tuple[int, int, int]:
    """Returns (max_depth, total_nodes, leaf_nodes) of a fitted DecisionTree."""
    if node is None:
        return 0, 0, 0
    if node.is_leaf:
        return 0, 1, 1

    left_d, left_n, left_l = inspect_tree_structure(node.left)
    right_d, right_n, right_l = inspect_tree_structure(node.right)
    return (
        1 + max(left_d, right_d),
        1 + left_n + right_n,
        left_l + right_l,
    )


