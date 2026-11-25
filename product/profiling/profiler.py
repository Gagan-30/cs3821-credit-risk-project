import os
import time
import tracemalloc
from typing import Optional, Tuple, Dict, Any, List
from product.decision_tree import DecisionTreeClassifier, TreeNode
from product.knn import KNNClassifier
from product.preprocessors import ZScoreStandardizer
from product.tests.synthetic_data import (
    generate_linear_clusters,
    generate_quadrant_clusters,
    load_stratified_credit_sample,
)

CREDIT_XLS_PATH = "default of credit card clients.xls"

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

def profile_decision_tree_recursion(dataset_label: str, ds) -> List[Dict[str, Any]]:
    """Profiles Decision Tree recursion depth, node count, fit latency, and peak memory."""
    results: List[Dict[str, Any]] = []

    for criterion in ("gini", "entropy"):
        for max_depth in (1, 2, 4, None):
            tracemalloc.start()
            t0 = time.perf_counter_ns()

            dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
            dt.fit(ds)

            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            actual_depth, total_nodes, leaf_nodes = inspect_tree_structure(dt.root)
            results.append({
                "dataset": dataset_label,
                "criterion": criterion,
                "max_depth_param": str(max_depth),
                "actual_depth": actual_depth,
                "total_nodes": total_nodes,
                "leaf_nodes": leaf_nodes,
                "fit_ms": round(elapsed_ms, 3),
                "peak_kb": round(peak_bytes / 1024.0, 2),
            })

    return results

def profile_knn_distance_scaling(
    dataset_label: str,
    datasets_by_size: List[Tuple[int, Any]]
) -> List[Dict[str, Any]]:
    """Profiles k-NN pairwise distance latency and peak memory across dataset sizes."""
    results: List[Dict[str, Any]] = []

    for n_samples, ds in datasets_by_size:
        train_raw, test_raw = ds.split(0.8)
        scaler = ZScoreStandardizer()
        train_ds = scaler.fit_transform(train_raw)
        test_ds = scaler.transform(test_raw)

        for metric in ("euclidean", "manhattan"):
            knn = KNNClassifier(k=5, metric=metric)
            knn.fit(train_ds)

            tracemalloc.start()
            t0 = time.perf_counter_ns()

            _ = knn.predict(test_ds)

            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            per_query_us = (elapsed_ms * 1000.0) / len(test_ds)
            results.append({
                "dataset": dataset_label,
                "n_train": len(train_ds),
                "n_test": len(test_ds),
                "n_features": ds.num_features,
                "metric": metric,
                "total_ms": round(elapsed_ms, 3),
                "per_query_us": round(per_query_us, 2),
                "peak_kb": round(peak_bytes / 1024.0, 2),
            })

    return results

def run_all_profiles() -> None:
    print("=== 1. DECISION TREE RECURSION & IMPURITY PROFILING ===")
    dt_datasets = [("Synthetic_400x2", generate_quadrant_clusters(n_samples=400, seed=42))]
    if os.path.exists(CREDIT_XLS_PATH):
        dt_datasets.append(("CreditCard_250x23", load_stratified_credit_sample(CREDIT_XLS_PATH, n_samples=250, seed=42)))

    header_dt = f"{'Dataset':<18} {'Criterion':<10} {'MaxDepth':<9} {'ActualDepth':<12} {'Nodes':<7} {'Leaves':<7} {'Fit (ms)':<10} {'Peak (KB)':<10}"
    print(header_dt)
    print("-" * len(header_dt))
    for label, ds in dt_datasets:
        for r in profile_decision_tree_recursion(label, ds):
            print(
                f"{r['dataset']:<18} {r['criterion']:<10} {r['max_depth_param']:<9} {r['actual_depth']:<12} "
                f"{r['total_nodes']:<7} {r['leaf_nodes']:<7} {r['fit_ms']:<10} {r['peak_kb']:<10}"
            )

    print("\n=== 2. k-NN PAIRWISE DISTANCE & MEMORY PROFILING ===")
    knn_groups = [
        (
            "Synthetic_10D",
            [(n, generate_linear_clusters(n_samples=n, n_features=10, seed=42)) for n in (100, 500, 1000)],
        )
    ]
    if os.path.exists(CREDIT_XLS_PATH):
        knn_groups.append(
            (
                "CreditCard_23D",
                [(n, load_stratified_credit_sample(CREDIT_XLS_PATH, n_samples=n, seed=42)) for n in (100, 500, 1000)],
            )
        )

    header_knn = f"{'Dataset':<16} {'Train N':<9} {'Test N':<8} {'Metric':<11} {'Total (ms)':<12} {'Per-Query (us)':<16} {'Peak (KB)':<10}"
    print(header_knn)
    print("-" * len(header_knn))
    for label, group in knn_groups:
        for r in profile_knn_distance_scaling(label, group):
            print(
                f"{r['dataset']:<16} {r['n_train']:<9} {r['n_test']:<8} {r['metric']:<11} "
                f"{r['total_ms']:<12} {r['per_query_us']:<16} {r['peak_kb']:<10}"
            )

if __name__ == "__main__":
    run_all_profiles()