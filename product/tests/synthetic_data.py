import math
import os
import random
from typing import List
from product.dataset import Dataset
from product.instance import Instance
from product.xls_parser import XLSParser

def generate_linear_clusters(
    n_samples: int = 100,
    n_features: int = 2,
    seed: int = 42
) -> Dataset:
    """Generates two linearly separable Gaussian-like clusters (labels 0 and 1)."""
    rng = random.Random(seed)
    instances: List[Instance] = []
    half = n_samples // 2

    for _ in range(half):
        feats = [rng.gauss(2.0, 0.5) for _ in range(n_features)]
        instances.append(Instance(features=feats, label=0))

    for _ in range(n_samples - half):
        feats = [rng.gauss(6.0, 0.5) for _ in range(n_features)]
        instances.append(Instance(features=feats, label=1))

    rng.shuffle(instances)
    feature_names = [f"f_{i}" for i in range(n_features)]
    return Dataset(instances=instances, feature_names=feature_names)

def generate_quadrant_clusters(n_samples: int = 120, seed: int = 42) -> Dataset:
    """Generates a 2D non-linear XOR-like quadrant dataset requiring multi-level tree splits."""
    rng = random.Random(seed)
    instances: List[Instance] = []
    quadrants = [
        ((1.0, 1.0), 0),
        ((5.0, 5.0), 0),
        ((1.0, 5.0), 1),
        ((5.0, 1.0), 1),
    ]

    for i in range(n_samples):
        (cx, cy), label = quadrants[i % 4]
        feats = [rng.gauss(cx, 0.4), rng.gauss(cy, 0.4)]
        instances.append(Instance(features=feats, label=label))

    rng.shuffle(instances)
    return Dataset(instances=instances, feature_names=["x", "y"])

