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

def generate_concentric_rings(n_samples: int = 120, seed: int = 42) -> Dataset:
    """Generates two concentric circular rings (inner = 0, outer = 1)."""
    rng = random.Random(seed)
    instances: List[Instance] = []
    half = n_samples // 2

    for i in range(n_samples):
        is_outer = i >= half
        radius = rng.uniform(4.0, 5.0) if is_outer else rng.uniform(0.5, 1.5)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        instances.append(Instance(features=[x, y], label=1 if is_outer else 0))

    rng.shuffle(instances)
    return Dataset(instances=instances, feature_names=["x", "y"])

def load_stratified_credit_sample(
    xls_path: str = "default of credit card clients.xls",
    n_samples: int = 300,
    seed: int = 42,
) -> Dataset:
    """
    Loads 'default of credit card clients.xls' via XLSParser and extracts a
    deterministic stratified subset preserving the observed ~22.12% default ratio.
    """
    if not os.path.exists(xls_path):
        raise FileNotFoundError(f"Credit card dataset not found at '{xls_path}'.")

    full_ds = XLSParser.load_credit_card_default(xls_path)
    rng = random.Random(seed)

    zeros = [inst for inst in full_ds if inst.label == 0]
    ones = [inst for inst in full_ds if inst.label == 1]

    default_ratio = len(ones) / len(full_ds)
    n_ones = max(1, int(round(n_samples * default_ratio)))
    n_zeros = max(1, n_samples - n_ones)

    sampled = rng.sample(zeros, min(n_zeros, len(zeros))) + rng.sample(ones, min(n_ones, len(ones)))
    rng.shuffle(sampled)
    return Dataset(instances=sampled, feature_names=list(full_ds.feature_names))