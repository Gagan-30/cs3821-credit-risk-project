from typing import List, Tuple, Optional, Any
import random
from .instance import Instance


class Dataset:
    """Collection of Instance objects with utility functions."""

    def __init__(
        self,
        instances: Optional[List[Instance]] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.instances: List[Instance] = instances if instances is not None else []
        self.feature_names: List[str] = feature_names if feature_names is not None else []

    def add(self, instance: Instance) -> None:
        self.instances.append(instance)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Instance:
        return self.instances[idx]

    def __repr__(self) -> str:
        return f"Dataset(instances={len(self.instances)}, features={self.num_features})"

    @property
    def num_features(self) -> int:
        return len(self.instances[0]) if self.instances else 0

    def get_features_matrix(self) -> List[List[float]]:
        return [inst.features for inst in self.instances]

    def get_labels(self) -> List[Any]:
        return [inst.label for inst in self.instances]

    def get_column(self, col_idx: int) -> List[float]:
        return [inst.features[col_idx] for inst in self.instances]

    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.instances)

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        split_idx = int(len(self.instances) * ratio)
        d1 = Dataset(self.instances[:split_idx], self.feature_names)
        d2 = Dataset(self.instances[split_idx:], self.feature_names)
        return d1, d2