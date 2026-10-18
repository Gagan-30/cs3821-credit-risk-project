from typing import List, Optional, Any


class Instance:
    """Represents a single feature vector and target label."""

    __slots__ = ("features", "label", "meta")

    def __init__(
        self,
        features: List[float],
        label: Optional[Any] = None,
        meta: Optional[dict] = None
    ):
        self.features: List[float] = [float(x) for x in features]
        self.label: Optional[Any] = label
        self.meta: dict = meta if meta is not None else {}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> float:
        return self.features[idx]

    def __repr__(self) -> str:
        return f"Instance(features_count={len(self.features)}, label={self.label})"