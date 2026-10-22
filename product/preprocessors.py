import math
from abc import ABC, abstractmethod
from typing import List
from .dataset import Dataset
from .instance import Instance


class BasePreprocessor(ABC):
    """Abstract base class for stateful feature preprocessors."""

    def __init__(self):
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        pass

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        pass

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


class MinMaxScaler(BasePreprocessor):
    """Scales feature vectors to the [0, 1] range."""

    def __init__(self):
        super().__init__()
        self.mins: List[float] = []
        self.maxs: List[float] = []

    def fit(self, dataset: Dataset) -> None:
        if len(dataset) == 0:
            raise ValueError("Cannot fit on an empty Dataset.")

        num_features = dataset.num_features
        self.mins = [float("inf")] * num_features
        self.maxs = [float("-inf")] * num_features

        for inst in dataset:
            for i, val in enumerate(inst.features):
                if val < self.mins[i]:
                    self.mins[i] = val
                if val > self.maxs[i]:
                    self.maxs[i] = val

        self.is_fitted = True

    def transform(self, dataset: Dataset) -> Dataset:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        scaled_dataset = Dataset(feature_names=dataset.feature_names)
        for inst in dataset:
            scaled_feats = []
            for i, val in enumerate(inst.features):
                denom = self.maxs[i] - self.mins[i]
                scaled_val = (val - self.mins[i]) / denom if denom != 0 else 0.0
                scaled_feats.append(scaled_val)
            scaled_dataset.add(Instance(features=scaled_feats, label=inst.label, meta=inst.meta))

        return scaled_dataset
