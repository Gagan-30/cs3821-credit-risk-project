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
