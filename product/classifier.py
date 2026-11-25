from abc import ABC, abstractmethod
from typing import List, Any
from .dataset import Dataset
from .instance import Instance


class Classifier(ABC):
    """Abstract Base Class defining the unified API contract for all classifiers."""

    def __init__(self, **kwargs):
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, train_data: Dataset) -> None:
        """Train classifier on provided dataset."""
        pass

    @abstractmethod
    def predict_instance(self, instance: Instance) -> Any:
        """Predict target label for a single Instance."""
        pass

    def predict(self, test_data: Dataset) -> List[Any]:
        """Predict target labels for all instances in a Dataset."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict().")
        return [self.predict_instance(inst) for inst in test_data]

    def predict_proba_instance(self, instance: Instance) -> List[float]:
        """Return class probability distributions for a single Instance."""
        raise NotImplementedError("Probability estimation not implemented for this model.")

    def predict_proba(self, test_data: Dataset) -> List[List[float]]:
        """Return class probabilities for all instances in a Dataset."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_proba().")
        return [self.predict_proba_instance(inst) for inst in test_data]