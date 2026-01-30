from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Evaluator(ABC):
    """
    Abstract base class for evaluators.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the evaluator."""
        pass

    @abstractmethod
    def update(self, predictions: Dict[str, Any], targets: List[Any]) -> None:
        """
        Update the evaluator with a batch of predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        """
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute and return the evaluation metrics.

        Returns:
            A dictionary containing evaluation metrics.
        """
        pass
