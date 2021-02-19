import abc
from typing import Any, Dict

import numpy as np
import pandas as pd


class AbstractModel(metaclass=abc.ABCMeta):

    _model: Any

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit on training data.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.

        Returns:
            None.

        """

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels on new data.

        Args:
            X: Input features.

        Returns:
            Predicted class label for each observation.

        """

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of each label.

        Args:
            X: Input features.

        Returns:
            Predicted probabilities of each label, for each observation.

        """

    @abc.abstractmethod
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Compute F1-score and LogLoss.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.

        Returns:
            A dictionary containing the values of the `f1_score` and the `logloss`
            of the model on the given data.

        """

    @abc.abstractmethod
    def tune_parameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Choose the best parameters with K-fold cross-validation.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.

        Returns:
            A dictionary containing the average scores across
            all CV validation partitions and best parameters.

        """
