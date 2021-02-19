from typing import Any, Dict

import numpy as np
import pandas as pd
from kerastuner import HyperParameters
from kerastuner.engine.hypermodel import KerasHyperModel
from kerastuner.oracles import BayesianOptimization
from tensorflow.python.framework.tensor_spec import TensorSpec

from autoyml.abstract_model import AbstractModel
from autoyml.decorators import require_fit
from autoyml.hypermodel import CustomKerasHyperModel, TunableNeuralNetwork
from autoyml.preprocessing import DataPreprocessor
from autoyml.tuner import CrossValidationTuner


class NeuralNetworkModel(AbstractModel):
    def __init__(self) -> None:
        self._model = None
        self._preprocessor = DataPreprocessor

    def fit(self, X: pd.DataFrame, y: np.ndarray, batch_size: int = 32, epochs: int = 20) -> None:
        """Fit on training data.

        Notes:
            The input of the model is determined by the features metadata.
            If a model has already been found (by hyperparameter tuning for example),
            the fitting is done on this model, else on the model with the default hyperparameters.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.

        Returns:
            None.

        """
        dataset = self._preprocessor.preprocess_fit(X, y, batch_size=batch_size)
        dataset_spec = self._preprocessor.get_dataset_spec(dataset)

        if self._model is None:
            self._model = self._model_factory(dataset_spec).build(hp=HyperParameters(), dataset=dataset)

        class_weight = self._preprocessor.get_class_weight(dataset)

        self._model.fit(dataset, epochs=epochs, class_weight=class_weight)

    @require_fit
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels on new data.

        Note:
            The model instance must be fitted before prediction.

        Args:
            X: Input features.

        Returns:
            Predicted class label for each observation.

        Raises:
            NotFittedError: If the `fit` method has not been called before.

        """
        return np.round(self.predict_proba(X)).astype(np.int32)

    @require_fit
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of each label.

        Args:
            X: Input features.

        Returns:
            Predicted probabilities of the positive label for each observation,
            as a numpy array of floats between 0 and 1.

        """
        return self._model.predict(self._preprocessor.preprocess_predict(X)).flatten()

    @require_fit
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Compute F1-score and LogLoss.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.

        Returns:
            A dictionary containing the values of the `f1_score` and the `logloss`
            of the model on the given data.

        """
        metrics = self._model.evaluate(self._preprocessor.preprocess_predict(X), y, return_dict=True)
        return {"logloss": metrics["loss"], "f1_score": metrics["f1_score"]}

    def tune_parameters(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        batch_size: int = 32,
        max_trials: int = 16,
    ) -> Dict[str, Any]:
        """Choose the best parameters with K-fold cross-validation.

        Note:
            It uses a customized version of the keras-tuner library's `Tuner`,
            making each trial use cross validation to evaluate the performance of the model.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.
            batch_size: Batch size to use for the training.
            max_trials: Maximum number of hypermarameters combinations to evaluate.

        Returns:
            A dictionary containing the average scores across
            all CV validation partitions and best parameters.

        """
        dataset = self._preprocessor.preprocess_fit(X, y, batch_size=batch_size)
        dataset_spec = self._preprocessor.get_dataset_spec(dataset)
        tuner = CrossValidationTuner(
            hypermodel=self._model_factory(dataset_spec),
            oracle=BayesianOptimization(objective="val_loss", max_trials=max_trials),
            project_name="tuner_checkpoints",
            overwrite=True,
        )
        tuner.search(dataset=dataset)
        hp = tuner.get_best_hyperparameters()[0]
        self._model = tuner.hypermodel.build(hp, dataset)
        self.fit(X, y)
        return {**hp.values, "scores": self.evaluate(X, y)}

    @staticmethod
    def _model_factory(dataset_spec: TensorSpec) -> KerasHyperModel:
        """Create a tunable model, with default values, compatible with the modified tuner.

        Args:
            dataset_spec: metadata on the features, to determine the input layers.

        Returns:
            A customized version of the keras-tuner library's `KerasHyperModel`,
            taking an optional `dataset` argument at build time
            (used to pre-fit the encoding layers).

        """
        return CustomKerasHyperModel(TunableNeuralNetwork(dataset_spec))
