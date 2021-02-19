import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from autoyml.errors import NotFittedError
from autoyml.neural_network_model import NeuralNetworkModel


def test_reproducibility() -> None:
    """Test that the predicitons of the model remain the same with a fixed seed."""
    seed = 1234
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X = pd.DataFrame({"feat1": ["a", "b", "a"], "feat2": [1, 2, 3]})
    y = np.array([0, 0, 1])

    nn_model = NeuralNetworkModel()
    nn_model.fit(X, y, epochs=1)

    probabilities = nn_model.predict_proba(X)

    assert np.allclose(
        probabilities,
        np.array([0.30143574, 0.5373031, 0.54928374], dtype=np.float32),
    )


def test_missing_values_handling() -> None:
    """Test that the model can handle missing values."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", np.nan],
            "feat2": [np.nan, 2, 3],
            "feat3": [np.nan, 2.1, 3.2],
        }
    )
    y = np.array([0, 0, 1])

    nn_model = NeuralNetworkModel()
    nn_model.fit(X, y, epochs=1)
    predictions = nn_model.predict(X)

    assert all(not np.isnan(prediction) for prediction in predictions)


def test_new_categories_handling() -> None:
    """Test that the model can handle new categories at prediction time."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", "c"],
            "feat2": [1, 2, 3],
        }
    )
    y = np.array([0, 0, 1])

    nn_model = NeuralNetworkModel()
    nn_model.fit(X, y, epochs=1)

    X_test = pd.DataFrame(
        {
            "feat1": ["d", "e", "c"],
            "feat2": [1, 2, 3],
        }
    )
    predictions = nn_model.predict(X_test)

    assert all(not np.isnan(prediction) for prediction in predictions)


def test_predict_output_format() -> None:
    """Test that the model returns the predictions in the desired format."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", "c"],
            "feat2": [1, 2, 3],
        }
    )
    y = np.array([0, 0, 1])

    nn_model = NeuralNetworkModel()
    nn_model.fit(X, y, epochs=1)

    predictions = nn_model.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.int32
    assert len(predictions) == 3
    assert all(prediction in [0, 1] for prediction in predictions)


def test_predict_proba_output_format() -> None:
    """Test that the model returns the predicted probabilities in the desired format."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", "c"],
            "feat2": [1, 2, 3],
        }
    )
    y = np.array([0, 0, 1])

    nn_model = NeuralNetworkModel()
    nn_model.fit(X, y, epochs=1)

    probabilities = nn_model.predict_proba(X)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.dtype == np.float32
    assert len(probabilities) == 3
    assert all(0 <= probability <= 1 for probability in probabilities)


def test_raise_error_if_not_fitted() -> None:
    """Test that the model raises a `NotFittedError` when trying to predict before fitting."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", "c"],
            "feat2": [1, 2, 3],
        }
    )

    nn_model = NeuralNetworkModel()

    with pytest.raises(NotFittedError):
        nn_model.predict(X)


def test_evaluate_format() -> None:
    """Test that the metrics are restitued in the correct format."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", "c"],
            "feat2": [1, 2, 3],
        }
    )
    y = np.array([0, 0, 1])

    nn_model = NeuralNetworkModel()
    nn_model.fit(X, y, epochs=1)

    metrics = nn_model.evaluate(X, y)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"f1_score", "logloss"}


def test_tune_parameters() -> None:
    """Test that the optimal parameters are restitued in the correct format."""
    X = pd.DataFrame(
        {
            "feat1": ["a", "b", "c", "a", "b"],
            "feat2": [1, 2, 3, 4, 5],
        }
    )
    y = np.array([0, 0, 1, 0, 1])

    nn_model = NeuralNetworkModel()

    best_parameters = nn_model.tune_parameters(X, y, batch_size=1, max_trials=1)

    assert isinstance(best_parameters, dict)
    assert "scores" in best_parameters
    assert set(best_parameters["scores"].keys()) == {"f1_score", "logloss"}
