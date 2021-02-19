from typing import Dict, cast

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.tensor_spec import TensorSpec


class DataPreprocessor:
    """A collection of methods used to transform and prepare data to be fed to a Keras model."""

    @classmethod
    def preprocess_fit(cls, X: pd.DataFrame, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
        """Prepare data for the end-to-end training.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.
            batch_size: Batch size to use for the training.

        Returns:
            A tuple of `tf.data.Dataset` object of constant batch size.

        """
        return cls._dataframe_to_dataset(cls._handle_missing_values(X), y, batch_size)

    @classmethod
    def preprocess_predict(cls, X: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare data for the prediction.

        Note:
            Keras models handle dictionaries of `pd.Series` natively.

        Args:
            X: Input features.

        Returns:
            A dictionary of the form {column_name[str]: column_values[pd.Series]}.

        """
        return dict(cls._handle_missing_values(X))

    @classmethod
    def get_class_weight(cls, dataset: tf.data.Dataset) -> Dict[int, float]:
        """Assign weights according to the proportion of positive samples.

        Note:
            Used for weighting the loss function during training only.

        Args:
            dataset: Input features and target as `tf.data.Dataset`.

        Returns:
            A dictionary containing the class weights to be used during fitting.

        """
        return {0: 1, 1: 1 / cls._get_target_mean(dataset) - 1}

    @staticmethod
    def get_dataset_spec(dataset: tf.data.Dataset) -> TensorSpec:
        """Get metadata on the features in the dataset.

        Args:
            dataset: Input features and target.

        Returns:
            A tensor containing feature names and dtypes.

        """
        # `.element_spec` is a tuple of the form (features spec, target spec).
        return dataset.element_spec[0]

    @staticmethod
    def _dataframe_to_dataset(X: pd.DataFrame, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
        """Convert a pd.DataFrame of features and its associated labels to the tf.data.Dataset format.

        Args:
            X: Input features.
            y: Ground truth labels as a numpy array of 0-s and 1-s.
            batch_size: Batch size to use for the training.

        Returns:
            A tensorflow `tf.data.Dataset` containing the input feature and the target,
            a variant of the tf.data.Dataset iterable by batches of fixed size.
            Each batch is a tuple of features and target.

        """
        features = X.copy()
        target = y.copy()
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), target))
        dataset = dataset.shuffle(buffer_size=len(features))
        dataset = dataset.batch(batch_size)
        return dataset

    @staticmethod
    def _handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
        """Replace all NaN values in `X` with zeroes or empty strings.

        Notes:
            Keras models cannot handle missing values.
            This is a basic way to handle missing data,
            feel free to override this method.

        Args:
            X: The dataframe to be filled.

        Returns:
            A copy of the dataframe, without missing values.
        """
        dataframe = X.copy()
        for column in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                dataframe[column] = dataframe[column].fillna(0)
            else:
                dataframe[column] = dataframe[column].fillna("")
        return dataframe

    @staticmethod
    def _get_target_mean(dataset: tf.data.Dataset) -> float:
        """Compute the mean of the target in a `tf.data.Dataset` object.

        Args:
            dataset: a `tf.data.Dataset`, iterable of the form (features, target)

        Returns:
            The mean of the target on all the batches.

        """

        def get_target_sum(dataset: tf.data.Dataset) -> np.int64:
            return dataset.reduce(np.int64(0), lambda x, y: x + tf.math.reduce_sum(y[1]))

        def get_target_length(dataset: tf.data.Dataset) -> np.int64:
            return tf.cast(dataset.reduce(np.int32(0), lambda x, y: x + len(y[1])), np.int64)

        return cast(float, get_target_sum(dataset) / get_target_length(dataset).numpy())
