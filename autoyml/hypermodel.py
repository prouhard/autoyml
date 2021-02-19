import gc
import traceback
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from kerastuner import HyperModel, HyperParameters
from kerastuner import config as keras_tuner_config_module
from kerastuner.engine.hypermodel import (
    KerasHyperModel,
    maybe_compute_model_size,
    maybe_distribute,
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import (
    CategoryEncoding,
    Normalization,
    StringLookup,
)
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.ops.gen_dataset_ops import BatchDataset


class TunableNeuralNetwork(HyperModel):
    """A subclass of the keras-tuner's `HyperModel`
    which can adapt its input architecture based on the training data.

    The input and encoding layers construction is inspired from the official keras documentation
    (https://keras.io/examples/structured_data/structured_data_classification_from_scratch/)
    but modified to adapt to any dataframe.

    """

    def __init__(self, features_spec: TensorSpec) -> None:
        """Instanciate the (unbuilt) model with the training data metadata.

        Args:
            features_spec: A tensor of metadata, used later to determine the input shape.

        Returns:
            None.

        """
        self._features_spec = features_spec

    def build(self, hp: HyperParameters, dataset: Optional[tf.data.Dataset] = None) -> keras.Model:
        """Main method, mandatory for the tuner.

        Note:
            keras-tuner usually does not accept extra arguments besides the hyperparameters.
            But the input and encoding layers of each model will vary depending on the training data.
            This is why we need to pass the dataset at build time.

        Args:
            hp: An instance of `HyperParameters`, automatically passed by the tuner.
                Pass `hp=HyperParameters()` to use the fiexed default values.
            dataset: The training data.

        Returns:
            A keras functional model, with tunable hyperparameters.

        """
        input_layer = self._build_input_layer()
        encoded_layer = self._build_encoded_layer(input_layer, dataset)

        return self._build_model(hp, input_layer, encoded_layer)

    def _build_input_layer(self) -> List[KerasTensor]:
        """Build the input layer according to the features metadata.

        Returns:
            A list of individual input layers,
            named and typed according to their corresponding individual features.

        """
        input_features = {
            feature_name: keras.Input(shape=(1,), name=feature_name, dtype=feature_value.dtype)
            for feature_name, feature_value in self._features_spec.items()
        }

        return [input_features[feature_name] for feature_name in self._features_spec]

    def _build_encoded_layer(self, input_layer: KerasTensor, dataset: tf.data.Dataset) -> KerasTensor:
        """Build the encoding layers and adapt them to the training data.

        It only distinguishes categorical and numeric features according to their type.
        Numeric features are normalized, and categorical features are one-hot encoded.

        Args:
            input_layer: The input layer, needed to connect the graph.
            dataset: Input features.

        Returns:
            A concatenated layer of all the encoded features.

        """
        encoded_features = [
            self._encode_numerical_feature(input_feature, input_feature.name, dataset)
            if pd.api.types.is_numeric_dtype(self._features_spec[input_feature.name].dtype.as_numpy_dtype)
            else self._encode_categorical_feature(input_feature, input_feature.name, dataset)
            for input_feature in input_layer
        ]

        return layers.concatenate(encoded_features)

    @staticmethod
    def _build_model(
        hp: HyperParameters,
        input_layer: KerasTensor,
        encoded_layer: KerasTensor,
    ) -> keras.Model:
        """Build the part of the architecture tunable by keras-tuner.

        Note:
            It is a relatively simple dense network, with self-normalizing layers.

        Args:
            hp: hyperparameters passed by the tuner.
            input layer: The input layer of the model.
            encoded_layer: The encoding layer of the model.

        Returns:
            A tunable keras functional model.

        """
        x = encoded_layer
        for i in range(hp.Int("dense_layers", 1, 3, default=2)):
            x = layers.Dense(
                units=hp.Int(f"units_layer_{i + 1}", min_value=32, max_value=256, step=32, default=64),
                activation="selu",
                kernel_initializer=tf.keras.initializers.LecunNormal(),
            )(encoded_layer)
            x = layers.AlphaDropout(0.5)(x)

        output_layer = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(input_layer, output_layer)
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-3)),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5, name="f1_score"),
            ],
        )

        return model

    @staticmethod
    def _encode_numerical_feature(
        feature: KerasTensor,
        name: str,
        dataset: Optional[BatchDataset],
    ) -> KerasTensor:
        """Normalize numerical features.

        Args:
            - feature: The input layer of the feature.
            - name: The feature's name (its column name in the original dataframe).
            - dataset: The training data, if not specified, return a no-op layer.

        Returns:
            The normalized tensor of the input feature.

        """
        # Return generic layer for the tuner initialization
        if not dataset:
            return KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None))

        # Create a Normalization layer for our feature
        normalizer = Normalization()

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the statistics of the data
        normalizer.adapt(feature_ds)

        # Normalize the input feature
        encoded_feature = normalizer(feature)

        return encoded_feature

    @staticmethod
    def _encode_categorical_feature(
        feature: KerasTensor,
        name: str,
        dataset: Optional[BatchDataset],
    ) -> KerasTensor:
        """One-hot encode categorical features.

        Args:
            - feature: The input layer of the feature.
            - name: The feature's name (its column name in the original dataframe).
            - dataset: The training data, if not specified, return a no-op layer.

        Returns:
            The one-hot encoded tensor of the input feature.

        """
        # Return generic layer for the tuner initialization
        if not dataset:
            return KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None))

        # Create a StringLookup layer which will turn strings into integer indices
        index = StringLookup()

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the set of possible string values and assign them a fixed integer index
        index.adapt(feature_ds)

        # Turn the string input into integer indices
        encoded_feature = index(feature)

        # Create a CategoryEncoding for our integer indices
        encoder = CategoryEncoding(output_mode="binary")

        # Learn the space of possible indices
        encoder.adapt(np.arange(index.vocab_size()))

        # Apply one-hot encoding to our indices{split + 1} / {n_splits}
        encoded_feature = encoder(encoded_feature)

        return encoded_feature


class CustomKerasHyperModel(KerasHyperModel):
    """Override the default `build` method to enable multiple parameters.

    Only the method signature (1) and one line (2) are changed:
        (1) def build(self, hp) -> def build(self, hp, *args, **kwargs)
        (2) self.hypermodel.build(hp) - > self.hypermodel.build(hp, *args, **kwargs)

    """

    def build(self, hp: HyperParameters, *args: Any, **kwargs: Any) -> keras.Model:
        for i in range(self._max_fail_streak + 1):
            # clean-up TF graph from previously stored (defunct) graph
            keras.backend.clear_session()
            gc.collect()

            # Build a model, allowing max_fail_streak failed attempts.
            try:
                with maybe_distribute(self.distribution_strategy):
                    # /!\ Below line is the only one changed compared to the original version: /!\
                    # model = self.hypermodel.build(hp)
                    model = self.hypermodel.build(hp, *args, **kwargs)
            except:  # noqa: E722 do not use bare 'except'
                if keras_tuner_config_module.DEBUG:
                    traceback.print_exc()

                print("Invalid model %s/%s" % (i, self._max_fail_streak))

                if i == self._max_fail_streak:
                    raise RuntimeError("Too many failed attempts to build model.")
                continue

            # Stop if `build()` does not return a valid model.
            if not isinstance(model, keras.models.Model):
                raise RuntimeError(
                    "Model-building function did not return " "a valid Keras Model instance, found {}".format(model)
                )

            # Check model size.
            size = maybe_compute_model_size(model)
            if self.max_model_size and size > self.max_model_size:
                print("Oversized model: %s parameters -- skipping" % (size))
                if i == self._max_fail_streak:
                    raise RuntimeError("Too many consecutive oversized models.")
                continue
            break

        return self._compile_model(model)
