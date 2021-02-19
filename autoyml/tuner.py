from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from kerastuner.engine.trial import Trial
from kerastuner.engine.tuner import Tuner

from autoyml.preprocessing import DataPreprocessor


class CrossValidationTuner(Tuner):
    """Overrides `run_trial` to enable cross-validation on `BatchDataset` objects.

    See https://github.com/keras-team/keras-tuner/issues/122 for the motivation.
    """

    def run_trial(
        self,
        trial: Trial,
        dataset: tf.data.Dataset,
        n_splits: int = 5,
    ) -> None:
        """Evaluate the current set of hypermarameters with cross-validation.

        Args:
            trial: A Trial instance passed by the tuner, with the hyperparameters.
            dataset: The training data.
            n_splits: The number of folds to use, defaults to 5.

        Returns:
            None

        """
        val_losses = []
        shuffled_dataset = dataset.shuffle(buffer_size=len(dataset))
        shards = [shuffled_dataset.shard(n_splits, i) for i in range(n_splits)]
        for split in range(n_splits):
            dataset_train, dataset_val = self._cv_concatenate(shards, split)
            model = self.hypermodel.build(trial.hyperparameters, dataset_train)
            print(f"Fitting model (CV {split + 1} / {n_splits})...")
            class_weight = DataPreprocessor.get_class_weight(dataset_train)
            model.fit(dataset_train, class_weight=class_weight)
            print(f"Evaluating model (CV {split + 1} / {n_splits})...")
            val_losses.append(model.evaluate(dataset_val))
        self.oracle.update_trial(trial.trial_id, {"val_loss": np.mean(val_losses)})
        self.save_model(trial.trial_id, model)

    def _cv_concatenate(self, shards: Sequence[tf.data.Dataset], index: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Isolate the validation batches (at position `index`),
        and concatenate the other ones in one dataset.

        Args:
            shards: A list of batches.
            index: The batches used for validation.

        Returns:
            The training and validation batches as a tuple.

        """
        if index == 0:
            return self._concatenate(shards[1:]), shards[0]
        elif index == len(shards) - 1:
            return self._concatenate(shards[:-1]), shards[-1]
        else:
            return self._concatenate(shards[:index]).concatenate(self._concatenate(shards[index + 1 :])), shards[index]

    @staticmethod
    def _concatenate(datasets: Sequence[tf.data.Dataset]) -> tf.data.Dataset:
        """Concatenate a sequence of tensorflow datasets.

        Args:
            datasets: The datasets to concatenate.

        Returns:
            A concatenated dataset.

        """
        dataset, *others = datasets
        for other in others:
            dataset = dataset.concatenate(other)
        return dataset
