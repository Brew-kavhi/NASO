# from django.db import transaction
import numpy as np
import tensorflow.keras.backend as K
from autokeras import tuners

from neural_architecture.models.types import AutoKerasTunerType
from plugins.interfaces.commands import InstallerInterface

# @transaction.atomic


class Installer(InstallerInterface):
    def install(self):
        # Implement the installation logic here
        try:
            # Create instances in the database
            AutoKerasTunerType.objects.create(
                module_name=self.get_module_name(),
                name="BaysianOptimizationTrial",
                native_tuner=False,
                required_arguments=[],
            )
            # You can add more database operations or any other setup logic here

            print("Plugin installed successfully.")
        except Exception as e:
            # Handle any exceptions that may occur during installation
            print(f"Plugin installation failed: {str(e)}")
        print("Installation completed")

    def uninstall(self):
        # Implement the uninstallation logic here
        AutoKerasTunerType.objects.filter(module_name=self.get_module_name()).delete()
        print("Uninstallation completed")


class BaysianOptimizationTrial(tuners.BayesianOptimization):
    """BayesianOptimization tuning with Gaussian process. and a callback on the trial end

    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted. Defaults to 10.
        num_initial_points: Optional number of randomly generated samples as
            initial training data for Bayesian optimization. If left
            unspecified, a value of 3 times the dimensionality of the
            hyperparameter space is used.
        alpha: Float, the value added to the diagonal of the kernel matrix
            during fitting. It represents the expected amount of noise in the
            observed performances in Bayesian optimization. Defaults to 1e-4.
        beta: Float, the balancing factor of exploration and exploitation. The
            larger it is, the more explorative it is. Defaults to 2.6.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are
            invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
            number of consecutive failed `Trial`s. When this number is reached,
            the search will be stopped. A `Trial` is marked as failed when none
            of the retries succeeded.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=10,
        num_initial_points=None,
        alpha=1e-4,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            num_initial_points,
            alpha,
            beta,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            max_retries_per_trial,
            max_consecutive_failed_trials,
            **kwargs,
        )

    def on_epoch_end(self, trial, model, epoch, logs=None):
        print("epoch ended")
        logs["model_size"] = int(
            np.sum([K.count_params(w) for w in model.trainable_weights])
        )

        print(logs)
        super().on_epoch_end(trial, model, epoch, logs)

    def on_trial_end(self, *args):
        print(args)
        print("trial ended")
        print(self.search_space_summary())
        super().on_trial_end(*args)


import tensorflow as tf


class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="custom_metric", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.int32, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.math.squared_difference(y_pred, y_true)
        count = tf.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values *= sample_weight
            count *= sample_weight
        self.sum.assign_add(tf.reduce_sum(values))
        self.count.assign_add(count)

    def result(self):
        return self.sum / tf.cast(self.count, tf.float32)

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)
