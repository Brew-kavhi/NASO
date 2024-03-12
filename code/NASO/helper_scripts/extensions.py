import asyncio

import numpy as np
import tensorflow as tf

from helper_scripts.power_management import get_cpu_power_usage, get_gpu_power_usage
from runs.models.training import Run, TrainingMetric

keras = tf.keras
K = keras.backend


def start_async_measuring(stop_event, run: Run, database_lock):
    task = asyncio.run(measure_power(stop_event, run))
    with database_lock:
        run.power_measurements = ",".join([str(power) for power in task])
        run.save()
    return task


async def measure_power(stop_event, run: Run):
    power = []
    while not stop_event.is_set():
        if run.gpu.startswith("GPU"):
            power.append(get_gpu_power_usage(run.gpu))
        elif run.gpu.startswith("CPU"):
            power.append(get_cpu_power_usage(run.gpu))
        await asyncio.sleep(2)
    return power


def custom_on_epoch_end_decorator(original_on_epoch_end, run):
    """
    This function is a decorator for the on_epoch_end function of the hyper tuner.
    It extends the on_epoch_end function with the functionality to save the model size and the metrics to the database.

    Args:
        original_on_epoch_end (function): The original on_epoch_end function.
        run (Run): The run object.

    Returns:
        function: The decorated on_epoch_end function.
    """

    def on_epoch_end(self, trial, model, epoch, logs=None):
        if "model_size" not in logs:
            logs["model_size"] = int(
                np.sum([K.count_params(w) for w in model.trainable_weights])
            )

        if "metrics" not in logs:
            logs["metrics"] = 0
        if hasattr(run.model, "metric_weights"):
            for metric_name in run.model.metric_weights:
                if metric_name in logs:
                    logs["metrics"] += (logs[metric_name]) * (
                        run.model.metric_weights[metric_name]
                    )

        metrics = {}
        for key in logs:
            metrics[key] = logs[key]
        metrics["trial_id"] = trial.trial_id

        metric = TrainingMetric(
            epoch=epoch,
            metrics=[
                {
                    "current": epoch,
                    "run_id": run.id,
                    "metrics": metrics,
                    "trial_id": trial.trial_id,
                },
            ],
        )
        metric.save()
        run.metrics.add(metric)

        if original_on_epoch_end:
            original_on_epoch_end(self, trial, model, epoch, logs)

    return on_epoch_end


def custom_on_epoch_begin_decorator(original_on_epoch_begin):
    """
    This function is a decorator for the on_epoch_begin function of the hyper tuner.
    It extends the on_epoch_begin function with the functionality to save the model size and the trial id to the logs.

    Args:
        original_on_epoch_begin (function): The original on_epoch_begin function.

    Returns:
        function: The decorated on_epoch_begin function.
    """

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        if "model_size" not in logs:
            logs["model_size"] = int(
                np.sum([K.count_params(w) for w in model.trainable_weights])
            )
        if "trial_id" not in logs:
            logs["trial_id"] = trial.trial_id

        if original_on_epoch_begin:
            original_on_epoch_begin(self, trial, model, epoch, logs)

    return on_epoch_begin


# Define a decorator function that wraps the on_epoch_end method
def custom_on_trial_end_decorator(original_on_trial_end):
    """
    This function is a decorator for the on_trial_end function of the hyper tuner.

    Args:
        original_on_trial_end (function): The original on_trial_end function.

    Returns:
        function: The decorated on_trial_end function.
    """

    def on_trial_end(self, trial):
        if original_on_trial_end:
            original_on_trial_end(self, trial)

    return on_trial_end


def custom_on_trial_begin_decorator(original_on_trial_begin):
    """
    This function is a decorator for the on_trial_begin function of the hyper tuner.

    Args:
        original_on_trial_begin (function): The original on_trial_begin function.

    Returns:
        function: The decorated on_trial_begin function.
    """

    def on_trial_begin(self, trial):
        if original_on_trial_begin:
            original_on_trial_begin(self, trial)

    return on_trial_begin


def custom_hypermodel_build(original_build_fn, run):
    """
    This function is a decorator for the build function of the hyper model.
    It extends the build function with the functionality to prune the model.

    Args:
        original_build_fn (function): The original build function.
        run (Run): The run object.

    Returns:
        function: The decorated build function.
    """

    def build_fn(hyper_parameters):
        if original_build_fn:
            model = original_build_fn(hyper_parameters)
            loss = model.loss
            optimizer = model.optimizer

            model = run.model.build_pruning_model(model, include_last_layer=False)
            if run.model.clustering_options:
                model = run.model.clustering_options.build_clustered_model(
                    model, include_last_layer=False
                )
            model.loss = loss
            model.optimizer = optimizer
            model.compile(optimizer, loss)

            return model
        raise Exception("No build function provided")

    return build_fn
