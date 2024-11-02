import asyncio
from jtop import jtop
from tensorflow.keras.optimizers import Adam

import keras
import numpy as np

from helper_scripts.power_management import get_cpu_power_usage, get_gpu_power_usage
from neural_architecture.NetworkCallbacks.logging_callback import LoggingCallback
from neural_architecture.NetworkCallbacks.timing_callback import TimingCallback
from runs.models.training import Run, TrainingMetric

K = keras.backend

def reconnect_jtop(jetson):
    try:
        if jetson:
            jetson.close()  # Close the previous connection if any
        jetson = jtop()    # Reconnect to jtop
        jetson.start()
        print("Reconnected to jtop server.")
        return jetson
    except Exception as e:
        print(f"Failed to reconnect to jtop: {e}")

def start_async_measuring(stop_event, run: Run, database_lock):
    task = asyncio.run(measure_power(stop_event, run))
    with database_lock:
        run.power_measurements = ",".join([str(power) for power in task])
        run.save()
    return task

def start_async_measurement_thread(coroutine, event, run, queue, interval=2, jetson=None):
    loop = asyncio.new_event_loop()  # Create a new event loop in this thread
    asyncio.set_event_loop(loop)     # Set the event loop for this thread
    loop.run_until_complete(coroutine(event, run, queue, interval, jetson))


async def measure_power(stop_event, run: Run, queue, interval, jetson):
    power = []
    await asyncio.sleep(0.001)
    if not jetson:
        jetson = reconnect_jtop(None)
    try:
        while not stop_event.is_set() and len(power) < 3:
            if jetson.ok():
                power.append(jetson.power['rail']['VDD_CPU_GPU_CV']['power']/1000)
            print(f"Power measurement async, {interval}")
            await asyncio.sleep(interval)
    except Exception as e:
        reconnect_jtop(jetson)
    print("stop event is set")
    queue.put(power)
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
            run.model.trial_model = {
                "model": model,
                "trial_id": trial.trial_id,
                "metrics": metric.metrics,
            }
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
def custom_on_trial_end_decorator(original_on_trial_end, run, train_data, val_data):
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

        inference_model = run.model.get_export_model(run.model.trial_model["model"])
        if run.model.clustering_options:
            inference_model = run.model.clustering_options.get_cluster_export_model(
                inference_model
            )

        batch_size = 32

        (_, dataset) = prepare_data_for_trial(
            train_data, val_data, trial, run.model.auto_model, batch_size
        )

        callbacks = (
            [TimingCallback()] + run.model.get_callbacks(run) + [LoggingCallback()]
        )

        inference_model.evaluate(
            dataset,
            batch_size=batch_size,
            verbose=2,
            steps=None,
            callbacks=callbacks,
            return_dict=True,
        )

        val_metrics = callbacks[-1].logs[0]
        train_metrics = run.model.trial_model["metrics"][0]["metrics"]

        log_metrics = val_metrics
        for metric in train_metrics:
            if metric not in log_metrics:
                log_metrics[metric] = train_metrics[metric]

        metric = TrainingMetric(
            epoch=run.model.epochs,
            metrics=[
                {
                    "current": run.model.epochs,
                    "run_id": run.id,
                    "metrics": log_metrics,
                    "trial_id": trial.trial_id,
                },
            ],
        )
        metric.save()
        run.inference_metrics.add(metric)

        score = 0
        if run.model.objective == "metrics" and hasattr(run.model, "metric_weights"):
            for metric_name in run.model.metric_weights:
                if metric_name in log_metrics and log_metrics[metric_name]:
                    score += (log_metrics[metric_name]) * (
                        run.model.metric_weights[metric_name]
                    )
            trial.score = score
        else:
            if run.model.objective in log_metrics:
                trial.score = log_metrics[run.model.objective]

    return on_trial_end


def prepare_data_for_trial(
    train_dataset, test_dataset, trial, auto_model, batch_size=1
) -> tuple:
    """
    Prepares the data for a trial in AutoKeras.

    Args:
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        trial_id: The ID of the trial.

    Returns:
        A tuple containing the pipeline, hyperparameters, trial data, and validation data.
    """

    # convert dataset to a shape , that autokeras can use.
    dataset, validation_data = auto_model._convert_to_dataset(
        train_dataset, y=None, validation_data=test_dataset, batch_size=batch_size
    )
    (
        _,
        dataset,
        validation_data,
    ) = auto_model.tuner._prepare_model_build(
        trial.hyperparameters, x=dataset, validation_data=validation_data
    )
    return (dataset, validation_data)


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

            model = run.model.build_pruning_model(model, include_last_layer=False)
            if run.model.clustering_options:
                model = run.model.clustering_options.build_clustered_model(
                    model, include_last_layer=False
                )
            model.loss = loss
            model.optimizer = Adam()
            model.compile(Adam(), loss)

            return model
        raise ValueError("No build function provided")

    return build_fn
