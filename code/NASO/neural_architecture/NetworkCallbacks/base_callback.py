import math
import types

import tensorflow as tf

from api.views.metrics import TensorflowMetricAPIView as MetricAPIView
from helper_scripts.pruning import calculate_sparsity, collect_prunable_layers
from helper_scripts.timer import Timer
from inference.models.inference import Inference
from neural_architecture.helper_scripts.architecture import calculate_flops
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining
from workers.helper_scripts.celery import get_compute_device_name

K = tf.keras.backend


class BaseCallback(tf.keras.callbacks.Callback):
    """
    Base class for custom callbacks in a neural network training process.
    responsible for updating the Celery task state with the current epoch number and the training metrics, saviong the metrics to teh database, and measuring the execution time.

    Args:
        celery_task (object): The Celery task object.
        run (object): The run object.
        epochs (int): The total number of epochs.

    Attributes:
        celery_task (object): The Celery task object.
        timer (Timer): The timer object for measuring execution time.
        epochs (int): The total number of epochs.
        run (object): The run object.
        gpu_consumption (str): The GPU consumption value.

    Methods:
        get_total_time(): Returns the total execution time.
        on_epoch_begin(epoch, logs): Callback function called at the beginning of each epoch.
        on_epoch_end(epoch, logs): Callback function called at the end of each epoch.
    """

    def __init__(self, celery_task, run, epochs, batch_size):
        super().__init__()
        self.celery_task = celery_task
        self.timer = Timer()
        self.epochs = epochs
        self.run = run
        self.batch_size = batch_size
        self.gpu_consumption = "NaN"
        try:
            device = self.get_physical_device(run.gpu)
            if device:
                run.compute_device = get_compute_device_name(device)
            run.save()
        except ValueError:
            print("not available")

    def get_physical_device(self, device_name):
        all_devices = tf.config.list_physical_devices()
        for device in all_devices:
            if device.name.split("physical_device:")[1] == device_name:
                return device
        return None

    def get_total_time(self):
        """
        Returns the total execution time.

        Returns:
            float: The total execution time.
        """
        return round(self.timer.get_total_time(), 2)

    def on_train_begin(self, logs=None):
        # Collect all the prunable layers in the model.
        self.prunable_layers = collect_prunable_layers(self.model)
        if not self.prunable_layers:
            return
        # If the model is newly created/initialized, set the 'pruning_step' to 0.
        # If the model is saved and then restored, do nothing.
        if self.prunable_layers[0].pruning_step == -1:
            tuples = []
            mask_update_ops = []
            for layer in self.prunable_layers:
                tuples.append((layer.pruning_step, 0))
                if tf.executing_eagerly():
                    layer.conditional_mask_update()
                else:
                    mask_update_ops.append(layer.conditional_mask_update())
            K.batch_set_value(tuples)
            K.batch_get_value(mask_update_ops)
        if self.model:
            logs["flops"] = calculate_flops(self.model, self.batch_size)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Callback function called at the beginning of each epoch.
        Updates the Celery task state with the current epoch number and the training metrics.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training metrics for the current epoch.
        """
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": self.epochs,
                "gpu": {"device": self.run.gpu, "power": self.gpu_consumption},
                "run_id": self.run.id,
                "metrics": metrics,
                "autokeras": isinstance(self.run, AutoKerasRun),
                "inference": isinstance(self.run, Inference),
            },
        )

        self.timer.start()

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training metrics for the current epoch.
        """

        if self.model:
            # call prune –weights:
            weight_mask_ops = []
            for layer in self.prunable_layers:
                if tf.executing_eagerly():
                    if not epoch == self.epochs - 1:
                        layer.conditional_mask_update()
                    layer.weight_mask_op()
                else:
                    if not epoch == self.epochs - 1:
                        weight_mask_ops.append(layer.conditional_mask_update())
                    weight_mask_ops.append(layer.weight_mask_op())

            K.batch_get_value(weight_mask_ops)
            logs["sparsity"] = calculate_sparsity(self.model)
            logs["flops"] = calculate_flops(self.model, self.batch_size)
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        if isinstance(self.run, NetworkTraining):
            api_view = MetricAPIView()
            api_view.post(
                types.SimpleNamespace(
                    **{
                        "data": {
                            "epoch": epoch,
                            "metrics": [
                                {
                                    "final_metric": False,
                                    "trial_id": 0,
                                    "current": epoch,
                                    "total": self.epochs,
                                    "run_id": self.run.id,
                                    "metrics": metrics,
                                    "time": self.timer.get_total_time(),
                                },
                            ],
                        }
                    }
                ),
                self.run.id,
            )

        if "power_consumption" in logs:
            self.gpu_consumption = logs["power_consumption"]
        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": self.epochs,
                "run_id": self.run.id,
                "metrics": metrics,
                "gpu": {"device": self.run.gpu, "power": self.gpu_consumption},
                "autokeras": isinstance(self.run, AutoKerasRun),
                "inference": isinstance(self.run, Inference),
            },
        )
