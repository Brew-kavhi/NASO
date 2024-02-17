import math
import time

import django
import tensorflow as tf

from helper_scripts.extensions import calculate_sparsity
from helper_scripts.timer import Timer
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_runs import KerasModelRun
from runs.models.training import NetworkTraining, TrainingMetric
from inference.models.inference import Inference


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

    def __init__(self, celery_task, run, epochs):
        super().__init__()
        self.celery_task = celery_task
        self.timer = Timer()
        self.epochs = epochs
        self.run = run
        self.gpu_consumption = "NaN"
        try:
            device = self.get_physical_device(run.gpu)
            if device:
                run.compute_device = self.get_compute_device_name(device)
            run.save()
        except ValueError:
            print("not available")

    def get_physical_device(self, device_name):
        all_devices = tf.config.list_physical_devices()
        for device in all_devices:
            if device.name.split("physical_device:")[1] == device_name:
                return device
        return None

    def get_compute_device_name(self, device):
        details = tf.config.experimental.get_device_details(device)
        if "device_name" in details:
            return details["device_name"]
        return ""

    def get_total_time(self):
        """
        Returns the total execution time.

        Returns:
            float: The total execution time.
        """
        return round(self.timer.get_total_time(), 2)

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
                "autokeras_trial": isinstance(self.run, KerasModelRun),
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

        logs["sparsity"] = calculate_sparsity(self.model)
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        if isinstance(self.run, NetworkTraining):
            metric = TrainingMetric(
                neural_network=self.run,
                epoch=epoch,
                metrics=[
                    {
                        "current": epoch,
                        "total": self.epochs,
                        "run_id": self.run.id,
                        "metrics": metrics,
                        "time": self.timer.get_total_time(),
                    },
                ],
            )
            # Database could be locked. In that case, wait and try again
            try:
                metric.save()
            except django.db.utils.OperationalError:
                time.sleep(0.5)
                metric.save()
        elif isinstance(self.run, KerasModelRun):
            metric = TrainingMetric(
                epoch=epoch,
                metrics=[
                    {
                        "current": epoch,
                        "total": self.epochs,
                        "run_id": self.run.id,
                        "metrics": metrics,
                        "time": self.timer.get_total_time(),
                        "autokeras_trial": True,
                    },
                ],
            )
            metric.save()
            self.run.metrics.add(metric)

        if "energy_consumption" in logs:
            self.gpu_consumption = logs["energy_consumption"]
        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": self.epochs,
                "run_id": self.run.id,
                "metrics": metrics,
                "gpu": {"device": self.run.gpu, "power": self.gpu_consumption},
                "autokeras_trial": isinstance(self.run, KerasModelRun),
                "autokeras": isinstance(self.run, AutoKerasRun),
                "inference": isinstance(self.run, Inference),
            },
        )
