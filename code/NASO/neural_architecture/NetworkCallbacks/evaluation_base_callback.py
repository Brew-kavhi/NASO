import math
import types

import tensorflow as tf

from api.views.metrics import MetricsAPIView as InferenceMetricsAPIView
from api.views.metrics import TensorflowMetricAPIView
from helper_scripts.timer import Timer
from inference.models.inference import Inference
from runs.models.training import NetworkTraining, Run, TrainingMetric


class EvaluationBaseCallback(tf.keras.callbacks.Callback):
    """
    Callback for evaluating the execution time of testing and prediction in a neural network model.
    Logs the execution time of each batch and the average execution time of the testing/prediction.
    Furthrmore, it writes the prediction metrics to the database.

    Attributes:
        times (list): List to store the execution times of each batch.
        run (Run): The run object associated with the callback.
        timer (Timer): Timer object for measuring execution time.

    Methods:
        on_test_begin(logs=None): Called at the beginning of testing.
        on_test_end(logs=None): Called at the end of testing.
        on_test_batch_begin(batch, logs=None): Called at the beginning of each testing batch.
        on_test_batch_end(batch, logs=None): Called at the end of each testing batch.
        on_predict_begin(logs=None): Called at the beginning of prediction.
        on_predict_end(logs=None): Called at the end of prediction.
        on_predict_batch_begin(batch, logs=None): Called at the beginning of each prediction batch.
        on_predict_batch_end(batch, logs=None): Called at the end of each prediction batch.
    """

    times = []
    _batch = 0

    def __init__(self, run: Run | Inference):
        super().__init__()
        self.run = run
        self.timer = Timer()

    def on_test_begin(self, logs=None):
        """
        Called at the beginning of testing.

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.times = []

    def on_test_end(self, logs=None):
        """
        Called at the end of testing.
        Logs the execution time of the testing

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        logs["execution_time"] = sum(self.times) / len(self.times)
        keys = list(logs.keys())
        print(f"Stop testing; got log keys: {keys}")

    def on_test_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each testing batch.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.timer.start()

    def on_test_batch_end(self, batch, logs=None):
        """
        Called at the end of each testing batch.
        Logs the execution time of the batch.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        elapsed_time = self.timer.stop()
        self.times.append(elapsed_time)
        logs["execution_time"] = self.timer.get_total_time()

    def on_predict_begin(self, logs=None):
        """
        Called at the beginning of prediction.

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.times = []

    def on_predict_end(self, logs=None):
        """
        Called at the end of prediction.
        Logs the execution time of the prediction and writes metrics to the database

        Args:
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        logs["total_batches"] = self._batch
        if self.run.gpu.startswith("GPU"):
            logs["memory_consumption"] = tf.config.experimental.get_memory_info(
                self.run.gpu
            )["current"]
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]
        if self.run:
            if isinstance(self.run, Inference):
                metric_api = InferenceMetricsAPIView()
                metric_api.post(
                    types.SimpleNamespace(
                        **{
                            "data": {
                                "epoch": 0,
                                "metrics": [
                                    {
                                        "final_metric": False,
                                        "current": 0,
                                        "run_id": self.run.id,
                                        "metrics": metrics,
                                        "time": None,
                                    },
                                ],
                            }
                        }
                    ),
                    self.run.id,
                )
            elif isinstance(self.run, NetworkTraining):
                metric_api = TensorflowMetricAPIView()
                metric_api.post(
                    types.SimpleNamespace(
                        **{
                            "data": {
                                "epoch": 0,
                                "metrics": [
                                    {
                                        "final_metric": False,
                                        "current": 0,
                                        "run_id": self.run.id,
                                        "metrics": metrics,
                                        "time": None,
                                    },
                                ],
                            }
                        }
                    ),
                    self.run.id,
                    1,
                )
            else:
                metric = TrainingMetric(
                    epoch=0,
                    metrics=[
                        {
                            "run_id": self.run.id,
                            "metrics": metrics,
                        },
                    ],
                )
                metric.save()
                self.run.prediction_metrics.add(metric)

    def on_predict_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each prediction batch.
        Starts the time for measuring the execution time.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        self.timer.start()

    def on_predict_batch_end(self, batch, logs=None):
        """
        Called at the end of each prediction batch.
        Logs the execution time of the batch.

        Args:
            batch: The batch index.
            logs (dict): Dictionary of logs.

        Returns:
            None
        """
        elapsed_time = self.timer.stop()
        self.times.append(elapsed_time)
        logs["execution_time"] = self.timer.get_total_time()
        # add one because of the index things
        self._batch = batch + 1
