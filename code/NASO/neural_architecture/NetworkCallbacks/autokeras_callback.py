import math

import tensorflow as tf
from loguru import logger

from helper_scripts.timer import Timer
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import TrainingMetric


class AutoKerasCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for AutoKeras training. Saves metrics to the database

    Args:
        celery_task: The Celery task associated with the training.
        run: The AutoKerasRun object representing the current run.

    Attributes:
        celery_task: The Celery task associated with the training.
        run: The AutoKerasRun object representing the current run.
        timer: Timer object to measure the total training time.

    Methods:
        on_train_end: Callback method called at the end of training.

    """

    def __init__(self, celery_task, run: AutoKerasRun):
        super().__init__()
        self.celery_task = celery_task
        self.run = run
        self.timer = Timer()

    def on_train_begin(self, logs=None):
        self.timer.start()

    def on_train_end(self, logs=None):
        """
        Callback method called at the end of training.

        Args:
            logs: Dictionary containing the training metrics.

        """
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]
        self.timer.stop()
        metric = TrainingMetric(
            epoch=0,
            metrics=[
                {
                    "final_metric": True,
                    "run_id": self.run.id,
                    "metrics": metrics,
                    "time": self.timer.get_total_time(),
                },
            ],
        )
        metric.save()
        self.run.metrics.add(metric)
        logger.info("Autokeras training ended ")
