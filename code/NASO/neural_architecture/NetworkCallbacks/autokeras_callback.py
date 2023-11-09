import math

import tensorflow as tf
from loguru import logger

from helper_scripts.timer import Timer
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import TrainingMetric


class AutoKerasCallback(tf.keras.callbacks.Callback):
    def __init__(self, celery_task, run: AutoKerasRun):
        super().__init__()
        self.celery_task = celery_task
        self.run = run
        # i need the epochs here from the run.
        self.timer = Timer()

    def get_total_time(self):
        return round(self.timer.get_total_time(), 2)

    def on_epoch_begin(self, epoch, logs=[], *args):
        # start the timer here
        if "metrics" not in logs:
            logs["metrics"] = 0
        for metric_name in self.run.model.metric_weights:
            if metric_name in logs:
                logs["metrics"] += (
                    logs[metric_name] * self.run.model.metric_weights[metric_name]
                )

        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        epochs = epoch + 1

        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": epochs,
                "run_id": self.run.id,
                "metrics": metrics,
                "autokeras": True,
            },
        )

        # resume the timer here
        self.timer.start()

    def on_epoch_end(self, epoch, logs=[], *args):
        # stop the timer here:
        epoch_time = self.timer.stop()

        logs["execution_time"] = epoch_time
        logs["total_time"] = self.timer.get_total_time()

        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": epoch + 1,
                "run_id": self.run.id,
                "metrics": metrics,
                "autokeras": True,
            },
        )

    def on_train_end(self, logs=None):
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

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
        self.celery_task.update_state(
            state="SUCCESS",
            meta={
                "current": 1,
                "total": 1,
                "run_id": self.run.id,
                "metrics": metrics,
                "autokeras": True,
            },
        )
        logger.info("Autokeras training ended ")
