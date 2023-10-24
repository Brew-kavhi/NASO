import math

import tensorflow as tf
from loguru import logger

from helper_scripts.Timer import Timer
from neural_architecture.models.AutoKeras import AutoKerasRun
from runs.models.Training import TrainingMetric


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
        self.timer.stop()
        # so stop timesrs and then resume afterwards
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

        metric = TrainingMetric(
            epoch=epoch,
            metrics=[
                {
                    "current": epoch,
                    "run_id": self.run.id,
                    "metrics": metrics,
                    "time": self.timer.get_total_time(),
                },
            ],
        )
        metric.save()
        self.run.metrics.add(metric)

    def on_train_begin(self, logs=None):
        logger.info("Aurtokeras training began ")

    def on_train_end(self, logs=None):
        print("training end")
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
        logger.info("Aurtokeras training ended ")
