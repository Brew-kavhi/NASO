import math

import tensorflow as tf
from loguru import logger

from helper_scripts.Timer import Timer
from runs.models.Training import NetworkTraining, TrainingMetric


class CeleryUpdateCallback(tf.keras.callbacks.Callback):
    additional_callbacks = None
    def __init__(self, celery_task, run: NetworkTraining):
        super().__init__()
        self.celery_task = celery_task
        self.run = run
        # i need the epochs here from the run.
        self.timer = Timer()
        if run:
            self.additional_callbacks = run.fit_parameters.get_callbacks()

    def get_total_time(self):
        return round(self.timer.get_total_time(), 2)

    def on_epoch_begin(self, epoch, logs=None):
        # start the timer here
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        epochs = epoch
        if self.run:
            epochs = self.run.fit_parameters.epochs
        run_id = 0
        if self.run:
            run_id = self.run.id

        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": epochs,
                "run_id": run_id,
                "metrics": metrics,
            },
        )
        if self.additional_callbacks:
            for callback in self.additional_callbacks:
                try:
                    callback.on_epoch_begin(epoch, logs)
                except Exception as e:
                    logger.critical(
                        f"Callback {callback} threw an exception in epoch {epoch} begin:{e}"
                    )

        # resume the timer here
        self.timer.start()

    def on_epoch_end(self, epoch, logs=None):
        # stop the timer here:
        self.timer.stop()
        # so stop timesrs and then resume afterwards
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        if self.run:
            metric = TrainingMetric(
                neural_network=self.run,
                epoch=epoch,
                metrics=[
                    {
                        "current": epoch,
                        "total": self.run.fit_parameters.epochs,
                        "run_id": self.run.id,
                        "metrics": metrics,
                        "time": self.timer.get_total_time(),
                    },
                ],
            )
            metric.save()
        if self.additional_callbacks:
            for callback in self.additional_callbacks:
                try:
                    callback.on_epoch_end(epoch, logs)
                except Exception as e:
                    logger.critical(
                        f"Callback {callback} threw an exception in epoch {epoch} end:{e}"
                    )
