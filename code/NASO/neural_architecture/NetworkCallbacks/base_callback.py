import math

import tensorflow as tf

from helper_scripts.timer import Timer
from neural_architecture.models.autokeras import AutoKerasRun
from neural_architecture.models.model_runs import KerasModelRun
from runs.models.training import NetworkTraining, TrainingMetric


class BaseCallback(tf.keras.callbacks.Callback):
    def __init__(self, celery_task, run, epochs):
        super().__init__()
        self.celery_task = celery_task
        # i need the epochs here from the run.
        self.timer = Timer()
        self.epochs = epochs
        self.run = run

    def get_total_time(self):
        return round(self.timer.get_total_time(), 2)

    def on_epoch_begin(self, epoch, logs=None):
        # start the timer here
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": self.epochs,
                "run_id": self.run.id,
                "metrics": metrics,
                "autokeras_trial": isinstance(self.run, KerasModelRun),
                "autokeras": isinstance(self.run, AutoKerasRun),
            },
        )

        # resume the timer here
        self.timer.start()

    def on_epoch_end(self, epoch, logs=None):
        # stop the timer here:
        elapsed_time = self.timer.stop()
        logs["execution_time"] = elapsed_time
        logs["total_time"] = self.timer.get_total_time()

        # so stop timesrs and then resume afterwards
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

        self.celery_task.update_state(
            state="PROGRESS",
            meta={
                "current": (epoch + 1),
                "total": self.epochs,
                "run_id": self.run.id,
                "metrics": metrics,
                "autokeras_trial": isinstance(self.run, KerasModelRun),
                "autokeras": isinstance(self.run, AutoKerasRun),
            },
        )
