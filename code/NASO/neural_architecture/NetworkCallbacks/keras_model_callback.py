import math

import tensorflow as tf

from helper_scripts.timer import Timer
from neural_architecture.models.model_runs import KerasModelRun
from runs.models.training import TrainingMetric


class KerasModelCallback(tf.keras.callbacks.Callback):
    additional_callbacks = None

    def __init__(self, celery_task, run: KerasModelRun):
        super().__init__()
        self.celery_task = celery_task
        self.run = run

        self.timer = Timer()

    def get_total_time(self):
        return round(self.timer.get_total_time(), 2)

    def on_epoch_begin(self, epoch, logs=None):
        # resume the timer here
        self.timer.start()

    def on_epoch_end(self, epoch, logs=None):
        # stop the timer here:
        epoch_time = self.timer.stop()

        logs["execution_time"] = epoch_time
        logs["total_time"] = self.timer.get_total_time()

        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        if self.run:
            metric = TrainingMetric(
                epoch=epoch,
                metrics=[
                    {
                        "current": epoch,
                        "total": self.run.model.fit_parameters.epochs,
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
                    "total": self.run.model.fit_parameters.epochs,
                    "run_id": self.run.id,
                    "metrics": metrics,
                    "autokeras_trial": True,
                },
            )
