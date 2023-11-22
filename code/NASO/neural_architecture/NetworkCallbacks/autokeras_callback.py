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
        logger.info("Autokeras training ended ")
