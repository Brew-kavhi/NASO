import math

import tensorflow as tf

from helper_scripts.timer import Timer
from runs.models.training import Run, TrainingMetric


class EvaluationBaseCallback(tf.keras.callbacks.Callback):
    times = []

    def __init__(self, run: Run):
        super().__init__()
        self.run = run
        self.timer = Timer()

    def on_test_begin(self, logs=None):
        self.times = []

    def on_test_end(self, logs=None):
        logs["execution_time"] = sum(self.times) / len(self.times)
        keys = list(logs.keys())
        print(f"Stop testing; got log keys: {keys}")

    def on_test_batch_begin(self, batch, logs=None):
        self.timer.start()

    def on_test_batch_end(self, batch, logs=None):
        elapsed_time = self.timer.stop()
        self.times.append(elapsed_time)
        logs["execution_time"] = self.timer.get_total_time()

    def on_predict_begin(self, logs=None):
        self.times = []

    def on_predict_end(self, logs=None):
        logs["execution_time"] = sum(self.times) / len(self.times)
        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]
        if self.run:
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
        self.timer.start()

    def on_predict_batch_end(self, batch, logs=None):
        elapsed_time = self.timer.stop()
        self.times.append(elapsed_time)
        logs["execution_time"] = self.timer.get_total_time()
