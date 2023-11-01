# Define a decorator function that wraps the on_epoch_end method
import math

from runs.models.Training import TrainingMetric


def custom_on_epoch_end_decorator(original_on_epoch_end, run):
    def on_epoch_end(self, trial, model, epoch, logs=None):
        if "model_size" not in logs:
            logs["model_size"] = model.count_params()
        if "trial_id" not in logs:
            logs["trial_id"] = int(trial.trial_id)

        metrics = {}
        for key in logs:
            if not math.isnan(logs[key]):
                metrics[key] = logs[key]

        metric = TrainingMetric(
            epoch=epoch,
            metrics=[
                {
                    "current": epoch,
                    "run_id": run.id,
                    "metrics": metrics,
                    "trial_id": logs["trial_id"],
                },
            ],
        )
        metric.save()
        run.metrics.add(metric)

        if original_on_epoch_end:
            original_on_epoch_end(self, trial, model, epoch, logs)

    return on_epoch_end


def custom_on_epoch_begin_decorator(original_on_epoch_begin):
    def on_epoch_begin(self, trial, model, epoch, logs=None):
        if "model_size" not in logs:
            logs["model_size"] = model.count_params()
        if "trial_id" not in logs:
            logs["trial_id"] = int(trial.trial_id)

        if original_on_epoch_begin:
            original_on_epoch_begin(self, trial, model, epoch, logs)

    return on_epoch_begin


# Define a decorator function that wraps the on_epoch_end method
def custom_on_trial_end_decorator(original_on_trial_end):
    def on_trial_end(self, trial):
        if original_on_trial_end:
            original_on_trial_end(self, trial)

    return on_trial_end


def custom_on_trial_begin_decorator(original_on_trial_begin):
    def on_trial_begin(self, trial):
        if original_on_trial_begin:
            original_on_trial_begin(self, trial)

    return on_trial_begin
