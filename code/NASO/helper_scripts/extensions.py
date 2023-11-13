from runs.models.training import TrainingMetric


def custom_on_epoch_end_decorator(original_on_epoch_end, run):
    def on_epoch_end(self, trial, model, epoch, logs=None):
        if "model_size" not in logs:
            logs["model_size"] = model.count_params()

        if "metrics" not in logs:
            logs["metrics"] = 0
        if hasattr(run.model, "metric_weights"):
            for metric_name in run.model.metric_weights:
                if metric_name in logs:
                    logs["metrics"] += (logs[metric_name]) * (
                        run.model.metric_weights[metric_name]
                    )

        metrics = {}
        for key in logs:
            metrics[key] = logs[key]
        metrics["trial_id"] = trial.trial_id

        metric = TrainingMetric(
            epoch=epoch,
            metrics=[
                {
                    "current": epoch,
                    "run_id": run.id,
                    "metrics": metrics,
                    "trial_id": trial.trial_id,
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
            logs["trial_id"] = trial.trial_id

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


def custom_hypermodel_build(original_build_fn, run):
    def build_fn(hp):
        if original_build_fn:
            model = original_build_fn(hp)
            loss = model.loss
            optimizer = model.optimizer

            model = run.model.build_pruning_model(model)
            model.loss = loss
            model.optimizer = optimizer

            return model
        raise Exception("No build function provided")

    return build_fn
