import os
from typing import Any

from django.db import models
from keras.models import load_model

from neural_architecture.models.model_optimization import PrunableNetwork
from neural_architecture.NetworkCallbacks.evaluation_base_callback import (
    EvaluationBaseCallback,
)
from runs.models.training import (
    EvaluationParameters,
    FitParameters,
    Metric,
    Run,
    TrainingMetric,
)


class KerasModel(PrunableNetwork):
    model_file = models.CharField(max_length=100)
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=200)
    size = models.IntegerField(default=-1)
    metrics = models.ManyToManyField(Metric)
    evaluation_parameters = models.ForeignKey(
        EvaluationParameters, on_delete=models.CASCADE
    )
    fit_parameters = models.ForeignKey(FitParameters, on_delete=models.CASCADE)
    saving_prefix = "keras_models/"
    model = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.saving_prefix += self.name
        self.model_file = f"{self.saving_prefix}/{self.name}.keras"

    def __str__(self):
        return self.name

    def set_model(self, model):
        # this model needs to have the fit, predict and evvaluate methods
        if (
            not hasattr(model, "fit")
            or not hasattr(model, "predict")
            or not hasattr(model, "evaluate")
        ):
            raise ValueError("model does not have a fit, predict or evaluation method")

        self.model = model

        if not os.path.exists(self.saving_prefix):
            os.makedirs(self.saving_prefix)

        # now save the model and set the model_file attribute:
        self.model.save(self.model_file, overwrite=True)
        self.size = self.model.count_params()
        self.save()
        self.model = self.build_pruning_model(self.model)

    def save_model(self):
        if not self.model:
            self.model = self.get_export_model(self.load_model())
        if not os.path.exists(self.saving_prefix):
            os.makedirs(self.saving_prefix)
        self.model.save(self.model_file, overwrite=True)

    def get_model(self):
        return self.model

    def load_model(self):
        # check if file exists:
        if not os.path.exists(self.model_file):
            raise ValueError(f"model {self.model_file} does not exist")
        model = load_model(self.model_file)
        return self.build_pruning_model(model)

    def fit(self, *args, **kwargs):
        if not self.model:
            self.model = self.load_model()
        if "epochs" not in kwargs:
            kwargs["epochs"] = self.fit_parameters.epochs
        if "callbacks" not in kwargs:
            kwargs["callbacks"] = self.get_pruning_callbacks()
        else:
            kwargs["callbacks"] += self.get_pruning_callbacks()
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if not self.model:
            self.model = self.load_model()
        return self.model.evaluate(*args, **kwargs)

    def predict(self, dataset, run: "KerasModelRun"):
        if not self.model:
            self.model = self.load_model()
        batch_size = 1
        return self.model.predict(
            dataset,
            batch_size,
            verbose=2,
            steps=None,
            callbacks=self.fit_parameters.get_callbacks(run)
            + [EvaluationBaseCallback(run)],
        )


class KerasModelRun(Run):
    model = models.ForeignKey(KerasModel, on_delete=models.CASCADE)
    metrics = models.ManyToManyField(TrainingMetric, related_name="kerasmodel_metrics")
    prediction_metrics = models.ManyToManyField(
        TrainingMetric,
        related_name="keras_model_prediction_metrics",
    )
