import os
from typing import Any

from django.db import models
from keras.models import load_model

from runs.models.training import (
    EvaluationParameters,
    FitParameters,
    Metric,
    Run,
    TrainingMetric,
)


class KerasModel(models.Model):
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

    def get_model(self):
        return self.model

    def load_model(self):
        # check if file exists:
        if not os.path.exists(self.model_file):
            raise ValueError(f"model {self.model_file} does not exist")
        return load_model(self.model_file)

    def fit(self, *args, **kwargs):
        if not self.model:
            self.model = self.load_model()
        if "epochs" not in kwargs:
            kwargs["epochs"] = self.fit_parameters.epochs
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if not self.model:
            self.model = self.load_model()
        if "epochs" not in kwargs:
            kwargs["epochs"] = self.fit_parameters.epochs
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if not self.model:
            self.model = self.load_model()
        return self.model.evaluate(*args, **kwargs)


class KerasModelRun(Run):
    model = models.ForeignKey(KerasModel, on_delete=models.CASCADE)
    metrics = models.ManyToManyField(TrainingMetric, related_name="kerasmodel_metrics")
