import math

from django.core.exceptions import ValidationError
from django.db import models

from helper_scripts.git import get_current_git_hash
from helper_scripts.importing import get_callback, get_object
from naso.settings import APP_VERSION
from neural_architecture.models.Architecture import NetworkConfiguration
from neural_architecture.models.Dataset import Dataset
from neural_architecture.models.Types import (LossType, MetricType,
                                              OptimizerType, TypeInstance)
from neural_architecture.validators import validate_dtype


class Optimizer(TypeInstance):
    instance_type = models.ForeignKey(
        OptimizerType, on_delete=models.deletion.DO_NOTHING
    )
    weight_decay = models.FloatField(default=0.0)
    clipnorm = models.FloatField(null=True, blank=True)
    clipvalue = models.FloatField(null=True, blank=True)
    global_clipnorm = models.FloatField(null=True, blank=True)
    use_ema = models.BooleanField(default=False)
    ema_momentum = models.FloatField(default=0.99)
    ema_overwrite_frequency = models.IntegerField(null=True)
    jit_compile = models.BooleanField(default=True)


class LossFunction(TypeInstance):
    instance_type = models.ForeignKey(LossType, on_delete=models.deletion.DO_NOTHING)


class Metric(TypeInstance):
    instance_type = models.ForeignKey(MetricType, on_delete=models.deletion.DO_NOTHING)
    name = models.CharField(max_length=100)
    dtype = models.CharField(max_length=20, validators=[validate_dtype])


class NetworkHyperparameters(models.Model):
    # this should be extendable by plugins
    optimizer = models.ForeignKey(
        Optimizer, on_delete=models.deletion.CASCADE, null=True
    )
    loss = models.ForeignKey(LossFunction, on_delete=models.deletion.CASCADE, null=True)
    metrics = models.ManyToManyField(Metric)
    run_eagerly = models.BooleanField(default=False)
    steps_per_execution = models.IntegerField(default=1)
    jit_compile = models.BooleanField(default=False)

    def get_as_dict(self):
        model_optimizer = "adam"
        if self.optimizer:
            model_optimizer = get_object(
                self.optimizer.instance_type.module_name,
                self.optimizer.instance_type.name,
                self.optimizer.additional_arguments,
            )

        model_metrics = ["accuracy"]
        if self.metrics:
            model_metrics = []
            for metric in self.metrics.all():
                model_metrics.append(
                    get_object(
                        metric.instance_type.module_name,
                        metric.instance_type.name,
                        metric.additional_arguments,
                    )
                )
        model_metrics = ["accuracy"]

        model_loss = "sparse_categorical_crossentropy"
        if self.loss:
            model_loss = get_object(
                self.loss.instance_type.module_name,
                self.loss.instance_type.name,
                self.loss.additional_arguments,
            )

        return {
            "optimizer": model_optimizer,
            "loss": model_loss,
            "metrics": model_metrics,
            "run_eagerly": self.run_eagerly,
            "steps_per_execution": self.steps_per_execution,
            "jit_compile": self.jit_compile,
        }


class EvaluationParameters(models.Model):
    batch_size = models.IntegerField(default=32)
    steps = models.IntegerField(null=True)
    callbacks = models.JSONField(null=True)

    def get_callbacks(self):
        callbacks = []
        if not self.callbacks:
            return []
        for callback in self.callbacks:
            if "module_name" in callback and "class_name" in callback:
                callbacks.append(get_callback(callback))
        return callbacks

    def validate_callbacks_data(self):
        if not isinstance(self.callbacks, list):
            raise ValidationError(
                "JSON data for EvaluationParameters.callbacks should be a list of objects."
            )

        for item in self.callbacks:
            if (
                not isinstance(item, dict)
                or "module_name" not in item
                or "class_name" not in item
            ):
                raise ValidationError(
                    'Each item in the JSON list for EvaluationParameters.callbacks should be an object with a "module_name" attribute and a "class_name" attribute.'
                )
            if "additional_arguments" in item:
                for argument in item["additional_arguments"]:
                    if (
                        not isinstance(argument, dict)
                        or "name" not in argument
                        or "value" not in argument
                    ):
                        print(argument)
                        raise ValidationError(
                            "Each argument in the JSON list should be a dict with a name and a value."
                        )

    def save(self, *args, **kwargs):
        self.validate_callbacks_data()
        super(EvaluationParameters, self).save(*args, **kwargs)


class FitParameters(models.Model):
    batch_size = models.IntegerField(null=True)
    epochs = models.IntegerField()
    verbose = models.BinaryField(max_length=2, default=2)
    callbacks = models.JSONField(null=True)

    shuffle = models.BooleanField(default=True)
    # TODO implement this
    class_weight = models.JSONField(null=True)
    sample_weight = models.JSONField(null=True)
    initial_epoch = models.IntegerField(default=0)
    steps_per_epoch = models.IntegerField(null=True)

    max_queue_size = models.IntegerField(default=10)
    workers = models.IntegerField(default=1)
    use_multiprocessing = models.BooleanField(default=False)

    def get_callbacks(self):
        callbacks = []
        if self.callbacks:
            for callback in self.callbacks:
                print(callback)
                if "module_name" in callback and "class_name" in callback:
                    callbacks.append(get_callback(callback))
        return callbacks

    # TODO duplicate code, unify
    def validate_callbacks_data(self):
        if not isinstance(self.callbacks, list):
            raise ValidationError(
                "JSON data for FitParameters.callbacks should be a list of objects."
            )

        for item in self.callbacks:
            if (
                not isinstance(item, dict)
                or "module_name" not in item
                or "class_name" not in item
            ):
                raise ValidationError(
                    'Each item in the JSON list for FitParameters.callbacks should be an object with a "module_name" attribute and a "class_name" attribute.'
                )
            if "additional_arguments" in item:
                for argument in item["additional_arguments"]:
                    if (
                        not isinstance(argument, dict)
                        or "name" not in argument
                        or "value" not in argument
                    ):
                        print(argument)
                        raise ValidationError(
                            "Each argument in the JSON list should be a dict with a name and a value."
                        )

    def save(self, *args, **kwargs):
        self.validate_callbacks_data()
        super(FitParameters, self).save(*args, **kwargs)


class NetworkTraining(models.Model):
    network_config = models.ForeignKey(
        NetworkConfiguration, on_delete=models.deletion.DO_NOTHING
    )
    dataset = models.ForeignKey(
        Dataset, on_delete=models.deletion.DO_NOTHING, null=True, blank=True
    )
    hyper_parameters = models.ForeignKey(
        NetworkHyperparameters, on_delete=models.deletion.CASCADE
    )
    evaluation_parameters = models.ForeignKey(
        EvaluationParameters, on_delete=models.deletion.CASCADE
    )

    fit_parameters = models.ForeignKey(FitParameters, on_delete=models.deletion.CASCADE)
    final_metrics = models.ForeignKey(
        "TrainingMetric", on_delete=models.deletion.CASCADE, null=True
    )

    naso_app_version = models.CharField(max_length=10, default=APP_VERSION)

    git_hash = models.CharField(max_length=40, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.git_hash:
            self.git_hash = get_current_git_hash()
        super(NetworkTraining, self).save(*args, **kwargs)


class TrainingMetric(models.Model):
    # Reference to the neural network model, if needed
    neural_network = models.ForeignKey(NetworkTraining, on_delete=models.CASCADE)

    # Epoch number
    epoch = models.PositiveIntegerField()

    # JSONField to store metrics as a dictionary
    metrics = models.JSONField()

    def validate_json_data(self):
        if not isinstance(self.metrics, list):
            raise ValidationError(
                "JSON data for TrainingMetric.metrics should be a list of objects."
            )

        for item in self.metrics:
            if not isinstance(item, dict) or "metrics" not in item:
                raise ValidationError(
                    'Each item in the JSON list for TrainingMetric.metrics should be an object with a "metrics" attribute.'
                )
            if not isinstance(item["metrics"], dict):
                raise ValidationError(
                    "TrainingMetrics.metrics.metrics should a dictionary of metricsname and value"
                )
            for key in item["metrics"]:
                if math.isnan(item["metrics"][key]):
                    item["metrics"][key] = None

    def save(self, *args, **kwargs):
        self.validate_json_data()
        super(TrainingMetric, self).save(*args, **kwargs)

    def __str__(self):
        return f"Neural Network {self.neural_network} - Epoch {self.epoch}"
