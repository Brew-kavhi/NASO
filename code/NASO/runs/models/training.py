import math

from django.core.exceptions import ValidationError
from django.db import models

from helper_scripts.git import get_current_git_hash
from helper_scripts.importing import get_object
from naso.settings import APP_VERSION
from neural_architecture.models.architecture import NetworkConfiguration
from neural_architecture.models.dataset import Dataset
from neural_architecture.models.types import (
    CallbackType,
    LossType,
    MetricType,
    OptimizerType,
    TypeInstance,
)
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


class CallbackFunction(TypeInstance):
    instance_type = models.ForeignKey(
        CallbackType, on_delete=models.deletion.DO_NOTHING
    )


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
                self.optimizer.instance_type.required_arguments,
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
                        metric.instance_type.required_arguments,
                    )
                )

        model_loss = "sparse_categorical_crossentropy"
        if self.loss:
            model_loss = get_object(
                self.loss.instance_type.module_name,
                self.loss.instance_type.name,
                self.loss.additional_arguments,
                self.loss.instance_type.required_arguments,
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
    callbacks = models.ManyToManyField(
        CallbackFunction, related_name="evaluation_callbacks"
    )

    def get_callbacks(self, run: "NetworkTraining"):
        callbacks = []
        for callback in self.callbacks.all():
            if callback.instance_type.name == "EnergyCallback":
                callbacks.append(
                    get_object(
                        callback.instance_type.module_name,
                        callback.instance_type.name,
                        callback.additional_arguments + [{"name": "run", "value": run}],
                        callback.instance_type.required_arguments,
                    )
                )
            else:
                callbacks.append(
                    get_object(
                        callback.instance_type.module_name,
                        callback.instance_type.name,
                        callback.additional_arguments,
                        callback.instance_type.required_arguments,
                    )
                )
        return callbacks


class FitParameters(models.Model):
    batch_size = models.IntegerField(null=True)
    epochs = models.IntegerField()
    callbacks = callbacks = models.ManyToManyField(
        CallbackFunction, related_name="fitparameters_callbacks"
    )

    shuffle = models.BooleanField(default=True)
    # TODO implement this
    class_weight = models.JSONField(null=True)
    sample_weight = models.JSONField(null=True)
    initial_epoch = models.IntegerField(default=0)
    steps_per_epoch = models.IntegerField(null=True)

    max_queue_size = models.IntegerField(default=10)
    workers = models.IntegerField(default=1)
    use_multiprocessing = models.BooleanField(default=False)

    def get_callbacks(self, run: "NetworkTraining"):
        callbacks = []
        for callback in self.callbacks.all():
            if callback.instance_type.name == "EnergyCallback":
                callbacks.append(
                    get_object(
                        callback.instance_type.module_name,
                        callback.instance_type.name,
                        callback.additional_arguments + [{"name": "run", "value": run}],
                        callback.instance_type.required_arguments,
                    )
                )
            else:
                callbacks.append(
                    get_object(
                        callback.instance_type.module_name,
                        callback.instance_type.name,
                        callback.additional_arguments,
                        callback.instance_type.required_arguments,
                    )
                )
        return callbacks


class Run(models.Model):
    """
    A base model for training runs.

    Attributes:
        naso_app_version (str): The version of the NASO app used for the run.
        git_hash (str): The hash of the Git commit used for the run.
        dataset (Dataset): The dataset used for the run.
    """

    naso_app_version = models.CharField(max_length=10, default=APP_VERSION)

    git_hash = models.CharField(max_length=40, blank=True)

    dataset = models.ForeignKey(
        Dataset, on_delete=models.deletion.SET_NULL, null=True, blank=True
    )

    prediction_metrics = models.ManyToManyField(
        "TrainingMetric",
        related_name="tensorflow_prediction_metrics",
    )

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        """
        Saves the current instance of the Run model to the database. If the git_hash attribute is not set, it will be
        set to the current git hash using the get_current_git_hash() function. This method overrides the default save
        method of the parent model.
        """
        if not self.git_hash:
            self.git_hash = get_current_git_hash()
        super().save(*args, **kwargs)


class NetworkTraining(Run):
    """
    A class representing a network training run.

    Attributes:
        network_config (NetworkConfiguration): The configuration of the network being trained.
        hyper_parameters (NetworkHyperparameters): The hyperparameters used for training the network.
        evaluation_parameters (EvaluationParameters): The parameters used for evaluating the trained network.
        fit_parameters (FitParameters): The parameters used for fitting the network to the training data.
        final_metrics (TrainingMetric): The final metrics obtained after training the network.
    """

    network_config = models.ForeignKey(
        NetworkConfiguration, on_delete=models.deletion.DO_NOTHING
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
    prediction_metrics = models.ManyToManyField(
        "TrainingMetric",
        related_name="tensorflow_prediction_metrics",
    )

    def __str__(self):
        return self.network_config.name


class TrainingMetric(models.Model):
    """
    A model to store training metrics for a neural network model.

    Attributes:
        neural_network (NetworkTraining): A reference to the neural network model.
        epoch (int): The epoch number.
        metrics (dict): A dictionary of metrics names and their corresponding values.

    Methods:
        validate_json_data(): Validates that the metrics data is in the correct format.
    """

    neural_network = models.ForeignKey(
        NetworkTraining, on_delete=models.CASCADE, null=True
    )
    epoch = models.PositiveIntegerField()
    metrics = models.JSONField()

    def validate_json_data(self):
        """
        Validates the JSON data for the TrainingMetric object.

        Raises:
            ValidationError: If the metrics attribute is not a list of objects, or if any item in the list
                does not have a "metrics" attribute that is a dictionary of metric names and values.
        """

        if not isinstance(self.metrics, list):
            raise ValidationError(
                "JSON data for TrainingMetric.metrics should be a list of objects."
            )

        for item in self.metrics:
            if not isinstance(item, dict) or "metrics" not in item:
                raise ValidationError(
                    """Each item in the JSON list for TrainingMetric.metrics should be an 
                    object with a "metrics" attribute."""
                )
            if not isinstance(item["metrics"], dict):
                raise ValidationError(
                    "TrainingMetrics.metrics.metrics should a dictionary of metricsname and value"
                )
            for metric in item["metrics"]:
                if isinstance(item["metrics"][metric], float) and math.isnan(
                    item["metrics"][metric]
                ):
                    item["metrics"][metric] = None

    def save(self, *args, **kwargs):
        self.validate_json_data()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Neural Network {self.neural_network} - Epoch {self.epoch}"
