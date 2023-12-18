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
    """
    A class representing a Keras model, whitch is in this case a model from a specific trial of an autokeras run.

    Attributes:
        model_file (CharField): The file path of the saved model.
        name (CharField): The name of the model.
        description (CharField): The description of the model.
        size (IntegerField): The number of parameters in the model.
        metrics (ManyToManyField): The metrics used for evaluation.
        evaluation_parameters (ForeignKey): The evaluation parameters for the model.
        fit_parameters (ForeignKey): The fit parameters for training the model.
        saving_prefix (str): The prefix used for saving the model.
        model (None or Model): The Keras model object.

    Methods:
        __init__: Initializes the KerasModel object.
        __str__: Returns the name of the model.
        set_model: Sets the Keras model and saves it.
        save_model: Saves the Keras model.
        get_model: Returns the Keras model.
        load_model: Loads the Keras model from the saved file.
        fit: Trains the Keras model.
        evaluate: Evaluates the Keras model.
        predict: Makes predictions using the Keras model.
    """

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
        """
        Sets the Keras model and saves it.

        Args:
            model: The Keras model object.

        Raises:
            ValueError: If the model does not have a fit, predict, or evaluate method.
        """
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
        compile_args = {
            "loss": self.model.loss,
            "optimizer": self.model.optimizer,
            "metrics": self.model.metrics,
        }
        self.model = self.build_pruning_model(self.model)
        self.model.compile(**compile_args)

    def save_model(self):
        """
        Saves the Keras model.
        """
        if not self.model:
            self.model = self.get_export_model(self.load_model())
        if not os.path.exists(self.saving_prefix):
            os.makedirs(self.saving_prefix)
        self.model.save(self.model_file, overwrite=True)

    def get_model(self):
        """
        Returns the Keras model.

        Returns:
            The Keras model object.
        """
        return self.model

    def load_model(self):
        """
        Loads the Keras model from the saved file.

        Returns:
            The loaded Keras model object.

        Raises:
            ValueError: If the model file does not exist.
        """
        # check if file exists:
        if not os.path.exists(self.model_file):
            raise ValueError(f"model {self.model_file} does not exist")
        model = load_model(self.model_file)
        return self.build_pruning_model(model)

    def fit(self, *args, **kwargs):
        """
        Trains the Keras model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The training history.

        Notes:
            If "epochs" is not provided in kwargs, it will be set to the default value from fit_parameters.
            If "callbacks" is not provided in kwargs, it will be set to the pruning callbacks.

        """
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
        """
        Evaluates the Keras model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The evaluation result.

        """
        if not self.model:
            self.model = self.load_model()
        return self.model.evaluate(*args, **kwargs)

    def predict(self, dataset, run: "KerasModelRun"):
        """
        Makes predictions using the Keras model.

        Args:
            dataset: The dataset for making predictions.
            run: The KerasModelRun object.

        Returns:
            The predictions.

        """
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
    """
    Represents a run of a Keras model.

    Attributes:
        model (ForeignKey): The Keras model associated with this run.
        metrics (ManyToManyField): The training metrics for this run.
        prediction_metrics (ManyToManyField): The prediction metrics for this run.
    """

    model = models.ForeignKey(KerasModel, on_delete=models.CASCADE)
    metrics = models.ManyToManyField(TrainingMetric, related_name="kerasmodel_metrics")
    prediction_metrics = models.ManyToManyField(
        TrainingMetric,
        related_name="keras_model_prediction_metrics",
    )

    def get_energy_consumption(self):
        energy = 0
        for metric in self.metrics.all():
            energy += metric.get_energy_consumption()
        return energy
