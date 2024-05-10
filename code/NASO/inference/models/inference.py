import os

import tensorflow as tf
from django.db import models

from helper_scripts.importing import get_object
from neural_architecture.helper_scripts.architecture import calculate_flops
from neural_architecture.models.dataset import Dataset
from neural_architecture.NetworkCallbacks.timing_callback import TimingCallback
from runs.models.training import (
    CallbackFunction,
    Metric,
    NetworkTraining,
    TrainingMetric,
)

from sklearn.metrics import accuracy_score
import numpy as np


class Inference(models.Model):
    """
    Stores an inference run of a given model file. Works bz loading a model
    file and then do prediction on a given dataset. The metrics obtained in here
    are stored in the databse
    """

    model_file = models.CharField(max_length=256, default="")
    name = models.CharField(max_length=100)
    description = models.TextField()
    dataset = models.ForeignKey(
        Dataset, on_delete=models.deletion.SET_NULL, null=True, blank=True
    )
    metrics = models.ManyToManyField(Metric)
    callbacks = models.ManyToManyField(
        CallbackFunction, related_name="inference_callbacks"
    )
    prediction_metrics = models.ManyToManyField(
        TrainingMetric,
        related_name="inference_prediction_metrics",
    )
    gpu = models.CharField(max_length=20, default="/gpu:0")
    worker = models.CharField(max_length=70)
    compute_device = models.CharField(max_length=20, default="")
    network_training = models.ForeignKey(
        NetworkTraining, on_delete=models.deletion.DO_NOTHING, null=True, blank=True
    )
    batch_size = models.IntegerField(default=1)
    flops = models.IntegerField(default=0)

    _model = None
    _train_data = None
    _test_data = None
    _eval_data = None

    def __str__(self):
        return self.name

    def get_callbacks(self):
        """
        Get the evaluation callbacks for the given network training run.

        Args:
            run (NetworkTraining): The network training run.

        Returns:
            list: A list of evaluation callbacks.
        """
        callbacks = []
        for callback in self.callbacks.all():
            if callback.instance_type.name == "EnergyCallback":
                callbacks.append(
                    get_object(
                        callback.instance_type.module_name,
                        callback.instance_type.name,
                        callback.additional_arguments
                        + [{"name": "run", "value": self}],
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

    def _try_link_run(self):
        # get the id from the model file, and then tr=y to get the networktraining object from it
        model_file_id = os.path.splitext(
            self.model_file[self.model_file.rfind("_") + 1 :]
        )[0]
        if not model_file_id:
            return
        # TODO: if the model file leis in the tensorflow dierctory then lookfor the network_config__id
        # otherwise look for tensorflow_model__id
        if self.model_file.rfind("/tensorflow/") > 0:
            matching_networks = NetworkTraining.objects.filter(
                network_config__id=model_file_id
            )
        else:
            matching_networks = NetworkTraining.objects.filter(
                tensorflow_model__id=model_file_id
            )
        if matching_networks.exists():
            self.network_training = matching_networks.first()
            self.save()

    def _load_model(self):
        self._try_link_run()
        self._model = tf.keras.models.load_model(self.model_file)
        if not self._train_data:
            self._load_data()

    def _load_data(self):
        (self._train_data, self._test_data, self._eval_data) = self.dataset.get_data()

    def predict(self, datapoint):
        """
        Do actual inference with a datapoint and get the result
        """
        if not self._model:
            self._load_model()

        return self._model.predict(datapoint)

    def measure_inference(self, callbacks=[], **kwargs):
        if not self._model:
            self._load_model()
        timer = TimingCallback()
        self.flops = calculate_flops(self._model, self.batch_size)
        self.save()
        predictions = self._model.predict(
            self._train_data.batch(self.batch_size),
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[timer] + self.get_callbacks() + callbacks,
        )
        try:
            predicted_classes = tf.argmax(
                predictions[list(predictions.keys())[0]], axis=1
            )

            true_labels = np.concatenate(
                [y for x, y in self._train_data.batch(self.batch_size)], axis=0
            )
            print(predicted_classes[0])
            print(true_labels[0])

            accuracy = accuracy_score(true_labels, predicted_classes)
            print("accuracy:", accuracy)
        except:
            print("Something went wrong with calculating the accuracy")
        return predictions
