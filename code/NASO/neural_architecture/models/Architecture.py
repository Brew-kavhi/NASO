from django.db import models

from neural_architecture.models.Graphs import Graph
from neural_architecture.models.Types import (
    ActivationFunctionType,
    NetworkLayerType,
    TypeInstance,
)
from neural_architecture.validators import validate_dtype


class SearchSpace(models.Model):
    graph_representation = models.ForeignKey(
        Graph, on_delete=models.deletion.DO_NOTHING
    )


class LayerConfig(models.Model):
    # Thjis stores the id of the layer object
    layer_id = models.IntegerField()
    layer_type = models.CharField(max_length=150)

    class Meta:
        unique_together = [["layer_type", "layer_id"]]


class NetworkLayer(TypeInstance):
    layer_type = models.ForeignKey(
        NetworkLayerType, on_delete=models.deletion.DO_NOTHING
    )

    trainable = models.BooleanField(default=True)
    name = models.CharField(max_length=60)
    dtype = models.CharField(max_length=20, validators=[validate_dtype])
    dynamic = models.BooleanField(default=False)

    def __str__(self):
        return "Layer"

    def get_size(self):
        # the given module shuld someehow support this. consult tensorflow doc for layer
        raise NotImplementedError(
            "Hey, Don't forget to implement the sdize calculation!"
        )

    def build_tensorflow_layer(self):
        # This function should return a tensorflow object representing the object with this configuration
        raise NotImplementedError(
            "Hey, Don't forget to implement the tensorflow build function!"
        )


class NetworkConfiguration(models.Model):
    layers = models.ManyToManyField(NetworkLayer)
    name = models.CharField(max_length=50)
    connections = models.JSONField(default=dict)
    model = None
    size = models.IntegerField(default=0)


class ActivationFunction(TypeInstance):
    type = models.ForeignKey(
        ActivationFunctionType, on_delete=models.deletion.DO_NOTHING
    )
    # add a few activation properties here
