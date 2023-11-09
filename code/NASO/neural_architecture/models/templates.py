from django.db import models

from neural_architecture.models.architecture import NetworkLayer
from neural_architecture.models.autokeras import AutoKerasNode


class KerasNetworkTemplate(models.Model):
    name = models.CharField(max_length=30, unique=True)
    layers = models.ManyToManyField(NetworkLayer, related_name="layers")
    connections = models.JSONField(default=dict)
    node_to_layer_id = models.JSONField(default=dict)

    def __str__(self):
        return self.name


class AutoKerasNetworkTemplate(models.Model):
    name = models.CharField(max_length=30, unique=True)
    blocks = models.ManyToManyField(AutoKerasNode, related_name="blocks")
    connections = models.JSONField(default=dict)
    node_to_layer_id = models.JSONField(default=dict)

    def __str__(self):
        return self.name
