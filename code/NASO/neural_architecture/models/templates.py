from django.db import models

from neural_architecture.models.architecture import NetworkLayer
from neural_architecture.models.autokeras import AutoKerasNode


class NetworkTemplate(models.Model):
    name = models.CharField(max_length=30, unique=True)
    connections = models.JSONField(default=dict)
    node_to_layer_id = models.JSONField(default=dict)

    class Meta:
        abstract = True

    def __str__(self):
        return self.name


class KerasNetworkTemplate(NetworkTemplate):
    layers = models.ManyToManyField(NetworkLayer, related_name="layers")


class AutoKerasNetworkTemplate(NetworkTemplate):
    blocks = models.ManyToManyField(AutoKerasNode, related_name="blocks")
