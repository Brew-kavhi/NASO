from django.db import models

from neural_architecture.models.architecture import NetworkLayer
from neural_architecture.models.autokeras import AutoKerasNode


class NetworkTemplate(models.Model):
    """
    Represents a network template.

    Attributes:
        name (str): The name of the network template.
        connections (dict): The connections between nodes in the network template.
        node_to_layer_id (dict): The mapping of nodes to layer IDs in the network template.
    """

    name = models.CharField(max_length=30, unique=True)
    connections = models.JSONField(default=dict)
    node_to_layer_id = models.JSONField(default=dict)

    class Meta:
        abstract = True

    def __str__(self):
        return self.name


class KerasNetworkTemplate(NetworkTemplate):
    """
    A template for defining a Keras network.

    Attributes:
        layers (ManyToManyField): A many-to-many relationship field that connects
            the KerasNetworkTemplate to NetworkLayer instances.
    """

    layers = models.ManyToManyField(NetworkLayer, related_name="layers")


class AutoKerasNetworkTemplate(NetworkTemplate):
    """
    A template for creating AutoKeras network models.

    This template is used to define the structure of AutoKeras network models.
    It inherits from the `NetworkTemplate` class and includes a many-to-many
    relationship with `AutoKerasNode` models.

    Attributes:
        blocks (ManyToManyField): A many-to-many relationship with `AutoKerasNode`
            models, representing the blocks in the network.

    """

    blocks = models.ManyToManyField(AutoKerasNode, related_name="blocks")
