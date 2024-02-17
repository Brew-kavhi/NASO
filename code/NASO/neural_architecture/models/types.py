import abc

from django.core.exceptions import ValidationError
from django.db import models

from neural_architecture.helper_scripts.architecture import (
    build_connected_layer,
    edges_from_source,
    edges_to_target,
    is_head_node,
    is_merge_node,
)


# This handles all python classses.
# that is in these types i just want to save what optimizers are availabel and how to call these classes
# instantiation with all arguments is done by the actual models, taht jsut have this type assigned
class BaseType(models.Model):
    """
    Base class for defining types in the neural architecture models.
    It does not resemable a class itself, but is used to save the type of a class in the database.

    Attributes:
        module_name (str): The name of the module.
        name (str): The name of the type.
        required_arguments (list): A list of required arguments for the type.

    Meta:
        abstract (bool): Specifies that this is an abstract base class.
        unique_together (tuple): Specifies that the combination of module_name and name should be unique.

    Methods:
        __str__(): Returns the name of the type.
        validate_json_data(): Validates the required_arguments attribute.
        save(): Overrides the save method to perform additional validation before saving the object.
    """

    module_name = models.CharField(max_length=150)
    name = models.CharField(max_length=100)
    required_arguments = models.JSONField(null=True)

    class Meta:
        abstract = True
        unique_together = (("module_name", "name"),)

    def __str__(self):
        return self.name

    def validate_json_data(self):
        if not isinstance(self.required_arguments, list):
            raise ValidationError("JSON data should be a list.")

        for item in self.required_arguments:
            if (
                not isinstance(item, dict)
                or "name" not in item
                or "default" not in item
            ):
                raise ValidationError(
                    "Each item in the JSON list should be a dict with a name and a default value."
                )

    def save(self, *args, **kwargs):
        self.validate_json_data()
        super().save(*args, **kwargs)


class OptimizerType(BaseType):
    """
    Represents an optimizer type for neural network models.

    Attributes:
        keras_native_optimizer (bool): Indicates whether to use the native optimizer provided by Keras.
    """

    keras_native_optimizer = models.BooleanField(default=True)


class CallbackType(BaseType):
    """
    Represents a callback type used in neural network models.

    Attributes:
        keras_native_callback (bool): Indicates whether the callback is a native Keras callback.
        registers_metrics (str): A comma-separated list of metrics that the callback registers.

    """

    keras_native_callback = models.BooleanField(default=True)
    registers_metrics = models.TextField()


class LossType(BaseType):
    """
    LossType class represents a type of loss used in a neural network model.

    Attributes:
        keras_native_loss (bool): Indicates whether the loss is a native Keras loss or not.
    """

    keras_native_loss = models.BooleanField(default=True)


class MetricType(BaseType):
    """
    Represents a metric type used in neural network models.

    Attributes:
        keras_native_metric (bool): Indicates whether the metric is a native Keras metric.
    """

    keras_native_metric = models.BooleanField(default=True)


class NetworkLayerType(BaseType):
    """
    Represents a type of network layer.

    Attributes:
        keras_native_layer (bool): Indicates whether the layer is a native Keras layer.
    """

    keras_native_layer = models.BooleanField(default=True)


class ActivationFunctionType(BaseType):
    """
    Represents an activation function type.

    Attributes:
        keras_native_activation (bool): Indicates whether the activation function is a native Keras activation.
    """

    keras_native_activation = models.BooleanField(default=True)


class TypeInstance(models.Model):
    """
    Represents a type instance with additional arguments.
    This isd actually an instance of a class

    Attributes:
        additional_arguments (list): A list of additional arguments for the type instance.

    Methods:
        __str__(): Returns a string representation of the type instance.
        validate_json_data(): Validates the JSON data for the additional arguments.
        save(*args, **kwargs): Saves the type instance to the database.
        print_all_fields(): Prints all the fields of the type instance.
    """

    additional_arguments = models.JSONField()

    class Meta:
        abstract = True

    def __str__(self):
        # Use getattr to access the 'type' property based on the subclass
        try:
            type_name = getattr(
                self, self._meta.get_field("instance_type").attname, None
            )
        except Exception:
            try:
                type_name = getattr(
                    self, self._meta.get_field("node_type").attname, None
                )
            except Exception:
                type_name = "Unknown"
        return str(type_name) if type_name else ""

    def validate_json_data(self):
        """
        Validates the JSON data for the additional arguments.
        Needs to be an array of objects with name and value attributes.

        Raises:
            ValidationError: If the JSON data is not a list of objects or if any object is missing the "name" or "value" attributes.
        """
        if not isinstance(self.additional_arguments, list):
            raise ValidationError("JSON data should be a list of objects.")

        for item in self.additional_arguments:
            if not isinstance(item, dict) or "name" not in item or "value" not in item:
                raise ValidationError(
                    'Each item in the JSON list should be an object with "name" and "value" attributes.'
                )

    def save(self, *args, **kwargs):
        """
        Saves the type instance to the database.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.validate_json_data()
        super().save(*args, **kwargs)

    def print_all_fields(self):
        """
        Prints all the fields of the type instance.
        """
        print(self._meta.fields)


class BuildModelFromGraph(models.Model):
    """
    A base class for building a model from a graph representation.

    Attributes:
        connections (dict): A dictionary representing the connections between nodes in the graph.
        outputs (dict): A dictionary to store the output layers of the model.
        layer_outputs (dict): A dictionary to store the output of each layer in the model.
    """

    connections = models.JSONField(default=dict)
    outputs: dict = {}
    layer_outputs: dict = {}

    class Meta:
        abstract = True

    def edges_from_source(self, node_id):
        """
        Get all edges that originate from a given node.

        Args:
            node_id (int): The ID of the source node.

        Returns:
            list: A list of edges originating from the given node.
        """
        return edges_from_source(node_id, self.connections)

    def edges_to_target(self, node_id):
        """
        Get all edges that target a given node.

        Args:
            node_id (int): The ID of the target node.

        Returns:
            list: A list of edges targeting the given node.
        """
        return edges_to_target(node_id, self.connections)

    def is_merge_node(self, node_id):
        """
        Check if a given node is a merge node.

        A merge node is a node that is the target of multiple edges.

        Args:
            node_id (int): The ID of the node to check.

        Returns:
            bool: True if the node is a merge node, False otherwise.
        """
        return is_merge_node(node_id, self.connections)

    def is_head_node(self, node_id):
        """
        Check if a given node is a head node.

        A head node is a node that is not a source node or an inner node.

        Args:
            node_id (int): The ID of the node to check.

        Returns:
            bool: True if the node is a head node, False otherwise.
        """
        return is_head_node(node_id, self.connections)

    @abc.abstractmethod
    def get_block_for_node(self, node_id):
        """
        Get the block/layer corresponding to a given node.

        This method should return a block/layer that can be called with the output of the previous layer.

        Args:
            node_id (int): The ID of the node.

        Returns:
            object: The block/layer corresponding to the given node.
        """

    def build_connected_layers(self, node_id):
        """
        Build the connected layers of the model starting from a given node.

        Args:
            node_id (int): The ID of the starting node.
        """
        self.outputs = build_connected_layer(
            node_id, self.connections, self.get_block_for_node, self.layer_outputs, {}
        )
