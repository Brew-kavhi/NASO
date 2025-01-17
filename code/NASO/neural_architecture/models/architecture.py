import os
import zipfile

import keras
from decouple import config
from django.db import models
from loguru import logger

from helper_scripts.importing import get_object
from neural_architecture.models.graphs import Graph
from neural_architecture.models.model_optimization import (
    ClusterableNetwork,
    PrunableNetwork,
)
from neural_architecture.models.types import (
    ActivationFunctionType,
    BuildModelFromGraph,
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
    """
    Represents a network layer in a neural architecture.

    Attributes:
        layer_type (ForeignKey): The type of the network layer.
        trainable (bool): Indicates whether the layer is trainable or not.
        name (str): The name of the layer.
        dtype (str): The data type of the layer.
        dynamic (bool): Indicates whether the layer is dynamic or not.
    """

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
        """
        Calculates the size of the layer.

        Returns:
            int: The size of the layer.

        Raises:
            NotImplementedError: If the size calculation is not implemented.
        """
        raise NotImplementedError(
            "Hey, Don't forget to implement the size calculation!"
        )

    def build_tensorflow_layer(self):
        """
        Builds a TensorFlow layer object based on the configuration.

        Returns:
            tensorflow.python.keras.layers.Layer: The TensorFlow layer object.

        Raises:
            NotImplementedError: If the TensorFlow build function is not implemented.
        """
        raise NotImplementedError(
            "Hey, Don't forget to implement the TensorFlow build function!"
        )


class NetworkModel(PrunableNetwork):
    model_type = "tensorflow"

    class Meta:
        abstract = True

    size = models.IntegerField(default=0)
    name = models.CharField(max_length=50)
    save_model = models.BooleanField(default=False)
    model_file = models.CharField(max_length=100)
    clustering_options = models.ForeignKey(
        ClusterableNetwork, null=True, on_delete=models.deletion.CASCADE
    )
    load_model = models.BooleanField(default=False)

    def save_model_on_disk(self, model):
        if self.save_model:
            file_path = f"{config('TENSORFLOW_MODEL_PATH')}{self.model_type}/{self.name}_{self.id}.keras"
            if not os.path.exists(
                config("TENSORFLOW_MODEL_PATH") + self.model_type + "/"
            ):
                os.makedirs(config("TENSORFLOW_MODEL_PATH") + self.model_type + "/")

            export_model = self.get_export_model(model)
            if self.clustering_options:
                export_model = self.clustering_options.get_cluster_export_model(
                    export_model
                )

            export_model.save(file_path, save_format="keras")

            with zipfile.ZipFile(
                f"{config('TENSORFLOW_MODEL_PATH')}{self.model_type}/{self.name}_{self.id}.zip",
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zip_file:
                zip_file.write(file_path)

            self.model_file = file_path
            self.save()
            logger.success(f"Saved model to {self.name}_{self.id}.keras")
        else:
            logger.info(f"Not saving model, but save is {self.save}")

    def build_model(self, input_shape):
        """
        Builds/returns a tf.Keras.Model that is used for training and evaluation and everything
        """

    def get_gzipped_model_size(self) -> int:
        if self.save_model:
            return os.path.getsize(os.path.splitext(self.model_file)[0] + ".zip")
        return -1


class NetworkConfiguration(BuildModelFromGraph, NetworkModel):
    """
    Represents a network configuration with layers, connections, and model-related information.
    """

    layers = models.ManyToManyField(NetworkLayer)
    connections = models.JSONField(default=dict)
    node_to_layer_id = models.JSONField(default=dict)
    model = None

    inputs: dict = {}

    def build_model(self, input_shape=(28, 28)):
        """
        Builds and returns a Keras model based on the network configuration.

        Args:
            input_shape (tuple): The shape of the input data. Defaults to (28, 28).

        Returns:
            keras.Model: The built Keras model.
        """
        if (
            self.load_model
            and len(self.model_file) > 0
            and os.path.exists(self.model_file)
        ):
            logger.info(f"Loading model from {self.model_file}")
            return keras.models.load_model(self.model_file)

        self.inputs["input_node"] = keras.Input(input_shape)
        self.layer_outputs["input_node"] = self.inputs["input_node"]
        self.build_connected_layers("input_node")
        return keras.Model(self.inputs, self.outputs)

    def get_block_for_node(self, node_id):
        """
        Retrieves the block associated with a given node ID.

        Args:
            node_id (int): The ID of the node.

        Returns:
            object: The block associated with the node.
        """
        node_id = self.node_to_layer_id[node_id]
        node = self.layers.get(pk=node_id)
        block = get_object(
            node.layer_type.module_name,
            node.layer_type.name,
            node.additional_arguments,
            node.layer_type.required_arguments,
        )
        return block


class ActivationFunction(TypeInstance):
    """
    Represents an activation function used in a neural network.

    Attributes:
        type (ActivationFunctionType): The type of activation function.
        # add any additional activation properties here
    """

    type = models.ForeignKey(
        ActivationFunctionType, on_delete=models.deletion.DO_NOTHING
    )
    # add a few activation properties here
