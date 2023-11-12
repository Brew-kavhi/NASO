import os

import tensorflow as tf
from django.db import models
from loguru import logger

from helper_scripts.importing import get_object
from neural_architecture.models.graphs import Graph
from neural_architecture.models.model_optimization import PrunableNetwork
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


class NetworkConfiguration(PrunableNetwork, BuildModelFromGraph):
    layers = models.ManyToManyField(NetworkLayer)
    name = models.CharField(max_length=50)
    connections = models.JSONField(default=dict)
    node_to_layer_id = models.JSONField(default=dict)
    model = None
    size = models.IntegerField(default=0)

    save_model = models.BooleanField(default=False)
    model_file = models.CharField(max_length=100)
    load_model = models.BooleanField(default=False)

    inputs: dict = {}

    def build_model(self):
        if (
            self.load_model
            and len(self.model_file) > 0
            and os.path.exists(self.model_file)
        ):
            logger.info(f"Loading model from {self.model_file}")
            return tf.keras.models.load_model(self.model_file)

        self.inputs["input_node"] = tf.keras.Input((28, 28))
        self.layer_outputs["input_node"] = self.inputs["input_node"]
        self.build_connected_layers("input_node")
        return tf.keras.Model(self.inputs, self.outputs)

    def get_block_for_node(self, node_id):
        node_id = self.node_to_layer_id[node_id]
        node = self.layers.get(pk=node_id)
        block = get_object(
            node.layer_type.module_name,
            node.layer_type.name,
            node.additional_arguments,
            node.layer_type.required_arguments,
        )
        return block

    def save_model_on_disk(self, model):
        if self.save_model:
            file_path = f"keras_models/tensorflow/{self.name}_{self.id}.h5"
            if not os.path.exists("keras_models/tensorflow/"):
                os.makedirs("keras_models/tensorflow/")

            export_model = self.get_export_model(model)

            export_model.save(file_path, include_optimizer=True)
            self.model_file = file_path
            self.save()
            logger.success(f"Saved model to {self.name}_{self.id}.h5")


class ActivationFunction(TypeInstance):
    type = models.ForeignKey(
        ActivationFunctionType, on_delete=models.deletion.DO_NOTHING
    )
    # add a few activation properties here
