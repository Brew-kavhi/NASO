from django.db import models

from helper_scripts.importing import get_object
from neural_architecture.models.Graphs import Graph
from neural_architecture.models.Types import (ActivationFunctionType,
                                              NetworkLayerType, TypeInstance)
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
    node_to_layer_id = models.JSONField(default=dict)
    model = None
    size = models.IntegerField(default=0)

    layer_outputs: dict = {}
    inputs: dict = {}
    outputs: dict = {}  #

    def build_model(self):
        for input_node in self.get_input_nodes():
            self.layer_outputs[input_node] = self.inputs[input_node]
            self.build_connected_layers(input_node)
        return (self.inputs, self.outputs)

    def edges_from_source(self, node_id):
        return [d for d in self.connections if d["source"] == node_id]

    def edges_to_target(self, node_id):
        return [d for d in self.connections if d["target"] == node_id]

    def is_merge_node(self, node_id):
        # merge node if this node is target opf multiple edges
        return len(self.edges_to_target(node_id)) > 1

    def is_head_node(self, node_id):
        # it is a head node, if this node is not a source:
        return not len(self.edges_from_source(node_id))

    def get_input_nodes(self):
        # input nodes are nodes that are no target. but at list one source:
        for node in self.node_to_layer_id:
            incoming_edges = self.edges_to_target(node)
            if len(incoming_edges) == 0:
                # check if this node is a source somewhere:
                outgoing_nodes = self.edges_from_source(node)
                if len(outgoing_nodes) > 0:
                    self.inputs[node] = self.get_block_for_node(node)

        return self.inputs

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

    def build_connected_layers(self, node_id):
        for edge in self.edges_from_source(node_id):
            if self.is_merge_node(edge["target"]):
                can_merge = False
                merge_sources = []
                for merge_edge in self.edges_to_target(edge["target"]):
                    if merge_edge["source"] in self.layer_outputs:
                        can_merge = True
                        merge_sources.append(self.layer_outputs[merge_edge["source"]])
                    else:
                        # as soon
                        can_merge = False
                        break
                if can_merge:
                    self.layer_outputs[edge["target"]] = self.get_block_for_node(
                        edge["target"]
                    )(merge_sources)
            else:
                self.layer_outputs[edge["target"]] = self.get_block_for_node(
                    edge["target"]
                )(self.layer_outputs[edge["source"]])
            if not self.is_head_node(edge["target"]):
                # call this exact same loop again for the target node this time if its not a head
                self.build_connected_layers(edge["target"])
            else:
                # this  node is s head, so add it to outputs:
                self.outputs[edge["target"]] = self.layer_outputs[edge["target"]]


class ActivationFunction(TypeInstance):
    type = models.ForeignKey(
        ActivationFunctionType, on_delete=models.deletion.DO_NOTHING
    )
    # add a few activation properties here
