import abc

from django.core.exceptions import ValidationError
from django.db import models


# This handles all python classses.
# that is in these types i just want to save what optimizers are availabel and how to call these classes
# instantiation with all arguments is done by the actual models, taht jsut have this type assigned
class BaseType(models.Model):
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
    keras_native_optimizer = models.BooleanField(default=True)


class CallbackType(BaseType):
    keras_native_callback = models.BooleanField(default=True)
    registers_metrics = models.TextField()


class LossType(BaseType):
    keras_native_loss = models.BooleanField(default=True)


class MetricType(BaseType):
    keras_native_metric = models.BooleanField(default=True)


class NetworkLayerType(BaseType):
    keras_native_layer = models.BooleanField(default=True)


class ActivationFunctionType(BaseType):
    keras_native_activation = models.BooleanField(default=True)


class TypeInstance(models.Model):
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
        if not isinstance(self.additional_arguments, list):
            raise ValidationError("JSON data should be a list of objects.")

        for item in self.additional_arguments:
            if not isinstance(item, dict) or "name" not in item or "value" not in item:
                raise ValidationError(
                    'Each item in the JSON list should be an object with "name" and "value" attributes.'
                )

    def save(self, *args, **kwargs):
        self.validate_json_data()
        super().save(*args, **kwargs)

    def print_all_fields(self):
        print(self._meta.fields)


class BuildModelFromGraph(models.Model):
    connections = models.JSONField(default=dict)
    outputs: dict = {}
    layer_outputs: dict = {}

    class Meta:
        abstract = True

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

    @abc.abstractmethod
    def get_block_for_node(self, node_id):
        # this method should return a block/layer that can be called with the output of the previous layer
        pass

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
