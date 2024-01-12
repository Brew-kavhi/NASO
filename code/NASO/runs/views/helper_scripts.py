import inspect
import json

from neural_architecture.management.commands.loadneuralutilities import (
    build_arguments as build_required_arguments,
)
from neural_architecture.models.architecture import NetworkLayer, NetworkLayerType
from runs.models.training import NetworkConfiguration


def build_network_config(name, model):
    """
    This funtion should build a networl condfiguration with connections and layers from an actual tensorflow model
    """
    network_config = NetworkConfiguration(name=name)
    network_config.save()
    layer_info = {}
    node_to_layers = {}
    for i, layer in enumerate(model.layers):
        # ge or create the layer in the atabase
        layer_config = layer.get_config()
        class_name = type(layer).__name__
        module_name = type(layer).__module__
        if module_name.startswith("keras.src.layers"):
            module_name = "tensorflow.keras.layers"
        naso_layer_type = NetworkLayerType.objects.filter(
            module_name=module_name, name=class_name
        )
        if naso_layer_type:
            naso_layer_type = naso_layer_type.first()
        else:
            # create a new one
            constructor = inspect.signature(type(layer).__init__)
            required_arguments = build_required_arguments(
                constructor.parameters.items()
            )
            naso_layer_type = NetworkLayerType(
                module_name=module_name,
                name=class_name,
                required_arguments=required_arguments,
            )
            naso_layer_type.save()
        # these keys are alsways inmcluded in teh config m, but we dont need them in the config as they mess with positional arguments
        layer_config.pop("name", None)
        layer_config.pop("trainable", None)
        layer_config.pop("dtype", None)
        if "kernel_initializer" in layer_config:
            layer_config.pop("kernel_initializer", None)
        if "bias_initializer" in layer_config:
            layer_config.pop("bias_initializer", None)

        layer_info[i] = {
            "id": f"{layer.name}_{i}",
            "name": layer.name,
            "config": layer_config,
            "naso_type": naso_layer_type.id,
        }

        if not layer.name.startswith("input_"):
            naso_layer, _ = NetworkLayer.objects.get_or_create(
                layer_type_id=layer_info[i]["naso_type"],
                name=layer_info[i]["id"],
                additional_arguments=build_additional_arguments(layer_config),
            )
            node_to_layers[layer_info[i]["id"]] = naso_layer.id
            network_config.layers.add(naso_layer)

    network_config.save()
    connections = []
    for i, layer in enumerate(model.layers):
        inbound_layers = [
            inp[0].name for inp in layer._inbound_nodes[0].iterate_inbound()
        ]

        for source_layer_name in inbound_layers:
            source_layer_id = [
                v for k, v in layer_info.items() if v["name"] == source_layer_name
            ][0]
            source_id = source_layer_id["id"]
            if source_layer_name.startswith("input_"):
                # DOC: we ned input_node as source name for all inputs
                source_id = "input_node"

            target_layer_id = f"{layer.name}_{i}"
            connection = {
                "id": f"{source_layer_id['id']}_{target_layer_id}",
                "source": source_id,
                "target": target_layer_id,
            }
            connections.append(connection)

    network_config.connections = connections
    network_config.node_to_layer_id = node_to_layers
    network_config.save()
    return network_config


def build_additional_arguments(confguration):
    arguments = []
    for k, v in confguration.items():
        if isinstance(v, dict):
            argument = {"name": k, "value": json.dumps(str(v))}
        else:
            if type(v) == str:
                argument = {"name": k, "value": v}
            else:
                argument = {"name": k, "value": str(v)}
        arguments.append(argument)

    return arguments
