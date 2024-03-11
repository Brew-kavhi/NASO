import tensorflow as tf


def copy_model(model, layers):
    def get_layer(node_id):
        return layers[node_id]

    # get the networks connectivity
    connections = []
    for i, layer in enumerate(model.layers):
        inbound_layers = [
            inp[0].name for inp in layer._inbound_nodes[0].iterate_inbound()
        ]
        for source_layer_name in inbound_layers:
            connection = {"source": source_layer_name, "target": layer.name}
            connections.append(connection)

    # get the input node
    input = None
    for key in layers.keys():
        if key.startswith("input_"):
            input = layers[key]
            break
    if input:
        # build outputs
        layer_outputs = {input.name: model.inputs[0]}
        outputs = build_connected_layer(
            input.name, connections, get_layer, layer_outputs, {}
        )
        new_model = tf.keras.Model(inputs=model.inputs[0], outputs=outputs)
        return new_model


def build_connected_layer(
    layer_name, connections, layer_factory, layer_outputs={}, outputs={}
):
    for edge in edges_from_source(layer_name, connections):
        if is_merge_node(edge["target"], connections):
            can_merge = False
            merge_sources = []
            for merge_edge in edges_to_target(edge["target"], connections):
                if merge_edge["source"] in layer_outputs:
                    can_merge = True
                    merge_sources.append(layer_outputs[merge_edge["source"]])
                else:
                    can_merge = False
                    break
            if can_merge:
                layer_outputs[edge["target"]] = layer_factory(edge["target"])(
                    merge_sources
                )
        else:
            layer_outputs[edge["target"]] = layer_factory(edge["target"])(
                layer_outputs[edge["source"]]
            )
        if not is_head_node(edge["target"], connections):
            return build_connected_layer(
                edge["target"], connections, layer_factory, layer_outputs, outputs
            )
        else:
            outputs[edge["target"]] = layer_outputs[edge["target"]]
            return outputs


def is_head_node(layer_name, connections):
    return not len(edges_from_source(layer_name, connections))


def edges_to_target(target, connections):
    return [d for d in connections if d["target"] == target]


def edges_from_source(source, connections):
    return [d for d in connections if d["source"] == source]


def is_merge_node(layer_name, connections):
    """
    Check if a given node is a merge node.

    A merge node is a node that is the target of multiple edges.

    Args:
        node_id (int): The ID of the node to check.

    Returns:
        bool: True if the node is a merge node, False otherwise.
    """
    return len(edges_to_target(layer_name, connections)) > 1


def calculate_flops(model, batch_size):
    total_flops = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            total_flops += conv_flops(layer, batch_size)
        elif isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(
            layer, tf.keras.layers.AveragePooling2D
        ):
            total_flops += pooling_flops(layer, batch_size)
        elif isinstance(layer, tf.keras.layers.Dense):
            total_flops += fc_flops(layer, batch_size)

    return total_flops


def conv_flops(layer, batch_size):
    input_shape = layer.input_shape[1:]
    output_shape = layer.output_shape[1:]
    kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
    Cin = input_shape[-1]
    Cout = output_shape[-1]
    Hout, Wout = output_shape[1:3]

    flops = Cin * Cout * kernel_size * Hout * Wout * batch_size
    return flops


def pooling_flops(layer, batch_size):
    input_shape = layer.input_shape[1:]
    Cin = input_shape[-1]
    Hin, Win = input_shape[1:3]

    flops = Cin * Hin * Win * batch_size
    return flops


def fc_flops(layer, batch_size):
    input_shape = layer.input_shape[1:]
    output_shape = layer.output_shape[1:]
    Cin = input_shape[-1]
    Cout = output_shape[-1]

    flops = Cin * Cout * batch_size
    return flops
