import numpy as np
import tensorflow as tf

# TODO(b/151772467): Move away from depending on private APIs
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.ops import variables
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from neural_architecture.helper_scripts.architecture import is_feedforward
from plugins.interfaces.pruning_method import PruningInterface

keras = tf.keras
K = keras.backend


def calculate_sparsity(model):
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    params = []

    tfmot_prunable_layers = pruning_wrapper.collect_prunable_layers(model)
    other_prunable_layers = collect_prunable_layers(model)

    for layer in tfmot_prunable_layers:
        for _, mask, _ in layer.pruning_vars:
            params.append(mask)

    params.append(model.optimizer.iterations)

    values = K.batch_get_value(params)
    del values[-1]
    del params[-1]

    sparsity_values = []
    if len(values[::2]) > 0:
        sparsity_values = [
            (1 - np.mean(mask_value))
            * np.count_nonzero(np.isnan(mask))
            / trainable_count
            for mask_value in values[::2]
        ]

    for layer in other_prunable_layers:
        if layer not in tfmot_prunable_layers:
            sparsity_values.append(
                layer.sparsity * layer.count_params() / trainable_count
            )
    return np.sum(sparsity_values)


def collect_prunable_layers(model) -> list[PruningInterface]:
    prunable_layers = []
    for layer in model.layers:
        if issubclass(layer.__class__, PruningInterface):
            prunable_layers.append(layer)
    return prunable_layers


def smart_cond(
    pred, true_fn=None, false_fn=None, name=None
):  # pylint: disable=invalid-name
    """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

    If `pred` is a bool or has a constant value, we return either `true_fn()`
    or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

    Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

    Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

    Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
    """
    if isinstance(pred, variables.Variable):
        return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
    return smart_module.smart_cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)


def strip_low_magnitude_model(model):
    if not isinstance(model, keras.Model):
        raise ValueError(
            "Expected model to be a `tf.keras.Model` instance but got: ", model
        )

    def _strip_pruning_wrapper(layer):
        if isinstance(layer, keras.Model):
            # A keras model with prunable layers
            return keras.models.clone_model(
                layer, input_tensors=None, clone_function=_strip_pruning_wrapper
            )
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            # The _batch_input_shape attribute in the first layer makes a Sequential
            # model to be built. This makes sure that when we remove the wrapper from
            # the first layer the model's built state preserves.
            if not hasattr(layer.layer, "_batch_input_shape") and hasattr(
                layer, "_batch_input_shape"
            ):
                layer.layer._batch_input_shape = layer._batch_input_shape
            return layer.layer
        return layer

    return keras.models.clone_model(
        model, input_tensors=None, clone_function=_strip_pruning_wrapper
    )


def strip_pruning(model):
    if not isinstance(model, keras.Model):
        raise ValueError(
            "Expected model to be a `tf.keras.Model` instance but got: ", model
        )

    contains_pruning_layers = False
    for layer in model.layers:
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            return strip_low_magnitude_model(model)
        elif issubclass(layer.__class__, PruningInterface):
            contains_pruning_layers = True
            break
    if not contains_pruning_layers:
        return model

    if not is_feedforward(model):
        raise ValueError(
            "Expected model to e purely feedforward sequential model, but one layer seems to have multiple inputs"
        )

    x = model.inputs[0]

    for i, layer in enumerate(model.layers):
        if i == 0 and layer.name.startswith("input"):
            continue
        if issubclass(layer.__class__, PruningInterface):
            x = layer.get_reduced_layer(x)
        else:
            config = tf.keras.layers.serialize(layer)
            # TODO instantiate new layer from this config.from
            new_layer = tf.keras.layers.deserialize(config)
            x = new_layer(x)
    return tf.keras.Model(inputs=model.inputs[0], outputs=x)
