import inspect
import re

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.keras import compat as tf_compat
from tensorflow_model_optimization.python.core.sparsity.keras.prune_registry import (
    PruneRegistry,
)

from helper_scripts.pruning import smart_cond
from neural_architecture.models.model_optimization import PruningMethodTypes
from plugins.interfaces.commands import InstallerInterface
from plugins.interfaces.pruning_method import PruningInterface

keras = tf.keras
K = keras.backend
Wrapper = keras.layers.Wrapper


class Installer(InstallerInterface):
    def install(self):
        # Implement the installation logic here
        try:
            # Create instances in the database
            _, _ = PruningMethodTypes.objects.get_or_create(
                module_name=self.get_module_name(),
                name="SimilarityPruning",
                native_pruning_method=False,
                required_arguments=[
                    {"name": "threshold", "default": 0.9, "dtype": "float"},
                    {"name": "update_frequency", "default": 10, "dtype": "int"},
                    {
                        "name": "similarity_metric",
                        "default": "COS",
                        "dtype": "str",
                    },
                ],
            )
            # You can add more database operations or any other setup logic here

            print("Plugin installed successfully.")
        except Exception as e:
            # Handle any exceptions that may occur during installation
            print(f"Plugin installation failed: {str(e)}")
        print("Installation completed")

    def uninstall(self):
        # Implement the uninstallation logic here
        PruningMethodTypes.objects.filter(
            module_name=self.get_module_name(),
            name="SimilarityPruning",
        ).delete()
        print("Uninstallation completed")


def _to_snake_case(name: str) -> str:
    """Converts `name` to snake case.

    Example: "TensorFlow" -> "tensor_flow"

    Args:
    name: The name of some python class.

    Returns:
    `name` converted to snake case.
    """
    intermediate = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != "_":
        return insecure
    return "private" + insecure


class SimilarityPruning(Wrapper, PruningInterface):
    _sparsity = 0
    _weights = None
    _biases = None
    _is_conv_layer = False
    _pruned_filters = []
    _kept_filters = []

    def __init__(
        self,
        to_prune: keras.layers.Layer,
        threshold: float,
        similarity_metric: str,
        update_frequency: int = 10,
        **kwargs,
    ):
        kwargs.pop("pruning_policy")
        kwargs.update(
            {"name": f"{_to_snake_case(self.__class__.__name__)}_{to_prune.name}"}
        )
        super().__init__(layer=to_prune, **kwargs)
        self._threshold = tf.constant(threshold)
        self._is_conv_layer = isinstance(self.layer, keras.layers.Conv2D)
        if similarity_metric not in ["EUC", "COS", "IoU"]:
            raise ValueError(
                "Unsupported metric type '{}'. Should be 'EUC' or 'COS' or 'IoU'.".format(
                    similarity_metric
                )
            )
        self.epoch = 0
        self.update_frequency = update_frequency
        self._similiarity_measure = similarity_metric
        self._track_trackable(to_prune, name="layer")

    def _calc_similarity(self, weight1, weight2):
        # calc the simiarity based on the given metric
        similarity = 0
        if self._similiarity_measure == "COS":
            dot_product = tf.reduce_sum(tf.multiply(weight1, weight2))
            norm_weight1 = tf.norm(weight1)
            norm_weight2 = tf.norm(weight2)
            similarity = 0 + dot_product / (
                norm_weight1 * norm_weight2 + 1e-9
            )  # Adding small epsilon to avoid division by zero
        elif self._similiarity_measure == "EUC":
            # this gives the euclidian simliary = 1 - distance
            similarity = tf.nn.relu(
                1 - tf.norm(weight1 - weight2) / (tf.norm(weight1) * tf.norm(weight2))
            )
        elif self._similiarity_measure == "IoU":
            intersection = tf.reduce_sum(tf.minimum(weight1, weight2))
            union = tf.reduce_sum(tf.maximum(weight1, weight2))
            similarity = intersection / (union + 1e-9)
        return abs(similarity)

    def build(self, input_shape):
        super().build(input_shape)
        if self._is_conv_layer:
            weight_vars, mask_vars = [], []

            self.prunable_weights = [
                getattr(self.layer, weight)
                for weight in PruneRegistry._LAYERS_WEIGHTS_MAP[self.layer.__class__]
            ]

            # For each of the prunable weights, add mask and threshold variables
            for weight in self.prunable_weights:
                mask = self.add_weight(
                    "mask",
                    shape=weight.shape,
                    initializer=keras.initializers.get("ones"),
                    dtype=weight.dtype,
                    trainable=False,
                    aggregation=tf.VariableAggregation.MEAN,
                )
                weight_vars.append(weight)
                mask_vars.append(mask)
            self._pruning_vars = list(zip(weight_vars, mask_vars))

            # Add a scalar tracking the number of updates to the wrapped layer.
        self.pruning_step = self.add_weight(
            "pruning_step",
            shape=[],
            initializer=keras.initializers.Constant(-1),
            dtype=tf.int64,
            trainable=False,
        )

    def call(self, x, training, **kwargs):
        if training is None:
            training = K.learning_phase()

        def increment_step():
            with tf.control_dependencies(
                [tf_compat.assign(self.pruning_step, self.pruning_step + 1)]
            ):
                return tf.no_op("update")

        def add_update():
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        self.pruning_step,
                        np.int64(1),
                        message="ERROR MSG",
                    )
                ]
            ):
                with tf.control_dependencies([self.conditional_mask_update()]):
                    return tf.no_op("update")

        def no_op():
            return tf.no_op("no_update")

        if self._is_conv_layer:
            # Increment the 'pruning_step' after each step.
            # this function just registers it in the graph basically
            update_pruning_step = smart_cond(training, increment_step, no_op)
            # and only this function then finally addds it to the computation graph, this is from Layer
            self.add_update(update_pruning_step)

            # Update mask tensor after each 'pruning_frequency' steps.
            # update_mask = smart_cond(training, add_update, no_op)
            # self.add_update(update_mask)

            self.add_update(self.weight_mask_op())
            if hasattr(inspect, "getfullargspec"):
                args = inspect.getfullargspec(self.layer.call).args
            else:
                args = inspect.getargspec(self.layer.call).args
            # Propagate the training bool to the underlying layer if it accepts
            # training as an arg.
            if "training" in args:
                return self.layer.call(x, training=training, **kwargs)

        return self.layer.call(x, **kwargs)

    def calc_similarity_matrix(self, weights):
        similarity_mask = np.full(
            weights.shape, 1, dtype=np.float32
        )  # multiply by big number because we are not setting the whole matrix
        if self._is_conv_layer:
            num_filters = weights.shape[-1]

            self._pruned_filters = []
            self._kept_filters = [0]
            for i in range(num_filters):
                if i in self._pruned_filters:
                    continue
                if i not in self._kept_filters:
                    self._kept_filters.append(i)
                for j in range(i + 1, num_filters):
                    if j in self._pruned_filters:
                        continue
                    if self._is_conv_layer:
                        similarity = self._calc_similarity(
                            weights[:, :, :, i], weights[:, :, :, j]
                        )
                        if tf.math.greater_equal(similarity, self._threshold):
                            self._pruned_filters.append(j)
                            if j in self._kept_filters:
                                self._kept_filters.remove(j)
                            similarity_mask[:, :, :, j] = 0
                        else:
                            if (
                                j not in self._kept_filters
                                and j not in self._pruned_filters
                            ):
                                self._kept_filters.append(j)

            print(
                f"{self.layer.name} has {num_filters} many filters, {len(self._pruned_filters)} of which are pruned"
            )
            self._sparsity = 1 - np.mean(similarity_mask)
            # Update the layer with pruned self._weights
        return tf.constant(similarity_mask)

    @property
    def sparsity(self):
        return self._sparsity

    def count_params(self):
        return self.layer.count_params()

    def compute_output_shape(self, input_shape):
        if self._is_conv_layer:
            self.layer.filters = self.layer.filters - len(self._pruned_filters)
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config["layer"] = keras.layers.serialize(self.layer)
        return config

    @classmethod
    def from_config(cls, config):
        layer = keras.layers.deserialize(config.pop("layer"))
        return cls(layer, **config)

    @property
    def trainable(self):
        return self.layer.trainable

    @trainable.setter
    def trainable(self, value):
        self.layer.trainable = value

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights + self._non_trainable_weights

    @property
    def updates(self):
        return self.layer.updates + self._updates

    @property
    def losses(self):
        return self.layer.losses + self._losses

    def get_weights(self):
        return self.layer.get_weights()

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def _update_mask(self, weights):
        """Updates the mask for a given weight tensor.

        This function first calculates the similarity of the weights and then creates a mask based on similarity

        Args:
          weights: The weight tensor that needs to be masked.

        Returns:
          new_mask: A numpy array of the same size and shape as weights containing
            0 or 1 to indicate which of the values in weights falls below
            the threshold

        Raises:
          ValueError: if sparsity is not defined
        """
        with tf.name_scope("pruning_ops"):
            new_mask = self.calc_similarity_matrix(weights)
        return tf.dtypes.cast(new_mask, weights.dtype)

    def _weight_assign_objs(self):
        """Gather the assign objs for assigning weights<=weights*mask.

        The objs are ops for graph execution and tensors for eager
        execution.
        Taken as is from tensorflow model optimization:
        https://github.com/tensorflow/model-optimization/blob/v0.7.5/tensorflow_model_optimization/python/core/sparsity/keras/pruning_impl.py

        Returns:
          group of objs for weight assignment.
        """

        def update_fn(distribution, values_and_vars):
            # TODO(yunluli): Need this ReduceOp because the weight is created by the
            # layer wrapped, so we don't have control of its aggregation policy. May
            # be able to optimize this when distribution strategy supports easier
            # update to mirrored variables in replica context.
            reduced_values = distribution.extended.batch_reduce_to(
                tf.distribute.ReduceOp.MEAN, values_and_vars
            )
            var_list = [v for _, v in values_and_vars]
            values_and_vars = zip(reduced_values, var_list)

            def update_var(variable, reduced_value):
                return tf_compat.assign(variable, reduced_value)

            update_objs = []
            for value, var in values_and_vars:
                update_objs.append(
                    distribution.extended.update(var, update_var, args=(value,))
                )

            return tf.group(update_objs)

        assign_objs = []

        if tf.distribute.get_replica_context():
            values_and_vars = []
            for weight, mask in self._pruning_vars:
                masked_weight = tf.dtypes.cast(
                    tf.math.multiply(weight, mask), dtype=weight.dtype
                )
                values_and_vars.append((masked_weight, weight))
            if values_and_vars:
                assign_objs.append(
                    tf.distribute.get_replica_context().merge_call(
                        update_fn, args=(values_and_vars,)
                    )
                )
        else:
            for weight, mask, _ in self._pruning_vars:
                masked_weight = tf.dtypes.cast(
                    tf.math.multiply(weight, mask), dtype=weight.dtype
                )
            assign_objs.append(tf_compat.assign(weight, masked_weight))

        return assign_objs

    def weight_mask_op(self):
        if self._is_conv_layer:
            return tf.group(self._weight_assign_objs())
        return tf.no_op("update")

    def conditional_mask_update(self):
        """Returns an op to updates masks as per the pruning schedule. Pruning stedp is incremented wioth each execution obvisouly, so one epoch equals #batches pruning steps"""

        if self.epoch % self.update_frequency != 0:
            self.epoch += 1
            return
        self.epoch += 1

        def mask_update():
            """Updates mask without distribution strategy."""
            assign_objs = []

            for weight, mask in self._pruning_vars:
                new_mask = self._update_mask(weight)
                assign_objs.append(tf_compat.assign(mask, new_mask))

            return tf.group(assign_objs)

        def mask_update_distributed(distribution):
            """Updates mask with distribution strategy."""

            def update(var, value):
                return tf_compat.assign(var, value)

            def update_distributed():
                """Gather distributed update objs.

                The objs are ops for graph execution and tensors for eager execution.
                """
                assign_objs = []

                for weight, mask in self._pruning_vars:
                    new_mask = self._update_mask(weight)
                    assign_objs.append(
                        distribution.extended.update(mask, update, (new_mask,))
                    )

                return tf.group(assign_objs)

            return update_distributed()

        if self._is_conv_layer:
            if tf.distribute.get_replica_context():
                return tf.distribute.get_replica_context().merge_call(
                    mask_update_distributed
                )
            else:
                return mask_update()

    def get_reduced_layer(self, input):
        config = self.layer.get_config()
        if self._is_conv_layer:
            config["filters"] = len(self._kept_filters)
            new_layer = tf.keras.layers.Conv2D(**config)
            return new_layer(input)

        return self.layer(input)

    def strip_pruning(self):
        if not hasattr(self.layer, "_batch_input_shape") and hasattr(
            self, "_batch_input_shape"
        ):
            self.layer._batch_input_shape = self._batch_input_shape

        return self.layer
