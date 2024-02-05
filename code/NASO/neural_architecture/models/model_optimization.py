import autokeras as ak
import tensorflow_model_optimization as tfmot
from django.core.exceptions import ValidationError
from django.db import models
from loguru import logger
from tensorflow_model_optimization.python.core.sparsity.keras.prune_registry import (
    PruneRegistry,
)
from neural_architecture.helper_scripts.architecture import copy_model

from helper_scripts.importing import get_object
from neural_architecture.models.types import BaseType, TypeInstance


class PruningMethodTypes(BaseType):
    """
    This model stores the different pruning methods that are available.
    The funhction or class it refers to must be of the following signature:
    def pruning_function(model, **kwargs):
        # do something
        return model
    """

    native_pruning_method = models.BooleanField(default=True)


class PruningScheduleTypes(BaseType):
    """
    This model stores the different pruning schedules that are available.
    """

    native_pruning_schedule = models.BooleanField(default=True)


class PruningPolicyTypes(BaseType):
    """
    This model stores the different pruning policies that are available.
    """

    native_pruning_policy = models.BooleanField(default=True)


class PruningMethod(TypeInstance):
    """
    Represents a pruning method for neural network models.

    Attributes:
        instance_type (ForeignKey): The type of pruning method.
    """

    instance_type = models.ForeignKey(
        PruningMethodTypes, on_delete=models.deletion.CASCADE
    )

    def get_pruned_model(self, to_prune, *args, **kwargs):
        """
        Returns a pruned model based on the specified pruning parameters.

        Args:
            to_prune: The model to be pruned.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The pruned model.

        Raises:
            ValidationError: If the additional_arguments attribute is not a list.
        """

        # add to_prune to the additional_arguments
        if not isinstance(self.additional_arguments, list):
            raise ValidationError("JSON data should be a list of objects.")
        additional_arguments = [
            argument
            for argument in list(self.additional_arguments)
            if not argument["name"] == "to_prune"
        ]

        additional_arguments.insert(0, {"name": "to_prune", "value": to_prune})

        if "pruning_schedule" in kwargs:
            pruning_schedule = kwargs["pruning_schedule"]
            # check if there is already an argument with name pruning_schedule in the additional_arguments:
            pruning_schedule_arg = [
                arg for arg in additional_arguments if arg["name"] == "pruning_schedule"
            ]
            if pruning_schedule_arg:
                # replace the value with the new pruning schedule
                pruning_schedule_arg[0]["value"] = pruning_schedule
            else:
                # add the pruning schedule to the additional arguments
                additional_arguments.append(
                    {"name": "pruning_schedule", "value": pruning_schedule}
                )

        if "pruning_policy" in kwargs:
            pruning_policy = kwargs["pruning_policy"]
            # check if there is already an argument with name pruning_policy in the additional_arguments:
            pruning_policy_arg = [
                arg for arg in additional_arguments if arg["name"] == "pruning_policy"
            ]
            if pruning_policy_arg:
                # replace the value with the new pruning policy
                pruning_policy_arg[0]["value"] = pruning_policy
            else:
                # add the pruning policy to the additional arguments
                additional_arguments.append(
                    {"name": "pruning_policy", "value": pruning_policy}
                )
        self.additional_arguments = additional_arguments

        try:
            return get_object(
                self.instance_type.module_name,
                self.instance_type.name,
                self.additional_arguments,
                self.instance_type.required_arguments,
            )
        except ValueError:
            logger.critical("Could no apply pruning method to this layer")
            return to_prune
        except:
            logger.critical("something happened")
            return to_prune


class PruningSchedule(TypeInstance):
    """
    Represents a pruning schedule for neural network models.

    Attributes:
        instance_type (ForeignKey): The type of pruning schedule.
    """

    instance_type = models.ForeignKey(
        PruningScheduleTypes, on_delete=models.deletion.CASCADE
    )


class PruningPolicy(TypeInstance):
    """
    Represents a pruning policy for neural network models.

    Attributes:
        instance_type (ForeignKey): The type of pruning policy.
    """

    instance_type = models.ForeignKey(
        PruningPolicyTypes, on_delete=models.deletion.CASCADE
    )


class PrunableNetwork(models.Model):
    """
    A base class for prunable neural network models.

    Attributes:
        enable_quantization (bool): Flag indicating whether quantization is enabled.
        pruning_method (PruningMethod): The pruning method to be used.
        pruning_schedule (PruningSchedule): The pruning schedule to be used.
        pruning_policy (PruningPolicy): The pruning policy to be used.

    Methods:
        build_pruning_model(model): Builds a pruning model based on the specified pruning method, schedule, and policy.
        get_pruning_callbacks(): Returns the pruning callbacks based on the specified pruning method.
        get_export_model(model): Returns the exported model with pruning applied.

    """

    enable_quantization = models.BooleanField(default=False)
    pruning_method = models.ForeignKey(
        PruningMethod, on_delete=models.deletion.SET_NULL, null=True
    )
    pruning_schedule = models.ForeignKey(
        PruningSchedule, on_delete=models.deletion.SET_NULL, null=True
    )
    pruning_policy = models.ForeignKey(
        PruningPolicy, on_delete=models.deletion.SET_NULL, null=True
    )

    class Meta:
        abstract = True

    def build_pruning_model(self, model, include_last_layer=True):
        """
        Builds a pruning model based on the specified pruning method, schedule, and policy.

        Args:
            model: The original model to be pruned.

        Returns:
            The pruned model.

        """
        args = {"to_prune": model}
        if self.pruning_method:
            if self.pruning_schedule:
                args["pruning_schedule"] = get_object(
                    self.pruning_schedule.instance_type.module_name,
                    self.pruning_schedule.instance_type.name,
                    self.pruning_schedule.additional_arguments,
                    self.pruning_schedule.instance_type.required_arguments,
                )
            if self.pruning_policy:
                args["pruning_policy"] = get_object(
                    self.pruning_policy.instance_type.module_name,
                    self.pruning_policy.instance_type.name,
                    self.pruning_policy.additional_arguments,
                    self.pruning_policy.instance_type.required_arguments,
                )
            else:
                args["pruning_policy"] = EnsurePrunableModelPolicy(
                    model, include_last_layer
                )

            logger.info("Built pruning model")
            # here iterate ove the layers and prune every layer seperately
            model_layers = {}
            for i, layer in enumerate(model.layers):
                if args["pruning_policy"].allow_pruning(layer):
                    args["to_prune"] = layer
                    pruned_layer = self.pruning_method.get_pruned_model(**args)
                    model_layers[layer.name] = pruned_layer
                else:
                    print(f" layer {layer} is not prunable")
                    model_layers[layer.name] = layer
            return copy_model(model, model_layers)
        return model

    def get_pruning_callbacks(self):
        """
        Returns the pruning callbacks based on the specified pruning method.

        Returns:
            A list of pruning callbacks.

        """
        if self.pruning_method:
            return [
                tfmot.sparsity.keras.UpdatePruningStep(),
            ]
        return []

    def get_export_model(self, model):
        """
        Returns the exported model with pruning applied.

        Args:
            model: The original model.

        Returns:
            The exported model with pruning applied.

        """
        if self.pruning_method:
            export_model = tfmot.sparsity.keras.strip_pruning(model)
            print("final model")
            export_model.summary()
            return export_model
        return model


class EnsurePrunableModelPolicy(tfmot.sparsity.keras.PruningPolicy):
    _model = None
    _include_last_layer = True

    def __init__(self, model, include_last_layer=True):
        self._model = model
        self._include_last_layer = include_last_layer

    def allow_pruning(self, layer):
        if not self._include_last_layer and layer == self._model.layers[-1]:
            return False
        return allow_pruning(layer)

    def ensure_model_supports_pruning(self, model):
        """Checks that the model contains only supported layers.

        Args:
        model: A `tf.keras.Model` instance which is going to be pruned.

        Raises:
        ValueError: if the keras model doesn't support pruning policy, i.e. keras
            model contains an unsupported layer.
        """


def allow_pruning(layer):
    # if layer si autokeas head always return false
    if isinstance(layer, ak.Head):
        return False
    registry = PruneRegistry()
    allowance = (
        isinstance(layer, tfmot.sparsity.keras.PrunableLayer)
        or hasattr(layer, "get_prunable_weights")
        or registry.supports(layer)
    )
    return allowance
