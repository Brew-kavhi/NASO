import tensorflow_model_optimization as tfmot
from django.core.exceptions import ValidationError
from django.db import models
from loguru import logger

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
    instance_type = models.ForeignKey(
        PruningMethodTypes, on_delete=models.deletion.CASCADE
    )

    def get_pruned_model(self, to_prune, *args, **kwargs):
        # add to_prune to the additional_arguments
        if not isinstance(self.additional_arguments, list):
            raise ValidationError("JSON data should be a list of objects.")
        additional_arguments = list(self.additional_arguments)
        additional_arguments.append({"name": "to_prune", "value": to_prune})

        if "pruning_schedule" in kwargs:
            pruning_schedule = kwargs["pruning_schedule"]
            # check if there is already a argument  with name pruning_schedule in the additional_arguments:
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
            # check if there is already a argument  with name pruning_policy in the additional_arguments:
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

        return get_object(
            self.instance_type.module_name,
            self.instance_type.name,
            self.additional_arguments,
            self.instance_type.required_arguments,
        )


class PruningSchedule(TypeInstance):
    instance_type = models.ForeignKey(
        PruningScheduleTypes, on_delete=models.deletion.CASCADE
    )


class PruningPolicy(TypeInstance):
    instance_type = models.ForeignKey(
        PruningPolicyTypes, on_delete=models.deletion.CASCADE
    )


class PrunableNetwork(models.Model):
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

    def build_pruning_model(self, model):
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

            logger.info("Built pruning model")
            return self.pruning_method.get_pruned_model(**args)
        return model

    def get_pruning_callbacks(self):
        if self.pruning_method:
            return [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir="logs/"),
            ]
        return []

    def get_export_model(self, model):
        if self.pruning_method:
            export_model = tfmot.sparsity.keras.strip_pruning(model)
            print("final model")
            export_model.summary()
            return export_model
        return model
