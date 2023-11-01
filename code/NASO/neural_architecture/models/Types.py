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

        # TODO allow not only a list of strings, but rather a list of objects
        # that define the name of the argument and mayvbe provde a default
        for item in self.required_arguments:
            if (
                not isinstance(item, dict)
                or "name" not in item
                or "default" not in item
            ):
                print(item)
                raise ValidationError(
                    "Each item in the JSON list should be a dict with a name and a default value."
                )

    def save(self, *args, **kwargs):
        self.validate_json_data()
        super(BaseType, self).save(*args, **kwargs)


class OptimizerType(BaseType):
    keras_native_optimizer = models.BooleanField(default=False)


class CallbackType(BaseType):
    keras_native_callback = models.BooleanField(default=False)
    registers_metrics = models.TextField(null=True)


class LossType(BaseType):
    keras_native_loss = models.BooleanField(default=False)


class MetricType(BaseType):
    keras_native_metric = models.BooleanField(default=False)


class NetworkLayerType(BaseType):
    keras_native_layer = models.BooleanField(default=False)


class ActivationFunctionType(BaseType):
    keras_native_activation = models.BooleanField(default=False)


class TypeInstance(models.Model):
    additional_arguments = models.JSONField()

    class Meta:
        abstract = True

    def get_instance(self):
        raise NotImplementedError(
            "This should return an object of this type, with all the parameters"
        )

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
            except:
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
        super(TypeInstance, self).save(*args, **kwargs)

    def print_all_fields(self):
        print(self._meta.fields)


# TODO i guess this can all go into a basetype? with the srgument keras_native
