import importlib
import inspect
import autokeras

from neural_architecture.models.AutoKeras import AutoKerasNodeType

from django.core.management.base import BaseCommand

from neural_architecture.models.Types import (LossType, MetricType,
                                              NetworkLayerType, OptimizerType)


class Command(BaseCommand):
    help = "Loads all predefined activation functions, optimizers, loss function and so on in the database"

    def handle(self, *args, **options):
        # create all the objects here
        # for example the RegisteredActivationFunctions: tanh, ...
        # load optimizers here

        optimizers_module = importlib.import_module("tensorflow.keras.optimizers")
        optimizer_classes = inspect.getmembers(optimizers_module, inspect.isclass)

        for class_name, class_obj in optimizer_classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = []
            for param_name, param in constructor.parameters.items():
                if not param_name == "self" and not param_name == "kwargs":
                    if param.default is not inspect.Parameter.empty:
                        arguments.append(
                            {
                                "name": param_name,
                                "default": param.default,
                                "dtype": type(param.default).__name__,
                            }
                        )
                    else:
                        {
                            arguments.append(
                                {"name": param_name, "default": "", "type": "unknown"}
                            )
                        }
            try:
                _, _ = OptimizerType.objects.get_or_create(
                    module_name="tensorflow.keras.optimizers",
                    name=class_name,
                    keras_native_optimizer=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                print(f"Class: {class_name} has error {e}")

        # now the losses:
        losses_module = importlib.import_module("tensorflow.keras.losses")
        loss_classes = inspect.getmembers(losses_module, inspect.isclass)

        for class_name, class_obj in loss_classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = []
            for param_name, param in constructor.parameters.items():
                if not param_name == "self" and not param_name == "kwargs":
                    if param.default is not inspect.Parameter.empty:
                        arguments.append(
                            {
                                "name": param_name,
                                "default": param.default,
                                "dtype": type(param.default).__name__,
                            }
                        )
                    else:
                        {
                            arguments.append(
                                {"name": param_name, "default": "", "type": "unknown"}
                            )
                        }
            try:
                _, _ = LossType.objects.get_or_create(
                    module_name="tensorflow.keras.losses",
                    name=class_name,
                    keras_native_loss=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                print(f"Class: {class_name} has error {e}")

        metrics_module = importlib.import_module("tensorflow.keras.metrics")
        metric_classes = inspect.getmembers(metrics_module, inspect.isclass)

        for class_name, class_obj in metric_classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = []
            for param_name, param in constructor.parameters.items():
                if not param_name == "self" and not param_name == "kwargs":
                    if param.default is not inspect.Parameter.empty:
                        arguments.append(
                            {
                                "name": param_name,
                                "default": param.default,
                                "dtype": type(param.default).__name__,
                            }
                        )
                    else:
                        {
                            arguments.append(
                                {"name": param_name, "default": "", "type": "unknown"}
                            )
                        }
            try:
                _, _ = MetricType.objects.get_or_create(
                    module_name="tensorflow.keras.metrics",
                    name=class_name,
                    keras_native_metric=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                print(f"Class: {class_name} has error {e}")

        self.stdout.write(self.style.SUCCESS(f"Importing layers"))
        # TODO Add the layers and the activation functions here
        layers_module = importlib.import_module("tensorflow.keras.layers")
        layer_classes = inspect.getmembers(layers_module, inspect.isclass)

        for class_name, class_obj in layer_classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = []
            for param_name, param in constructor.parameters.items():
                if not param_name == "self" and not param_name == "kwargs":
                    if param.default is not inspect.Parameter.empty:
                        arguments.append(
                            {
                                "name": param_name,
                                "default": param.default,
                                "dtype": type(param.default).__name__,
                            }
                        )
                    else:
                        {
                            arguments.append(
                                {"name": param_name, "default": "", "type": "unknown"}
                            )
                        }
            try:
                _, _ = NetworkLayerType.objects.get_or_create(
                    module_name="tensorflow.keras.layers",
                    name=class_name,
                    keras_native_layer=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Class {class_name} has errors: {e}")
                )

        for module in ["blocks.basic","blocks.heads", "blocks.preprocessing", "blocks.reduction", "nodes"]
        autokeras_module = importlib.import_module('autokeras')# blocks
        autokeras_nodes = inspect.getmembers(autokeras_module, inspect.isclass)
        
        for class_name, class_obj in layer_classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = []
            for param_name, param in constructor.parameters.items():
                if not param_name == "self" and not param_name == "kwargs":
                    if param.default is not inspect.Parameter.empty:
                        arguments.append(
                            {
                                "name": param_name,
                                "default": param.default,
                                "dtype": type(param.default).__name__,
                            }
                        )
                    else:
                        {
                            arguments.append(
                                {"name": param_name, "default": "", "type": "unknown"}
                            )
                        }
            try:
                _, _ = NetworkLayerType.objects.get_or_create(
                    module_name="tensorflow.keras.layers",
                    name=class_name,
                    keras_native_layer=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Class {class_name} has errors: {e}")
                )

        self.stdout.write(
            self.style.SUCCESS(
                "Successfully installed all predefined nueral definitions."
            )
        )
