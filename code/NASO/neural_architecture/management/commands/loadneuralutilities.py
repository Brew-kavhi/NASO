import importlib
import inspect

from django.core.management.base import BaseCommand

from neural_architecture.models.AutoKeras import (AutoKerasNodeType,
                                                  AutoKerasTunerType)
from neural_architecture.models.Dataset import DatasetLoader
from neural_architecture.models.Types import (CallbackType, LossType,
                                              MetricType, NetworkLayerType,
                                              OptimizerType)


class Command(BaseCommand):
    help = "Loads all predefined activation functions, optimizers, loss function and so on in the database"

    def handle(self, *args, **options):
        # create all the objects here
        # for example the RegisteredActivationFunctions: tanh, ...
        # load optimizers here

        # TODO refactor .  this function to use multiple more clear functions with a single purpose
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
                        arguments.append(
                            {"name": param_name, "default": "", "type": "unknown"}
                        )
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
                        arguments.append(
                            {"name": param_name, "default": "", "type": "unknown"}
                        )
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
                        arguments.append(
                            {"name": param_name, "default": "", "type": "unknown"}
                        )
            try:
                _, _ = MetricType.objects.get_or_create(
                    module_name="tensorflow.keras.metrics",
                    name=class_name,
                    keras_native_metric=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                print(f"Class: {class_name} has error {e}")

        # TODO Add the activation functions here
        self.stdout.write(self.style.SUCCESS(f"Importing layers"))
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
                        arguments.append(
                            {"name": param_name, "default": "", "type": "unknown"}
                        )
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

        self.stdout.write(self.style.SUCCESS(f"Importing Callbacks"))
        callbacks_module = importlib.import_module("tensorflow.keras.callbacks")
        callback_classes = inspect.getmembers(callbacks_module, inspect.isclass)

        for class_name, class_obj in callback_classes:
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
                        arguments.append(
                            {"name": param_name, "default": "", "type": "unknown"}
                        )
            try:
                _, _ = CallbackType.objects.get_or_create(
                    module_name="tensorflow.keras.callbacks",
                    name=class_name,
                    keras_native_callback=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Class {class_name} has errors: {e}")
                )

        self.stdout.write(self.style.SUCCESS(f"Importing Autokeras Blocks"))
        for module in [
            "blocks.basic",
            "blocks.wrapper",
            "blocks.heads",
            "blocks.preprocessing",
            "blocks.reduction",
            "nodes",
        ]:
            autokeras_module = importlib.import_module("autokeras." + module)  # blocks
            autokeras_nodes = inspect.getmembers(autokeras_module, inspect.isclass)

            for class_name, class_obj in autokeras_nodes:
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
                            arguments.append(
                                {
                                    "name": param_name,
                                    "default": "",
                                    "type": "unknown",
                                }
                            )
                try:
                    _, _ = AutoKerasNodeType.objects.get_or_create(
                        module_name="autokeras." + module,
                        name=class_name,
                        autokeras_type=module,
                        required_arguments=arguments,
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"Class {class_name} from {'autokeras.' + module} has errors: {e}"
                        )
                    )

        tuners_module = importlib.import_module("autokeras.tuners")  # blocks
        tuners = inspect.getmembers(tuners_module, inspect.isclass)

        for class_name, class_obj in tuners:
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
                        arguments.append(
                            {"name": param_name, "default": "", "type": "unknown"}
                        )
            try:
                _, _ = AutoKerasTunerType.objects.get_or_create(
                    module_name="autokeras.tuners",
                    name=class_name,
                    native_tuner=True,
                    required_arguments=arguments,
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f"Class {class_name} from {'autokeras.' + module} has errors: {e}"
                    )
                )

        # load the Datasetloaders, that come preconfigured with the system:
        load_dataset_loaders()

        # load the energy callback:
        call, _ = CallbackType.objects.get_or_create(
            module_name="neural_architecture.NetworkCallbacks.EnergyCallback",
            name="EnergyCallback",
            keras_native_callback=False,
            required_arguments=[],
        )
        call.registers_metrics = '["energy_consumption", "trial_energy_consumption"]'
        call.save()

        self.stdout.write(
            self.style.SUCCESS(
                "Successfully installed all predefined nueral definitions."
            )
        )


def load_dataset_loaders():
    DatasetLoader.objects.get_or_create(
        module_name="neural_architecture.models.Dataset",
        class_name="TensorflowDatasetLoader",
        name="Tensorflow Datasets",
        description="These are all the datasets that are available in tensorflow_datasets.",
    )
    DatasetLoader.objects.get_or_create(
        module_name="neural_architecture.models.Dataset",
        class_name="SkLearnDatasetLoader",
        name="SkLearn Datasets",
        description="These are all the datasets that are available in sklearn.",
    )
