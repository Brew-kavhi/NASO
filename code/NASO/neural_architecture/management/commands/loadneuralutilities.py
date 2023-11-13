import importlib
import inspect

from django.core.management.base import BaseCommand

from neural_architecture.models.autokeras import AutoKerasNodeType, AutoKerasTunerType
from neural_architecture.models.dataset import DatasetLoader
from neural_architecture.models.model_optimization import (
    PruningMethodTypes,
    PruningPolicyTypes,
    PruningScheduleTypes,
)
from neural_architecture.models.types import (
    CallbackType,
    LossType,
    MetricType,
    NetworkLayerType,
    OptimizerType,
)


class Command(BaseCommand):
    help = "Loads all predefined activation functions, optimizers, loss function and so on in the database"

    def build_arguments(self, constructor_parameters):
        arguments = []
        for param_name, param in constructor_parameters:
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
                        {"name": param_name, "default": "", "dtype": "unknown"}
                    )
        return arguments

    def handle(self, *args, **options):
        # create all the objects here
        # for example the RegisteredActivationFunctions: tanh, ...
        # load optimizers here

        self.load_types(OptimizerType, "tensorflow.keras.optimizers")

        # now the losses:
        self.load_types(LossType, "tensorflow.keras.losses")

        self.load_types(MetricType, "tensorflow.keras.metrics")

        self.stdout.write(self.style.SUCCESS("Importing layers"))
        self.load_types(NetworkLayerType, "tensorflow.keras.layers")

        self.stdout.write(self.style.SUCCESS("Importing Callbacks"))
        self.load_types(CallbackType, "tensorflow.keras.callbacks")

        self.stdout.write(self.style.SUCCESS("Importing Autokeras Blocks"))
        for module in [
            "blocks.basic",
            "blocks.wrapper",
            "blocks.heads",
            "blocks.preprocessing",
            "blocks.reduction",
            "nodes",
        ]:
            self.load_types(AutoKerasNodeType, "autokeras." + module)

        self.load_types(AutoKerasTunerType, "autokeras.tuners")

        # load the Datasetloaders, that come preconfigured with the system:
        load_dataset_loaders()

        # next load pruning utilities:
        load_pruning_utilities()

        # load the energy callback:
        call, _ = CallbackType.objects.get_or_create(
            module_name="neural_architecture.NetworkCallbacks.energy_callback",
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

    def load_types(self, type_class, module_name):
        module = importlib.import_module(module_name)  # blocks
        classes = inspect.getmembers(module, inspect.isclass)
        self.stdout.write(self.style.SUCCESS(f"Importing {module_name}"))
        for class_name, class_obj in classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = self.build_arguments(constructor.parameters.items())
            try:
                type_instance, _ = type_class.objects.get_or_create(
                    module_name=module_name,
                    name=class_name,
                )
                type_instance.required_arguments = arguments
                type_instance.save()
            except Exception as exc:
                self.stdout.write(
                    self.style.ERROR(
                        f"Class {class_name} from {module_name}.{class_name} has errors: {exc}"
                    )
                )


def load_dataset_loaders():
    _, _ = DatasetLoader.objects.get_or_create(
        module_name="neural_architecture.models.dataset",
        class_name="TensorflowDatasetLoader",
        name="Tensorflow Datasets",
        description="These are all the datasets that are available in tensorflow_datasets.",
    )
    _, _ = DatasetLoader.objects.get_or_create(
        module_name="neural_architecture.models.dataset",
        class_name="SkLearnDatasetLoader",
        name="SkLearn Datasets",
        description="These are all the datasets that are available in sklearn.",
    )


def load_pruning_utilities():
    method, _ = PruningMethodTypes.objects.get_or_create(
        module_name="tensorflow_model_optimization.sparsity.keras",
        name="prune_low_magnitude",
    )
    method.required_arguments = [
        {
            "name": "block_size",
            "default": "(1,1)",
            "dtype": "tuple(int, int)",
        },
        {
            "name": "block_pooling_type",
            "default": "AVG",
            "dtype": "ENUM(AVG, MAX)",
        },
        {
            "name": "sparsity_m_by_n",
            "default": None,
            "dtype": "tuple(int, int)",
        },
    ]
    method.save()

    scheduler, _ = PruningScheduleTypes.objects.get_or_create(
        module_name="tensorflow_model_optimization.sparsity.keras",
        name="ConstantSparsity",
    )
    scheduler.required_arguments = [
        {
            "name": "target_sparsity",
            "default": "undefined",
            "dtype": "float",
        },
        {
            "name": "begin_step",
            "default": "0",
            "dtype": "int",
        },
        {
            "name": "end_step",
            "default": "-1",
            "dtype": "int",
        },
        {
            "name": "frequency",
            "default": "100",
            "dtype": "int",
        },
    ]
    scheduler.save()
    scheduler, _ = PruningScheduleTypes.objects.get_or_create(
        module_name="tensorflow_model_optimization.sparsity.keras",
        name="PolynomialDecay",
    )
    scheduler.required_arguments = [
        {
            "name": "initial_sparsity",
            "default": "undefined",
            "dtype": "float",
        },
        {
            "name": "final_sparsity",
            "default": "undefined",
            "dtype": "float",
        },
        {
            "name": "begin_step",
            "default": "0",
            "dtype": "int",
        },
        {
            "name": "end_step",
            "default": "undefined",
            "dtype": "int",
        },
        {
            "name": "power",
            "default": "3",
            "dtype": "int",
        },
        {
            "name": "frequency",
            "default": "100",
            "dtype": "int",
        },
    ]
    scheduler.save()
    PruningPolicyTypes.objects.get_or_create(
        module_name="tensorflow_model_optimization.sparsity.keras",
        name="PruneForLatencyOnXNNPack",
        required_arguments=[],
    )
