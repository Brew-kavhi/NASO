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
    TensorFlowModelType,
)


class Command(BaseCommand):
    """
    Loads all predefined activation functions, optimizers, loss function, and other utilities in the database.
    """

    help = "Loads all predefined activation functions, optimizers, loss function and so on in the database"

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

        load_keras_models()

        # next load pruning utilities:
        load_pruning_utilities()

        # load the energy callback:
        call, _ = CallbackType.objects.get_or_create(
            module_name="neural_architecture.NetworkCallbacks.energy_callback",
            name="EnergyCallback",
            keras_native_callback=False,
            required_arguments=[],
        )
        call.registers_metrics = '["power_consumption", "trial_power_consumption"]'
        call.save()

        self.stdout.write(
            self.style.SUCCESS(
                "Successfully installed all predefined neural definitions."
            )
        )

    def load_types(self, type_class, module_name):
        """
        Loads types from a specified module and saves them in the database.

        Args:
            type_class (class): The class representing the type in the database.
            module_name (str): The name of the module to import types from.

        Returns:
            None
        """
        module = importlib.import_module(module_name)  # blocks
        classes = inspect.getmembers(module, inspect.isclass)
        self.stdout.write(self.style.SUCCESS(f"Importing {module_name}"))
        for class_name, class_obj in classes:
            constructor = inspect.signature(class_obj.__init__)
            arguments = build_arguments(constructor.parameters.items())
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


def load_keras_models():
    _, _ = TensorFlowModelType.objects.get_or_create(
        module_name="keras.applications",
        name="MobileNet",
        description="""
        This function returns a Keras image classification model, optionally loaded with weights pre-trained on ImageNet.

        For image classification use cases, see this page for detailed examples.

        For transfer learning use cases, make sure to read the guide to transfer learning & fine-tuning.

        Note: each Keras Application expects a specific kind of input preprocessing. For MobileNet, call keras.applications.mobilenet.preprocess_input on your inputs before passing them to the model. mobilenet.preprocess_input will scale input pixels between -1 and 1.
        """,
        required_arguments=[
            {"name": "input_shape", "default": "None", "dtype": "tuple(int,int,int)"},
            {"name": "alpha", "default": "1.0", "dtype": "float"},
            {"name": "depth_multiplier", "default": "1", "dtype": "float"},
            {"name": "dropout", "default": "0.001", "dtype": "float"},
            {"name": "include_top", "default": "True", "dtype": "bool"},
            {"name": "weights", "default": "imagenet", "dtype": "str"},
            {"name": "pooling", "default": "None", "dtype": "ENUM(avg,max)"},
            {"name": "classes", "default": "1000", "dtype": "int"},
            {"name": "classifier_activation", "default": "softmax", "dtype": "str"},
        ],
    )

    _, _ = TensorFlowModelType.objects.get_or_create(
        module_name="keras.applications",
        name="VGG19",
        description="""
        For image classification use cases, see this page for detailed examples.

        For transfer learning use cases, make sure to read the guide to transfer learning & fine-tuning.

        The default input size for this model is 224x224.

        Note: each Keras Application expects a specific kind of input preprocessing. For VGG19, call keras.applications.vgg19.preprocess_input on your inputs before passing them to the model. vgg19.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
        """,
        required_arguments=[
            {"name": "include_top", "default": "True", "dtype": "bool"},
            {"name": "weights", "default": "imagenet", "dtype": "str"},
            {"name": "input_shape", "default": "None", "dtype": "tuple(int,int,int)"},
            {"name": "pooling", "default": "None", "dtype": "ENUM(avg,max)"},
            {"name": "classes", "default": "1000", "dtype": "int"},
            {"name": "classifier_activation", "default": "softmax", "dtype": "str"},
        ],
    )


def load_dataset_loaders():
    """
    Loads dataset loaders into the database.

    This function creates instances of DatasetLoader model and saves them into the database.
    The DatasetLoader objects represent different dataset loaders available for use in the application.

    Returns:
        None
    """
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
    """
    Loads pruning utilities into the database.

    This function creates or updates pruning method types, pruning schedule types,
    and pruning policy types in the database. It sets the required arguments for each type.

    Returns:
        None
    """
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


def build_arguments(constructor_parameters):
    arguments = []
    for param_name, param in constructor_parameters:
        if not param_name == "self" and not param_name == "kwargs":
            if (
                param.default is not inspect.Parameter.empty
                or param.annotation is not inspect.Parameter.empty
            ):
                arguments.append(
                    {
                        "name": param_name,
                        "default": param.default
                        if param.default is not inspect.Parameter.empty
                        else "",
                        "dtype": get_argument_type(param),
                    }
                )
            else:
                arguments.append(
                    {"name": param_name, "default": "", "dtype": "unknown"}
                )
    return arguments


def get_argument_type(parameter):
    if parameter.annotation is inspect.Parameter.empty:
        return type(parameter.default).__name__
    param_type = type(parameter.annotation).__name__
    if param_type == "type":
        # this parwmeter is of a standard type:
        return parameter.annotation.__name__
    # else it is some kind of typing.Union or typing.Optional
    return str(parameter.annotation)
