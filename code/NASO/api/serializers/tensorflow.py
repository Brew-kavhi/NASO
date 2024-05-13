from rest_framework import serializers

from api.serializers.architecture import (
    ClusterableNetworkSerializer,
    LossFunctionSerializer,
    NetworkConfigurationSerializer,
    OptimizerSerializer,
)
from api.serializers.model_optimization import (
    PruningMethodSerializer,
    PruningPolicySerializer,
    PruningScheduleSerializer,
)
from api.serializers.training import (
    DatasetSerializer,
    EvaluationParametersSerializer,
    FitParametersSerializer,
)
from api.serializers.types import TensorFlowModelTypeSerializer
from runs.models.training import (
    NetworkHyperparameters,
    NetworkTraining,
    TensorFlowModel,
)


class TensorFlowModelSerializer(serializers.ModelSerializer):
    instance_type = TensorFlowModelTypeSerializer()
    clustering_options = ClusterableNetworkSerializer()
    pruning_method = PruningMethodSerializer()
    pruning_schedule = PruningScheduleSerializer()
    pruning_policy = PruningPolicySerializer()

    class Meta:
        model = TensorFlowModel

        fields = "__all__"


class NetworkHyperparametersSerializer(serializers.ModelSerializer):
    optimizer = OptimizerSerializer()
    loss = LossFunctionSerializer()

    class Meta:
        model = NetworkHyperparameters
        fields = "__all__"


class NetworkTrainingSerializer(serializers.ModelSerializer):
    tensorflow_model = TensorFlowModelSerializer()
    network_config = NetworkConfigurationSerializer()
    hyper_parameters = NetworkHyperparametersSerializer()
    evaluation_parameters = EvaluationParametersSerializer()
    fit_parameters = FitParametersSerializer()
    dataset = DatasetSerializer()

    class Meta:
        model = NetworkTraining
        fields = [
            "network_config",
            "tensorflow_model",
            "hyper_parameters",
            "evaluation_parameters",
            "fit_parameters",
            "dataset",
            "description",
            "gpu",
            "worker",
            "compute_device",
        ]
