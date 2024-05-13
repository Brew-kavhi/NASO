from rest_framework import serializers

from api.serializers.model_optimization import (
    PruningMethodSerializer,
    PruningPolicySerializer,
    PruningScheduleSerializer,
)
from api.serializers.types import (
    CallbackFunctionTypeSerializer,
    LossFunctionTypeSerializer,
    NetworkLayerTypeSerializer,
    OptimizerTypeSerializer,
)
from neural_architecture.models.architecture import NetworkConfiguration, NetworkLayer
from neural_architecture.models.model_optimization import ClusterableNetwork
from runs.models.training import CallbackFunction, LossFunction, Optimizer


class LossFunctionSerializer(serializers.ModelSerializer):
    instance_type = LossFunctionTypeSerializer()

    class Meta:
        model = LossFunction
        fields = "__all__"


class CallbackFunctionSerializer(serializers.ModelSerializer):
    instance_type = CallbackFunctionTypeSerializer()

    class Meta:
        model = CallbackFunction
        fields = "__all__"


class ClusterableNetworkSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClusterableNetwork
        fields = "__all__"


class OptimizerSerializer(serializers.ModelSerializer):
    instance_type = OptimizerTypeSerializer()

    class Meta:
        model = Optimizer
        fields = "__all__"


class NetworkLayerSerializer(serializers.ModelSerializer):
    layer_type = NetworkLayerTypeSerializer()

    class Meta:
        model = NetworkLayer
        fields = "__all__"


class NetworkConfigurationSerializer(serializers.ModelSerializer):
    clustering_options = ClusterableNetworkSerializer()
    layers = NetworkLayerSerializer(many=True)
    pruning_method = PruningMethodSerializer()
    pruning_schedule = PruningScheduleSerializer()
    pruning_policy = PruningPolicySerializer()

    class Meta:
        model = NetworkConfiguration
        fields = "__all__"
