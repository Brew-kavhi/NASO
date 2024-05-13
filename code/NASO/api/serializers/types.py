from rest_framework import serializers

from neural_architecture.models.model_optimization import (
    PruningMethodTypes,
    PruningPolicyTypes,
    PruningScheduleTypes,
)
from neural_architecture.models.types import (
    AutoKerasNodeType,
    AutoKerasTunerType,
    CallbackType,
    LossType,
    NetworkLayerType,
    OptimizerType,
    TensorFlowModelType,
)


class LossFunctionTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = LossType
        fields = "__all__"


class OptimizerTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = OptimizerType
        fields = "__all__"


class CallbackFunctionTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = CallbackType
        fields = "__all__"


class NetworkLayerTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = NetworkLayerType
        fields = "__all__"


class AutoKerasNodeTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = AutoKerasNodeType
        fields = "__all__"


class AutoKerasTunerTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = AutoKerasTunerType
        fields = "__all__"


class PruningMethodTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = PruningMethodTypes
        fields = "__all__"


class PruningScheduleTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = PruningScheduleTypes
        fields = "__all__"


class PruningPolicyTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = PruningPolicyTypes
        fields = "__all__"


class TensorFlowModelTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = TensorFlowModelType

        fields = "__all__"
