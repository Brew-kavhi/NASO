from rest_framework import serializers

from api.serializers.types import (
    PruningMethodTypeSerializer,
    PruningPolicyTypeSerializer,
    PruningScheduleTypeSerializer,
)
from neural_architecture.models.model_optimization import (
    PruningMethod,
    PruningPolicy,
    PruningSchedule,
)


class PruningMethodSerializer(serializers.ModelSerializer):
    instance_type = PruningMethodTypeSerializer()

    class Meta:
        model = PruningMethod
        fields = "__all__"


class PruningScheduleSerializer(serializers.ModelSerializer):
    instance_type = PruningScheduleTypeSerializer()

    class Meta:
        model = PruningSchedule
        fields = "__all__"


class PruningPolicySerializer(serializers.ModelSerializer):
    instance_type = PruningPolicyTypeSerializer()

    class Meta:
        model = PruningPolicy
        fields = "__all__"
