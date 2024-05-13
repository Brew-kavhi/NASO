from rest_framework import serializers

from api.serializers.architecture import (
    CallbackFunctionSerializer,
    ClusterableNetworkSerializer,
    LossFunctionSerializer,
)
from api.serializers.training import DatasetSerializer
from api.serializers.types import (
    AutoKerasNodeTypeSerializer,
    AutoKerasTunerTypeSerializer,
)
from neural_architecture.models.autokeras import (
    AutoKerasModel,
    AutoKerasNode,
    AutoKerasRun,
    AutoKerasTuner,
)


class AutoKerasNodeSerializer(serializers.ModelSerializer):
    node_type = AutoKerasNodeTypeSerializer()

    class Meta:
        model = AutoKerasNode
        fields = "__all__"


class AutoKerasTunerSerializer(serializers.ModelSerializer):
    tuner_type = AutoKerasTunerTypeSerializer()

    class Meta:
        model = AutoKerasTuner
        fields = "__all__"


class AutoKerasModelSerializer(serializers.ModelSerializer):
    tuner = AutoKerasTunerSerializer()
    blocks = AutoKerasNodeSerializer(many=True)
    clustering_options = ClusterableNetworkSerializer()
    loss = LossFunctionSerializer()
    callbacks = CallbackFunctionSerializer(many=True)

    class Meta:
        model = AutoKerasModel
        fields = [
            "id",
            "project_name",
            "max_trials",
            "directory",
            "objective",
            "trial_folder",
            "max_model_size",
            "node_to_layer_id",
            "metric_weights",
            "epochs",
            "tuner",
            "blocks",
            "callbacks",
            "loss",
            "clustering_options",
        ]


class AutoKerasRunSerializer(serializers.ModelSerializer):
    model = AutoKerasModelSerializer()
    dataset = DatasetSerializer()

    class Meta:
        model = AutoKerasRun
        fields = [
            "gpu",
            "dataset",
            "model",
        ]
