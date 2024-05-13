from rest_framework import serializers

from api.serializers.architecture import CallbackFunctionSerializer
from neural_architecture.models.dataset import Dataset, DatasetLoader
from runs.models.training import EvaluationParameters, FitParameters


class EvaluationParametersSerializer(serializers.ModelSerializer):
    callbacks = CallbackFunctionSerializer(many=True)

    class Meta:
        model = EvaluationParameters
        fields = "__all__"


class FitParametersSerializer(serializers.ModelSerializer):
    callbacks = CallbackFunctionSerializer(many=True)

    class Meta:
        model = FitParameters
        fields = "__all__"


class DatasetLoaderSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetLoader
        fields = "__all__"


class DatasetSerializer(serializers.ModelSerializer):
    dataset_loader = DatasetLoaderSerializer()

    class Meta:
        model = Dataset
        fields = "__all__"


class TrainingMetricSerializer(serializers.Serializer):
    epoch = serializers.IntegerField()
    metrics = serializers.JSONField()

    def is_valid(self, raise_exception=False):
        valid = super().is_valid(raise_exception=raise_exception)

        if not valid:
            errors = self.errors
            if raise_exception:
                raise serializers.ValidationError(errors)
        metrics = self.validated_data.get("metrics")
        if not isinstance(metrics, list):
            if raise_exception:
                raise serializers.ValidationError(
                    "JSON data should be a list of metrics"
                )
            valid = False

        for item in metrics:
            if not isinstance(item, dict) or "metrics" not in item:
                if raise_exception:
                    raise serializers.ValidationError(
                        {
                            "metrics": """Each item in the JSON list for TrainingMetric.metrics should be an 
                        object with a "metrics" attribute."""
                        }
                    )
                valid = False
            if "final_metric" not in item:
                valid = False
                if raise_exception:
                    raise serializers.ValidationError(
                        {"metrics": "Should contain final_metric"}
                    )
            if "run_id" not in item:
                valid = False
                if raise_exception:
                    raise serializers.ValidationError(
                        {"metrics": "Should contain run_id"}
                    )
            if "current" not in item:
                valid = False
                if raise_exception:
                    raise serializers.ValidationError(
                        {"metrics": "Should contain current"}
                    )
            if "time" not in item:
                valid = False
                if raise_exception:
                    raise serializers.ValidationError("Should contain time")
            if not isinstance(item["metrics"], dict):
                if raise_exception:
                    raise serializers.ValidationError(
                        {
                            "metrics": "TrainingMetrics.metrics.metrics should a dictionary of metricsname and value"
                        }
                    )
                valid = False
        return valid

    def is_final_metric(self):
        metrics = self.validated_data.get("metrics")
        if "final_metrics" in metrics[0] and metrics[0]["final_metrics"]:
            return True
        return False
