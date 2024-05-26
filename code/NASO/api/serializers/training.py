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
            if not isinstance(item, dict):
                if raise_exception:
                    raise serializers.ValidationError(
                        {
                            "metrics": """Each item in the JSON list for TrainingMetric.metrics should be an 
                        object with a "metrics" attribute."""
                        }
                    )
                valid = False
            valid = self.validate_key_metyric("metrics", item, raise_exception)
            valid = self.validate_key_metyric("final_metric", item, raise_exception)
            valid = self.validate_key_metyric("run_id", item, raise_exception)
            valid = self.validate_key_metyric("current", item, raise_exception)
            valid = self.validate_key_metyric("time", item, raise_exception)
            if not isinstance(item["metrics"], dict):
                if raise_exception:
                    raise serializers.ValidationError(
                        {
                            "metrics": "TrainingMetrics.metrics.metrics should a dictionary of metricsname and value"
                        }
                    )
                valid = False
        return valid

    def validate_key_metyric(self, key: str, item: dict, raise_exception: bool) -> bool:
        valid = True
        if key not in item:
            valid = False
            if raise_exception:
                raise serializers.ValidationError({"metrics": f"Should contain {key}"})
        return valid

    def is_final_metric(self):
        metrics = self.validated_data.get("metrics")
        if "final_metrics" in metrics[0] and metrics[0]["final_metrics"]:
            return True
        return False

    def create(self, *args, **kwargs):
        # nothing to be done here
        raise NotImplementedError(
            "TrainingMetric create in serializer is not yert implemented."
        )

    def update(self, *args, **kwargs):
        # nothing to be done here
        raise NotImplementedError(
            "TrainingMetric update in serializer is not yert implemented."
        )
