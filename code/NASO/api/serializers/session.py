from rest_framework import serializers


class UpdateSessionSerializer(serializers.Serializer):
    comparison = serializers.CharField()
    values = serializers.ListField(child=serializers.CharField())
    delete = serializers.BooleanField()
    run_type = serializers.CharField()
