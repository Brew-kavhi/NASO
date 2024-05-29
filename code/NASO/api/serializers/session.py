from rest_framework import serializers


class UpdateSessionSerializer(serializers.Serializer):
    comparison = serializers.CharField()
    values = serializers.ListField(child=serializers.CharField())
    delete = serializers.BooleanField()
    run_type = serializers.CharField()

    def create(self, *args, **kwargs):
        # nothing to be done here
        raise NotImplementedError(
            "Session create in serializer is not yert implemented."
        )

    def update(self, *args, **kwargs):
        # nothing to be done here
        raise NotImplementedError(
            "Session update in serializer is not yert implemented."
        )
