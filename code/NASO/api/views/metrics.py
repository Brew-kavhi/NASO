import time

import django
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serializers.training import TrainingMetricSerializer
from helper_scripts.database import lock_safe_db_operation
from inference.models.inference import Inference
from runs.models.training import NetworkTraining, TrainingMetric


class TensorflowMetricAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, is_prediction=0):
        """
        This view returns the metrics for a run.

        Args:
            request (Request): The request object.
            pk (int): The primary key of the run.

        Returns:
            JsonResponse: Array of metrics for this run
        """
        run = NetworkTraining.objects.get(pk=pk)
        metrics = run.trainingmetric_set.all()
        data = []
        for metric in metrics:
            data.append(metric.metrics[0])
        return Response(data, status=status.HTTP_200_OK)

    def post(self, request, pk, is_prediction=0):
        run = NetworkTraining.objects.get(pk=pk)
        data = request.data
        serialized_data = TrainingMetricSerializer(data=data)
        if serialized_data.is_valid(True):
            metric = TrainingMetric(
                epoch=serialized_data.validated_data.get("epoch"),
                neural_network=run,
                metrics=serialized_data.validated_data.get("metrics"),
            )
            lock_safe_db_operation(metric.save)
            if serialized_data.is_final_metric():
                run.final_metrics = metric
                run.save()
            if is_prediction == 1:
                run.prediction_metrics.add(metric)
                run.save()

            return Response(data, status=status.HTTP_201_CREATED)
        return Response(serialized_data.errors, status=status.HTTP_400_BAD_REQUEST)


class MetricsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        inference = Inference.objects.get(pk=pk)
        metrics = inference.prediction_metrics.all()
        data = [metric.metrics[0] for metric in metrics]
        return Response(data, status=status.HTTP_200_OK)

    def post(self, request, pk):
        inference = Inference.objects.get(pk=pk)
        serialized_data = TrainingMetricSerializer(data=request.data)
        if serialized_data.is_valid(True):
            metric = TrainingMetric(
                epoch=serialized_data.validated_data.get("epoch"),
                metrics=serialized_data.validated_data.get("metrics"),
            )
            lock_safe_db_operatation(metric.save)
            inference.prediction_metrics.add(metric)
            inference.save()

            return Response(serialized_data.data, status=status.HTTP_201_CREATED)
        return Response(serialized_data.errors, status=status.HTTP_400_BAD_REQUEST)
