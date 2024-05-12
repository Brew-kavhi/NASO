from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from runs.models.training import NetworkTraining
from runs.views.softdelete import harddelete_run, undelete_run


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def rate_run(request, pk):
    """
    This view rates a run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    run = NetworkTraining.objects.get(pk=pk)
    rate = request.data.get("rate")
    if rate != "":
        run.rate = rate
    else:
        run.rate = 0
    run.save()
    return Response({"success": True})


class MetricAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, format=None):
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
        return JsonResponse(data, safe=False)

    def post(self, request, pk, format=None):
        run = NetworkTraining.objects.get(pk=pk)
        data = request.data

        return Response(data, status=status.HTTP_201_created)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hard_delete(request, pk):
    """
    This view finally deletes a run.

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    success = harddelete_run(pk, "tensorflow")
    return Response({"success": success, "id": pk})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def undelete(request, pk):
    """
    This view undoes the deletion of a run

    Args:
        request (Request): The request object.
        pk (int): The primary key of the run.

    Returns:
        Response: The response object with parameter 'success'
    """
    success = undelete_run(pk, "tensorflow")
    return Response({"success": success, "id": pk})
