from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from runs.models.training import NetworkTraining


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


def get_metrics_for_run(request, pk):
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
