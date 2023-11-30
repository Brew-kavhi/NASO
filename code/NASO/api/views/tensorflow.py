from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from runs.models.training import NetworkTraining


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def rate_run(request, pk):
    run = NetworkTraining.objects.get(pk=pk)
    rate = request.data.get("rate")
    run.rate = rate
    run.save()
    return Response({"success": True})


def get_metrics_for_run(request, pk):
    run = NetworkTraining.objects.get(pk=pk)
    metrics = run.trainingmetric_set.all()
    data = []
    for metric in metrics:
        data.append(metric.metrics[0])
    return JsonResponse(data, safe=False)
