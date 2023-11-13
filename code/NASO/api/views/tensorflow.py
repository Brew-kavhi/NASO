from django.http import JsonResponse

from runs.models.training import NetworkTraining


def get_metrics_for_run(request, pk):
    run = NetworkTraining.objects.get(pk=pk)
    metrics = run.trainingmetric_set.all()
    data = []
    for metric in metrics:
        data.append(metric.metrics[0])
    return JsonResponse(data, safe=False)
