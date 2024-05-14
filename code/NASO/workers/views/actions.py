from workers.models.celery_workers import CeleryWorker
from django.http import JsonResponse


def delete_worker(request, pk):
    worker = CeleryWorker.objects.get(pk=pk)
    worker.delete()
    return JsonResponse({"success": True, "id": pk}, safe=False)
