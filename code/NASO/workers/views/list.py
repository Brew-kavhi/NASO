from django.views.generic.base import TemplateView

from api.views.celery import is_worker_busy
from naso.models.page import PageSetup
from workers.helper_scripts.celery import get_all_workers
from workers.models.celery_workers import CeleryWorker


class ListWorkers(TemplateView):
    template_name = "workers/list.html"
    page = PageSetup(title="Workers", description="List")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        get_all_workers()
        workers = reversed(CeleryWorker.objects.all())
        self.page.actions = []
        self.context["page"] = self.page.get_context()
        details = []
        for worker in workers:
            worker_details = {
                "id": worker.id,
                "hostname": worker.hostname,
                "queue_name": worker.queue_name,
                "active": worker.active,
                "last_ping": worker.last_ping,
                "last_active": worker.last_active,
                "concurrency": worker.concurrency,
                "devices": worker.devices,
                "tasks_executing": 0,
            }
            if worker.active:
                worker_details["tasks_executing"] = is_worker_busy(worker.hostname)
            details.append(worker_details)
        self.context["workers"] = details
        return self.render_to_response(self.context)
