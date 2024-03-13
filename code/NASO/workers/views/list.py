from django.views.generic.base import TemplateView

from workers.models.celery_workers import CeleryWorker
from naso.models.page import PageSetup
from workers.helper_scripts.celery import get_all_workers


class ListWorkers(TemplateView):
    template_name = "workers/list.html"
    page = PageSetup(title="Workers", description="List")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        get_all_workers()
        workers = reversed(CeleryWorker.objects.all())
        self.page.actions = []
        self.context["page"] = self.page.get_context()
        self.context["workers"] = workers
        return self.render_to_response(self.context)
