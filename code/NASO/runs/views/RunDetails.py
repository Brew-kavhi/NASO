import json

from django.http import HttpResponse
from django.views.generic.base import TemplateView, View

from naso.celery import get_celery_task_state
from naso.models.page import PageSetup


class TrainingProgress(View):
    def get(self, request, task_id):
        context = get_celery_task_state(task_id)
        return HttpResponse(json.dumps(context), content_type="application/json")


class RunDetails(TemplateView):
    template_name = "runs/run_details.html"

    page = PageSetup(title="Experimente", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        return self.render_to_response(self.context)


class AutoKerasRunDetails(TemplateView):
    template_name = "runs/run_details.html"

    page = PageSetup(title="Experimente", description="Details")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        return self.render_to_response(self.context)
