import json

from django.http import HttpResponse
from django.views.generic.base import TemplateView, View
from neural_architecture.models.AutoKeras import AutoKerasRun

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

    def get(self, request,  *args, **kwargs):
        run = AutoKerasRun.objects.get(pk=kwargs['pk'])
        self.context['run'] = run
        print(kwargs['pk'])
        return self.render_to_response(self.context)
