import os
import shutil

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


class ListRuns(TemplateView):
    template_name = "runs/runs_list.html"
    page = PageSetup(title="Experimente", description="Liste")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        training_runs = reversed(NetworkTraining.objects.all())
        autokeras_runs = reversed(AutoKerasRun.objects.all())
        self.context["network_training_data"] = training_runs
        self.context["autokeras_runs"] = autokeras_runs
        return self.render_to_response(self.context)


def delete_run(request, pk):
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(NetworkTraining, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})


def delete_autokeras_run(request, pk):
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(AutoKerasRun, pk=pk)

    # Delete the folder:
    if len(obj.model.directory) > 0:
        folder = "auto_model/" + obj.model.directory
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})
