from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone as tz
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView, DeleteView

from naso.models.page import PageSetup
from runs.models.Training import NetworkTraining


class ListRuns(TemplateView):
    template_name = "runs/runs_list.html"
    page = PageSetup(title="Experimente", description="Liste")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        training_runs = reversed(NetworkTraining.objects.all())
        self.context["network_training_data"] = training_runs
        return self.render_to_response(self.context)


def delete_run(request, pk):
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(NetworkTraining, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})
