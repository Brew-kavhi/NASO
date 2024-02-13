from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from neural_architecture.models.autokeras import AutoKerasRun
from runs.models.training import NetworkTraining


class ListRuns(TemplateView):
    template_name = "runs/runs_list.html"
    page = PageSetup(title="Experimente", description="Liste")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        training_runs = reversed(
            NetworkTraining.objects.all().only(
                "id",
                "naso_app_version",
                "hyper_parameters__optimizer__instance_type",
                "hyper_parameters__loss__instance_type",
                "final_metrics__metrics",
                "network_config__name",
                "fit_parameters__epochs",
                "fit_parameters__batch_size",
            )
        )
        autokeras_runs = reversed(
            AutoKerasRun.objects.all().only(
                "id",
                "naso_app_version",
                "model__tuner__tuner_type",
                "model__project_name",
                "metrics",
                "model__max_trials",
                "model__max_model_size",
                "model__objective",
            )
        )
        self.page.actions = []
        self.page.add_pageaction(reverse_lazy("runs:autokeras:new"), "Neuer Autokeras")
        self.page.add_pageaction(reverse_lazy("runs:new"), "Neuer Tensorflow")
        self.context["page"] = self.page.get_context()
        self.context["network_training_data"] = training_runs
        self.context["autokeras_runs"] = autokeras_runs
        self.context["network_training_ids"] = [
            run.id for run in NetworkTraining.objects.all().only("id")
        ]
        self.context["autokeras_run_ids"] = [
            keras_run.id for keras_run in AutoKerasRun.objects.all().only("id")
        ]
        return self.render_to_response(self.context)


def delete_run(request, pk):
    """
    Deletes a NetworkTraining object with the given primary key.

    Args:
        request: The HTTP request object.
        pk: The primary key of the NetworkTraining object to be deleted.

    Returns:
        A JSON response indicating the successful deletion of the object.
    """
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(NetworkTraining, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})


def delete_autokeras_run(request, pk):
    """
    Deletes an AutoKerasRun object and its associated folder.

    Args:
        request (HttpRequest): The HTTP request object.
        pk (int): The primary key of the AutoKerasRun object to be deleted.

    Returns:
        JsonResponse: A JSON response indicating the successful deletion of the object.

    Raises:
        Http404: If the AutoKerasRun object with the given primary key does not exist.
    """
    # Fetch the object or return a 404 response if it doesn't exist
    obj = get_object_or_404(AutoKerasRun, pk=pk)

    # Delete the object
    obj.delete()

    # Return a JSON response to indicate successful deletion
    return JsonResponse({"message": "Object deleted successfully", "id": pk})
